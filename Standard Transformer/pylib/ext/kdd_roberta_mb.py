# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
    FairseqDropout,
    LayerDropModuleList,
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params

import numpy as np

from .kdd_roberta_mb_layer import (
    merge_branches, TransformerMBSentenceEncoderLayer,
)

logger = logging.getLogger(__name__)


@register_model("kdd_roberta_mb")
class KDDRobertaMBModel(FairseqEncoderModel):
    @classmethod
    def hub_models(cls):
        return {
            "roberta.base": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz",
            "roberta.large": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz",
            "roberta.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.mnli.tar.gz",
            "roberta.large.wsc": "http://dl.fbaipublicfiles.com/fairseq/models/roberta.large.wsc.tar.gz",
        }

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

        # New arguments for multi-branch.
        parser.add_argument('--encoder-branches', type=int, metavar='N',
                            help='num encoder branches')
        parser.add_argument('--encoder-pffn-branches', type=int, metavar='N',
                            help='num encoder PFFN branches, default to --encoder-branches')
        parser.add_argument('--branch-dropout', type=float, metavar='D',
                            help='sets dropout of each branch')
        parser.add_argument('--enable-head-dropout', action='store_true',
                            help='enable head-level dropout in multihead attention')
        parser.add_argument('--pffn-branch-dropout', type=float, metavar='D',
                            help='sets dropout of each PFFN branch, default to --branch-dropout')
        parser.add_argument('--no-join-pffn', dest='join_pffn', action='store_false',
                            help='Join FC1 and FC2 of PFFN together')
        parser.add_argument('--join-layer', action='store_true',
                            help='Join the whole layer together')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = KDDRobertaMBEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):
        if classification_head_name is not None:
            features_only = True

        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = KDDRobertaMBClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        cls.upgrade_args(x["args"])

        logger.info(x["args"])
        return KDDRobertaMBHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


class KDDRobertaMBHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        # self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(
        self, sentence: str, *addl_sentences, no_separator=False
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = "<s> " + self.bpe.encode(sentence) + " </s>"
        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator else ""
            bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError(
                "tokens exceeds maximum length: {} > {}".format(
                    tokens.size(-1), self.model.max_positions()
                )
            )
        features, extra = self.model(
            tokens.to(device=self.device),
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra["inner_states"]
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        features = self.extract_features(tokens.to(device=self.device))
        x_mask = (tokens != self.model.encoder.sentence_encoder.padding_idx).to(self.device)
        logits = self.model.classification_heads[head](features, x_mask)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = "<mask>"
        assert (
            masked_token in masked_input and masked_input.count(masked_token) == 1
        ), "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(
            masked_token
        )

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (
            (" {0} ".format(masked_token))
            .join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans])
            .strip()
        )
        tokens = self.task.source_dictionary.encode_line(
            "<s> " + text_spans_bpe + " </s>",
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.model_eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(
            topk_predicted_token_bpe.split(" ")
        ):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith("\u2581"):
                predicted_token = " " + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(
                            " {0}".format(masked_token), predicted_token
                        ),
                        values[index].item(),
                        predicted_token,
                    )
                )
            else:
                topk_filled_outputs.append(
                    (
                        masked_input.replace(masked_token, predicted_token),
                        values[index].item(),
                        predicted_token,
                    )
                )
        return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(
            self.task, "disambiguate_pronoun"
        ), "roberta.disambiguate_pronoun() requires a model trained with the WSC task."
        with utils.model_eval(self.model):
            return self.task.disambiguate_pronoun(
                self.model, sentence, use_cuda=self.device.type == "cuda"
            )


class KDDRobertaMBLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class KDDRobertaMBClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class KDDRobertaMBEncoder(FairseqEncoder):
    """RoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        multi_branch_options = {
            'encoder_branches': args.encoder_branches,
            'encoder_pffn_branches': args.encoder_pffn_branches,
            'branch_dropout': args.branch_dropout,
            'enable_head_dropout': args.enable_head_dropout,
            'pffn_branch_dropout': args.pffn_branch_dropout,
            'join_pffn': args.join_pffn,
            'join_layer': args.join_layer,
        }
        self.sentence_encoder = TransformerMBSentenceEncoder(
            multi_branch_options,
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
        )
        args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

        self.lm_head = KDDRobertaMBLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {"inner_states": inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class TransformerMBSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.
    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).
    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens
    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        multi_branch_options: dict,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable

        # Multi-branch options.
        self.num_branches = multi_branch_options['encoder_branches']
        self.branch_dropout = multi_branch_options['branch_dropout']
        self.pffn_branch_dropout = multi_branch_options['pffn_branch_dropout']
        self.join_layer = multi_branch_options['join_layer']

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if self.layerdrop > 0.0:
            raise NotImplementedError('layerdrop > 0.0 not implemented in MB now')
        if self.join_layer:
            self.layers_branches = nn.ModuleList([])
            for _ in range(num_encoder_layers):
                layer_i_branches = nn.ModuleList([])
                self.layers_branches.append(layer_i_branches)
                for _ in range(self.num_branches):
                    layer_i_branches.append(self.build_transformer_sentence_encoder_layer(
                        embedding_dim=self.embedding_dim,
                        ffn_embedding_dim=ffn_embedding_dim,
                        num_attention_heads=num_attention_heads,
                        dropout=self.dropout_module.p,
                        attention_dropout=attention_dropout,
                        activation_dropout=activation_dropout,
                        activation_fn=activation_fn,
                        export=export,
                        q_noise=q_noise,
                        qn_block_size=qn_block_size,
                    ))
        else:
            self.layers = nn.ModuleList([])
            self.layers.extend([
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    multi_branch=True,
                    multi_branch_options=multi_branch_options,
                )
                for _ in range(num_encoder_layers)
            ])

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        *,
        multi_branch=False,
        multi_branch_options=None,
    ):
        if multi_branch:
            return TransformerMBSentenceEncoderLayer(
                multi_branch_options,
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
        return TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def layer_mb(self, layer_branches, x, padding_mask, attn_mask):
        layer_outputs = [m(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask) for m in layer_branches]
        return merge_branches(layer_outputs, self.branch_dropout, self.training)

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_tpu = tokens.device.type == "xla"

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not is_tpu and not padding_mask.any():
            padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # encoder layers
        if self.join_layer:
            for layer_branches in self.layers_branches:
                x, _ = self.layer_mb(layer_branches, x, padding_mask, attn_mask)
                if not last_state_only:
                    inner_states.append(x)
        else:
            for layer in self.layers:
                x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask)
                if not last_state_only:
                    inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


@register_model_architecture("kdd_roberta_mb", "kdd_roberta_mb")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.spectral_norm_classification_head = getattr(
        args, "spectral_nrom_classification_head", False
    )

    # Multi-branch.
    args.encoder_branches = getattr(args, 'encoder_branches', 4)
    args.encoder_pffn_branches = getattr(args, 'encoder_pffn_branches', args.encoder_branches)
    args.branch_dropout = getattr(args, 'branch_dropout', 0.1)
    args.enable_head_dropout = getattr(args, 'enable_head_dropout', False)
    args.pffn_branch_dropout = getattr(args, 'pffn_branch_dropout', args.branch_dropout)
    args.join_pffn = getattr(args, 'join_pffn', True)
    args.join_layer = getattr(args, 'join_layer', False)

    if args.enable_head_dropout:
        raise NotImplementedError('head dropout not implemented now')


@register_model_architecture("kdd_roberta_mb", "kdd_roberta_mb_base")
def roberta_mb_base_architecture(args):
    base_architecture(args)


@register_model_architecture("kdd_roberta_mb", "kdd_roberta_mb_large")
def roberta_mb_large_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)

