# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("sentence_ordinal")
class SentenceOrdinalCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name, num_classes):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.num_classes = num_classes

        

        # self.tril = torch.nn.Parameter(tril, requires_grad=False)
        # self.xx = torch.nn.Parameter(torch.zeros(1)) # Make pytorch happy https://github.com/pytorch/pytorch/blob/4d7abdbdadd440cb4b8412f1e309cae14a687b49/torch/nn/parallel/distributed.py#L393

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        
        # lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        # loss = F.nll_loss(lprobs, targets, reduction="sum")

        tril = torch.tril(torch.ones(self.num_classes, self.num_classes, device=targets.device))

        ordered_targets = torch.index_select(tril, 0, targets)
        loss = F.binary_cross_entropy_with_logits(logits, ordered_targets, reduction="sum") / 100


        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        #preds = logits.argmax(dim=1)

        pad = torch.zeros((sample_size, 1), device=logits.device)
        preds = torch.cat([(logits > 0), pad], dim=-1).argmin(dim=-1) - 1
        logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
