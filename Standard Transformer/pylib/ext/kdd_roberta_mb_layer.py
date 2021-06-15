# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa multi-branch layers.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


def merge_branches(branch_outputs, branch_dropout, training):
    """Merge branches.
    
    branch_outputs: list of Tensor or tuple of Tensors
    """
    if not isinstance(branch_outputs[0], (tuple, list)):
        branch_outputs = [(t,) for t in branch_outputs]
    branch_output_lists = tuple(zip(*branch_outputs))

    N = len(branch_outputs)
    branch_selection = branch_outputs[0][0].new(N).fill_(1.0 / N)
    branch_selection_d = F.dropout(branch_selection, p=branch_dropout, training=training)

    merged_branch_outputs = []
    for branch_output_list_i in branch_output_lists:
        if branch_output_list_i[0] is None:
            merged_branch_outputs.append(None)
        else:
            branch_output_i = torch.stack(branch_output_list_i, dim=0)
            branch_selection_d_expanded = branch_selection_d[(slice(None),) + tuple(None for _ in range(branch_output_i.ndimension() - 1))]
            merged_branch_outputs.append(torch.mean(branch_selection_d_expanded * branch_output_i, dim=0))

    if len(merged_branch_outputs) == 1:
        return merged_branch_outputs[0]
    else:
        return tuple(merged_branch_outputs)


class TransformerMBSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        multi_branch_options: dict,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
    
        # Multi-branch options.
        self.num_pffn_branches = multi_branch_options['encoder_pffn_branches']
        self.num_branches = multi_branch_options['encoder_branches']
        self.branch_dropout = multi_branch_options['branch_dropout']
        self.pffn_branch_dropout = multi_branch_options['pffn_branch_dropout']
        self.enable_head_dropout = multi_branch_options['enable_head_dropout']
        self.join_pffn = multi_branch_options['join_pffn']

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.activation_dropout_module = FairseqDropout(
            activation_dropout, module_name=self.__class__.__name__
        )

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn_branches = nn.ModuleList([
            self.build_self_attention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                # head_dropout=self.branch_dropout if self.enable_head_dropout else None,   # [TODO]: Implement head dropout
            )
            for _ in range(self.num_branches)
        ])

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1_branches = nn.ModuleList([
            self.build_fc1(
                self.embedding_dim,
                ffn_embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            for _ in range(self.num_pffn_branches)
        ])
        self.fc2_branches = nn.ModuleList([
            self.build_fc2(
                ffn_embedding_dim,
                self.embedding_dim,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )
            for _ in range(self.num_pffn_branches)
        ])

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self,
        embed_dim,
        num_attention_heads,
        dropout,
        self_attention,
        q_noise,
        qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def self_attn(self, query, key, value, **kwargs):
        self_attn_outputs = [m(query, key, value, **kwargs) for m in self.self_attn_branches]
        return merge_branches(self_attn_outputs, self.branch_dropout, not self.enable_head_dropout and self.training)
    
    def fc1(self, x):
        fc1_outputs = [m(x) for m in self.fc1_branches]
        return merge_branches(fc1_outputs, self.pffn_branch_dropout, self.training)
    
    def fc2(self, x):
        fc2_outputs = [m(x) for m in self.fc2_branches]
        return merge_branches(fc2_outputs, self.pffn_branch_dropout, self.training)
    
    def fc1fc2(self, x):
        outputs = []
        for fc1, fc2 in zip(self.fc1_branches, self.fc2_branches):
            o = self.activation_fn(fc1(x))
            o = self.activation_dropout_module(o)
            o = fc2(o)
            outputs.append(o)
        return merge_branches(outputs, self.pffn_branch_dropout, self.training)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        if self.join_pffn:
            x = self.fc1fc2(x)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
