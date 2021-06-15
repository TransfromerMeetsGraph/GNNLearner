# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
import torch
from fairseq import utils, metrics
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask


logger = logging.getLogger(__name__)


@register_task("sentence_prediction_custom_task")
# @register_task("sentence_prediction_custom_task")
class SentencePredictionCustomTask(SentencePredictionTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--add-prev-output-tokens",
            action="store_true",
            default=False,
            help="add prev_output_tokens to sample, used for encoder-decoder arch",
        )

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args, data_dictionary, label_dictionary)

    ### Train step
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion.forward_kl(model, sample, optimizer)
            return loss, sample_size, logging_output
    
    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if len(logging_outputs) > 0 and "kl_loss" in logging_outputs[0]:
            metrics.log_scalar('kl_loss', sum(l['kl_loss'] for l in logging_outputs))
        
    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )

        return model
