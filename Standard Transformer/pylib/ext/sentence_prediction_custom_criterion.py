# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("sentence_prediction_custom_criterion")
class SentencePredictionCustomCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name, regression_target, kl_weight):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target
        self.kl_weight = kl_weight

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--kl-weight',
                            default=1.0,
                            type=float,
                            help='add kl loss')
        # fmt: on

    def compute_kl_loss(self, model, p_out, q_out, reduce=True):
        p_tec = F.softmax(p_out, dim = -1, dtype=torch.float32)
        p = F.log_softmax(p_out, dim=-1, dtype=torch.float32)
        q_tec = F.softmax(q_out, dim = -1, dtype=torch.float32)
        q = F.log_softmax(q_out, dim=-1, dtype=torch.float32)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
 
        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def forward_kl(self, model, sample, optimizer, reduce=True):
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

        another_logits, _ = model(
                **sample["net_input"],
                features_only=True,
                classification_head_name=self.classification_head_name,
        )
        
        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            another_lprobs = F.log_softmax(another_logits, dim=-1, dtype=torch.float32)
            base_loss = F.nll_loss(lprobs, targets, reduction="sum")
            another_loss = F.nll_loss(another_lprobs, targets, reduction="sum")
            kl_loss = self.compute_kl_loss(model, logits, another_lprobs)
            loss = base_loss + another_loss + self.kl_weight * kl_loss
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            another_logits = another_logits.view(-1).float()
            # mse_loss = F.mse_loss(logits, targets, reduction="sum") + F.mse_loss(another_logits, targets, reduction="sum")
            l1_loss = F.l1_loss(logits, targets, reduction="sum") + F.l1_loss(another_logits, targets, reduction="sum")
            kl_loss = F.mse_loss(logits, another_logits, reduction="sum")
            # kl_loss = F.l1_loss(logits, another_logits, reduction="sum")
            loss = l1_loss + self.kl_weight * kl_loss
        optimizer.backward(loss)


        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "kl_loss": kl_loss.data,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_output

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

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            # loss = F.mse_loss(logits, targets, reduction="sum")
            loss = F.l1_loss(logits, targets, reduction='sum')


        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
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
