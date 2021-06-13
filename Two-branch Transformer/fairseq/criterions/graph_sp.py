from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from typing import List, Dict, Any
from fairseq import metrics, utils
from omegaconf import II
from fairseq.models.doublemodel import DoubleModel
from fairseq.models.onemodel import OneModel


@dataclass
class GraphSentencePredictionCriterionConfig(FairseqDataclass):
    data_type: str = II("model.datatype")
    distill_step: int = field(default=30000)
    use_byol: bool = field(default=False)
    loss_p: float = field(default=1.0)
    only_reg: bool = field(default=False)


@register_criterion("graph_sp", dataclass=GraphSentencePredictionCriterionConfig)
class GraphSentencePredictionCriterion(FairseqCriterion):
    def __init__(self, task, data_type, distill_step, use_byol, loss_p, only_reg):
        super().__init__(task)
        self.data_type = data_type
        self.distill_step = distill_step
        self.use_byol = use_byol
        self.loss_p = loss_p
        self.only_reg = only_reg

    def forward(self, model, sample, reduce=True):

        logits, output_dict = model(
            net_input0=sample["net_input0"],
            net_input1=sample["net_input1"],
            ret_contrastive=self.use_byol,
        )

        if self.data_type in ["tt", "gg"]:

            target_cls = model.get_targets(sample["target_cls"], None).view(-1)
            target_reg = model.get_targets(sample["target_reg"], None).view(-1)
            sample_size = target_cls.numel()

            x_c, x_r = logits
            # groundtruth
            lprobs = F.log_softmax(x_c, dim=-1, dtype=torch.float32)
            loss_c_g = F.nll_loss(lprobs, target_cls, reduce="sum")

            y_pred = x_r.view(-1).float()
            target_reg = target_reg.float()
            if self.loss_p == 1.:
                loss_r_g = F.l1_loss(y_pred, target_reg, reduce="sum")
            else:
                loss_r_g = (torch.abs(y_pred - target_reg) + 1e-6).pow(self.loss_p).mean()
            # loss_r_g = F.mse_loss(y_pred, target_reg, reduce="sum")

            target_cls_d = self.reg2cls(x_r).view(-1)
            loss_c_d = F.nll_loss(lprobs, target_cls_d, reduce="sum")

            target_reg_d = self.cls2reg(x_c)
            target_reg_d = target_reg_d.float()
            if self.loss_p == 1:
                loss_r_d = F.l1_loss(y_pred, target_reg_d, reduce="sum")
            else:
                loss_r_d = (torch.abs(y_pred - target_reg_d) + 1e-6).pow(self.loss_p).mean()
            # loss_r_d = F.mse_loss(y_pred, target_reg_d, reduce="sum")

            coeff = 0.1 if model.num_updates < self.distill_step else 1
            if self.only_reg:
                loss =  loss_c_g * 0 + loss_r_g * 10 + (loss_c_d + loss_r_d * 10) * 0
            else:
                loss = loss_c_g + loss_r_g * 10 + (loss_c_d + loss_r_d * 10) * coeff

            byol_loss = 0
            if self.use_byol:
                for i in range(2):
                    byol_loss = byol_loss + self.get_contrastive_logits(
                        *output_dict["contrastive"][i]
                    )

            loss = loss + byol_loss * 0.1

            # if not self.regression_target:
            #     lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            #     loss = F.nll_loss(lprobs, targets, reduce='sum')
            # else:
            #     logits = logits.view(-1).float()
            #     targets = targets.float()
            #     loss = F.mse_loss(logits, targets, reduce="sum")

            logging_out = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
                "loss_c_g": utils.item(loss_c_g),
                "loss_r_g": utils.item(loss_r_g),
                "loss_c_d": utils.item(loss_c_d),
                "loss_r_d": utils.item(loss_r_d),
                "coeff": coeff,
            }
            logging_out.update(byol=utils.item(byol_loss))
            logging_out.update(
                mae=utils.item(F.l1_loss(y_pred, target_reg, reduction="sum"))
                * self.task.label_scaler.stds
            )
            # if not self.regression_target:
            # preds = logits.argmax(dim=1)
            # logging_out["ncorrect"] = (preds == targets).sum()

            preds = x_c.argmax(dim=1)
            logging_out["ncorrect"] = (preds == target_cls).sum()

            return loss, sample_size, logging_out

    def get_contrastive_logits(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        loss = torch.einsum("nc,nc->n", [anchor, positive])
        loss = -loss.sum()
        return loss

    def cls2reg(self, x_c):
        x_c = x_c.detach()
        with torch.no_grad():
            x_c = torch.argmax(x_c, dim=-1)
            return self.task.cls2reg(x_c)

    def reg2cls(self, x_r):
        x_r = x_r.detach()
        with torch.no_grad():
            return self.task.reg2cls(x_r)

    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        byol = sum(log.get("byol", 0) for log in logging_outputs)
        mae = sum(log.get("mae", 0) for log in logging_outputs)
        
        metrics.log_scalar("byol", byol / sample_size, sample_size, round=3)
        metrics.log_scalar("mae", mae / sample_size, sample_size, round=3)
        losses_name = ["loss_c_g", "loss_r_g", "loss_c_d", "loss_r_d"]
        for name in losses_name:
            tmp = sum(log.get(name, 0) for log in logging_outputs)
            metrics.log_scalar(name, tmp / sample_size, sample_size, round=3)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=3)

        metrics.log_scalar("coeff", logging_outputs[0].get("coeff", 1), 1, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
