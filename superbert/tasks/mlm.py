import torch
from torch import nn
import torch.nn.functional as F
from .builder import TASKS
from superbert.utils.my_metrics import Accuracy, Scalar
from superbert.utils.weight import init_weights
from superbert.modeling import heads
import numpy as np
import math
import random


@TASKS.register_module("mlm")
class MLM(nn.Module):
    def __init__(self, weight, model, **kwargs):
        super().__init__()
        self.weight = weight
        setattr(self, f"train_accuracy", Accuracy().cuda())
        setattr(self, f"val_accuracy", Accuracy().cuda())
        model.mlm_score = heads.MLMHead(model.config)
        model.mlm_score.apply(init_weights)
        # mask_text = kwargs.get("mask_text", False)
        self.must_infer = kwargs.get("infer", False)
        model.mask_text = True


    def forward(self, pl_module, batch, infer):
        if self.must_infer:
            infer = pl_module.infer(batch, mask_text=True, mask_image=self.mask_image)
        mlm_logits = pl_module.mlm_score(infer["text_feats"])
        mlm_labels = batch["text_labels_mlm"].cuda()

        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, pl_module.config.vocab_size),
            mlm_labels.view(-1),
            ignore_index=-100,
        )
        phase = "train" if pl_module.training else "val"
        acc = getattr(self, f"{phase}_accuracy")(mlm_logits, mlm_labels)
        ret = {
            "mlm_loss": mlm_loss,
            "mlm_logits": mlm_logits.detach(),
            "mlm_labels": mlm_labels.detach(),
            "mlm_ids": batch["text_ids"],
            "mlm_accuracy":acc
        }

        # phase = "train" if pl_module.training else "val"
        # loss = getattr(self, f"{phase}_mlm_loss")(ret["mlm_loss"])
        # acc = getattr(self, f"{phase}_mlm_accuracy")(
        #     mlm_logits, mlm_labels
        # )
        # pl_module.log(f"mlm/{phase}/loss", loss)
        # pl_module.log(f"mlm/{phase}/accuracy", acc)

        return ret
