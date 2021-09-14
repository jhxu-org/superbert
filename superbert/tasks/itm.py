import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
from .builder import TASKS
from superbert.utils.my_metrics import Accuracy
from superbert.modeling import heads
from superbert.utils.weight import init_weights


@TASKS.register_module("itm")
class ITM(nn.Module):
    def __init__(self, weight, model, **kwargs):
        super().__init__()
        self.weight = weight
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        model.itm_score = heads.ITMHead(model.config.hidden_size)
        model.itm_score.apply(init_weights)
        self.must_infer = kwargs.get("infer", False)

    def forward(self, pl_module, batch, infer):

        itm_labels = torch.Tensor(batch['itm_labels']).long().cuda()
        if self.must_infer:
            infer = pl_module.infer(batch, mask_text=False, mask_image=False)
        itm_logits = pl_module.itm_score(infer["cls_feats"])

        itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
        phase = "train" if pl_module.training else "val"
        acc = getattr(self, f"{phase}_accuracy")(itm_logits, itm_labels)
        ret = {
            "itm_loss": itm_loss,
            "itm_logits": itm_logits.detach(),
            "itm_labels": itm_labels.detach(),
            "itm_accuracy":acc
        }
        return ret