import torch
from torch import nn
import torch.nn.functional as F
from .builder import TASKS
from superbert.utils.weight import init_weights

@TASKS.register_module("objects_count")
class ObjectCount(nn.Module):
    def __init__(self, weight, model, **kwargs):
        super().__init__()
        self.weight = weight
        hs = model.cfg.img_feature_dim
        model.object_classifier = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, 1),
        )
        model.object_classifier.apply(init_weights)
        self.must_infer = kwargs.get("infer", False)
        
    def forward(self, pl_module, batch, infer):
        if self.must_infer:
            infer = pl_module.infer(batch)
        # cls_token = infer["text_feats"][infer["text_ids"] == 101]
        cls_labels = batch['cls_labels']
        loss = 0
        object_logits = []
        object_labels = []
        for ids, feat, label in zip(batch["text_ids"], infer["text_feats"], cls_labels):
            objects_num = torch.tensor(label).to(feat.device).float().view(-1).contiguous()
            cls_token = feat[ids == 101]
            object_logit = torch.exp(pl_module.object_classifier(cls_token)).view(-1).contiguous()
            object_logit = object_logit.view(-1).contiguous()
            max_tokens = object_logit.shape[0]
            object_logits.append(object_logit)
            object_labels.append(objects_num[:max_tokens])
            # assert object_logit.shape[0] == objects_num.shape[0]
            # loss += F.mse_loss(object_logit, objects_num)
        # vcr_logits = pl_module.vcr_classifier(infer["text_feats"]).squeeze(-1)
        object_logits = torch.cat(object_logits, dim=0)
        object_labels = torch.cat(object_labels, dim=0)
        
        objects_num_loss = F.l1_loss(object_logits, object_labels)
        if pl_module.cfg.n_gpu > 1: objects_num_loss = objects_num_loss.mean()

        ret = {
            "object_logits": object_logits.detach(),
            "object_labels": object_labels.detach(),
            "objects_num_loss": objects_num_loss
        }
        # phase = "train" if pl_module.training else "val"
        # loss = getattr(pl_module, f"{phase}_objects_num_loss")(ret["objects_num_loss"])
        # acc = getattr(pl_module, f"{phase}_objects_num_accuracy")(
        #     ret["object_logits"], ret["object_labels"]
        # )
        ret['accuracy'] = torch.div(sum(ret["object_logits"].long() == ret["object_labels"].long()), len(ret["object_logits"]))
        ret['score'] = ret["object_logits"].long() == ret["object_labels"].long()
        return ret