import torch
from torch import nn
import torch.nn.functional as F
from .builder import TASKS
from superbert.utils.my_metrics import Accuracy
from superbert.modeling import heads
from superbert.utils.weight import init_weights

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

@TASKS.register_module("itm_contrative")
class ITM_CONTRATIVE(nn.Module):
    def __init__(self, weight, model, **kwargs):
        super().__init__()
        self.weight = weight
        self.train_accuracy = Accuracy().cuda()
        self.val_accuracy = Accuracy().cuda()
        model.itm_score = heads.ITMHead(model.config.hidden_size)
        model.itm_score.apply(init_weights)
        self.similarity = Similarity(0.05)
        self.must_infer = kwargs.get("infer", False)


    def forward(self, pl_module, batch):

        pos_len = len(batch["text"]) // 2
        neg_len = len(batch["text"]) - pos_len
        itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).cuda()
        itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

        itm_images = [
            torch.stack(
                [
                    ti if itm_labels[i] == 1 else fi
                    for i, (ti, fi) in enumerate(zip(bti, bfi))
                ]
            )
            for bti, bfi in zip(batch["image"], batch["false_image_0"])
        ]

        batch = {k: v for k, v in batch.items()}
        batch["image"] = itm_images

        inferv1 = pl_module.infer(batch, mask_text=False, mask_image=False)
        inferv2 = pl_module.infer(batch, mask_text=False, mask_image=False)
        itm_logitsv1 = inferv1["cls_feats"]
        itm_logitsv2 = inferv2["cls_feats"]

        cos_sim = self.similarity(inferv1["cls_feats"].unsqueeze(1), inferv2["cls_feats"].unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(itm_logitsv1.device)
        loss_fct = nn.CrossEntropyLoss()

        contrative_loss = (loss_fct(cos_sim, labels))

        itm_loss = F.cross_entropy(itm_logitsv1, itm_labels.long()) + F.cross_entropy(itm_logitsv2, itm_labels.long())
        phase = "train" if pl_module.training else "val"
        acc = getattr(self, f"{phase}_accuracy")(itm_logitsv1, itm_labels)
        ret = {
            "itm_loss": itm_loss,
            "itm_contrative_loss": contrative_loss,
            "itm_logits": itm_logitsv1,
            "itm_labels": itm_labels,
            "itm_accuracy":acc
        }
        return ret