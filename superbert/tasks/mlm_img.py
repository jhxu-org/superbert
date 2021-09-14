from ast import Str
import torch
from torch import nn
import torch.nn.functional as F
from .builder import TASKS
from superbert.utils.my_metrics import Accuracy, Scalar
from superbert.utils.weight import init_weights
from superbert.modeling import heads



@TASKS.register_module("mlm_img")
class MLM_IMG(nn.Module):
    def __init__(self, weight, model, **kwargs):
        super().__init__()
        self.weight = weight
        setattr(self, f"train_accuracy", Accuracy().cuda())
        setattr(self, f"val_accuracy", Accuracy().cuda())
        model.img_mlm_score = heads.IMMLMHead(model.config)
        model.img_mlm_score.apply(init_weights)
        self.mask_image = kwargs.get("mask_image", False)
        self.loss_fn = kwargs.get("loss_fn", 'l1')
        model.mask_image = kwargs.get("mask_image", False)
        model.mask_text = True

        # if self.mask_image:
        #     self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))
            # self.mask_generator = MaskingGenerator(kwargs['num_masking_patches'])
        self.must_infer = kwargs.get("infer", False)

    def forward(self, pl_module, batch, infer):

        # if self.mask_image:
        #     num_patch = batch['image'][0].shape[1]
        #     with torch.no_grad():
        #         batch_size, seq_len, _ = batch['image'][0].size()
        #         mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        #         img_mask = torch_mask_tokens(batch['image'][0]).unsqueeze(-1).type_as(mask_token)
        #         batch['image'][0] = batch['image'][0].cuda()*(1-img_mask) + mask_token*img_mask
        # ret = torch_mask_tokens(batch['image'])
        if self.must_infer:
            infer = pl_module.infer(batch, mask_text=True, mask_image=self.mask_image)
        mlm_logits = pl_module.img_mlm_score(infer["image_feats"][:, 1:])[infer['image_mask']==1]
        mlm_labels = infer["image_mask_labels"]
        
        # mlm_loss = F.cross_entropy(
        #     mlm_logits.view(-1, pl_module.config.vocab_size),
        #     mlm_labels.view(-1),
        #     ignore_index=-100,
        # )
        if self.loss_fn == 'l1':
            mlm_loss = F.l1_loss(
                mlm_logits,
                mlm_labels,
            )
        elif self.loss_fn == 'kl':
            loss_fct = torch.nn.KLDivLoss(reduction="batchmean")
            log_softmax = torch.nn.LogSoftmax(dim=-1)
            reshaped_logits = mlm_logits.contiguous().view(-1, 768)
            reshaped_logits = log_softmax(reshaped_logits)
            mlm_loss = loss_fct(reshaped_logits, mlm_labels.view(-1, 768).contiguous())
        elif self.loss_fn == 'l2':
            mlm_loss = F.mse_loss(
                mlm_logits,
                mlm_labels,
            )
        phase = "train" if pl_module.training else "val"
        # acc = getattr(self, f"{phase}_accuracy")(mlm_logits, mlm_labels)
        ret = {
            "img_mlm_loss": mlm_loss,
            # "mlm_accuracy":acc
        }

        # phase = "train" if pl_module.training else "val"
        # loss = getattr(self, f"{phase}_mlm_loss")(ret["mlm_loss"])
        # acc = getattr(self, f"{phase}_mlm_accuracy")(
        #     mlm_logits, mlm_labels
        # )
        # pl_module.log(f"mlm/{phase}/loss", loss)
        # pl_module.log(f"mlm/{phase}/accuracy", acc)

        return ret
