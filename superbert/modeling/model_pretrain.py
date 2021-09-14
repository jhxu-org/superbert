from torch._C import NoneType
from superbert.tasks.builder import build_tasks
from .vision_transformer import *
from torch import nn
import torch
from .modeling_bert import BertImgForPreTraining, BertConfig, PatchEmbed, FeatEmbed
from .builder import MODEL
import torch.nn.functional as F
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from superbert.utils.utils import torch_mask_tokens
@MODEL.register_module('superbert')
class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, cfg, log=None):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.config = BertConfig.from_pretrained(
            cfg.config_name if cfg.config_name else cfg.model_name_or_path,
        )
        self.config.img_layer_norm_eps = cfg.img_layer_norm_eps
        self.config.use_img_layernorm = cfg.use_img_layernorm

        # discrete code
        self.config.img_feature_dim = cfg.img_feature_dim
        self.config.img_feature_type = cfg.img_feature_type
        self.config.hidden_dropout_prob = cfg.drop_out
        if cfg.texta_false_prob < 0.5 and (cfg.texta_false_prob > 0 or not cfg.use_b):
            cfg.num_contrast_classes = 3
        else:
            cfg.num_contrast_classes = 2
        self.config.num_contrast_classes = cfg.num_contrast_classes
        
        if cfg.img_feature_type == 'vit':
            self.patch_embed = PatchEmbed(
            img_size=224, patch_size=16, in_chans=3, embed_dim=cfg.img_feature_dim)
            self.dist_token = nn.Parameter(torch.zeros(1, 1, cfg.img_feature_dim))
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.img_feature_dim))
            self.patch_dim = 224 // 16
            trunc_normal_(self.dist_token, std=0.02)
            trunc_normal_(self.pos_embed, std=0.02)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.img_feature_dim))
            self.pos_drop = nn.Dropout(p=self.config.hidden_dropout_prob)
            if self.cfg.add_norm_before_transformer:
                self.pre_norm = nn.LayerNorm( cfg.img_feature_dim, eps=1e-6)

        elif cfg.img_feature_type == 'patch':
            self.image_model = None
        else:
            self.image_model = FeatEmbed(self.config.img_feature_dim, self.config.hidden_size)

        
        self.cross_model = BertImgForPreTraining.from_pretrained(
                cfg.model_name_or_path,
                from_tf=bool('.ckpt' in cfg.model_name_or_path),
                config=self.config)
        
        self.mask_image = False
        self.mask_text = False
        self.cfg["loss_names"] = [task['type'] for task in cfg.tasks]
        self.tasks = [task.cuda() for task in build_tasks(cfg.tasks, {"model":self})]
        self.current_tasks = cfg["loss_names"]


        if self.mask_image:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        # self.image_model.eval()
        # self.set_metrics()

    def visual_embed(self, _x, max_image_len=200, mask_it=False):
        _, _, ph, pw = self.patch_embed.proj.weight.shape

        x = self.patch_embed(_x)
        x_mask = (_x.sum(dim=1) != 0).float()[:, None, :, :]
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        )
        x_mask = x_mask.flatten(1)

        
        if (
            max_image_len < 0
            or max_image_len is None
            or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w
            max_image_len = eff.max()
        else:
            eff = x_h * x_w
            max_image_len = min(eff.max(), max_image_len)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(
                    torch.ones(nv).float(), p, replacement=True
                )
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(B, -1, C)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)

        if mask_it:
            with torch.no_grad():
                batch_size, seq_len, _ = x.size()
                mask_token = self.mask_token.expand(batch_size, seq_len, -1)
                image_mask = torch_mask_tokens(x).type_as(mask_token)
                image_labels = x[image_mask==1]
                image_mask  = image_mask.unsqueeze(-1)
                x = x*(1-image_mask) + mask_token*image_mask
            
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.pos_embed[:, 0, :][:, None, :].expand(B, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.pos_drop(x)

        if self.cfg.add_norm_before_transformer:
            x = self.pre_norm(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        if mask_it:
            return x, x_mask, (patch_index, (H, W)), image_labels, image_mask.squeeze(-1)
        else:
            return x, x_mask, (patch_index, (H, W)), None, None

    def infer(self, batch):
        do_mlm = "_mlm" if self.mask_text else ""
        input_ids = batch[f"text_ids{do_mlm}"].cuda()
        input_masks = batch['text_masks'].cuda()
        segment_ids = batch['text_segment_ids'].cuda()
        lm_label_ids = batch[f"text_labels{do_mlm}"].cuda()
        images  = batch['image'][0].cuda()
        
        if self.cfg.img_feature_type == 'det':
            images = self.image_model(images)
            image_masks = torch.ones([images.shape[0],images.shape[1]]).to(images.device)
        elif self.cfg.img_feature_type == 'vit':
            # with torch.no_grad():
            assert self.patch_embed != None
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
                image_mask,
            ) = self.visual_embed(
                images,
                max_image_len=self.cfg.max_img_seq_length,
                mask_it=self.mask_image,
            )  
        else:
            image_embeds = images
            image_masks = torch.ones([images.shape[0],images.shape[1]]).to(images.device)
        input_masks = torch.cat([input_masks, image_masks], dim=1)
        output = self.cross_model(img_feats=image_embeds, 
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks,
                position_ids=None, head_mask=None)  #img_masks=image_masks, 

        cls_feats = output[1]
        text_feats, image_feats = (
        output[0][:, : input_ids.shape[1]],
        output[0][:, input_ids.shape[1] :],
        )
        return {
            "cls_feats":cls_feats,
            "text_feats":text_feats,
            "image_feats":image_feats,
            "image_masks":image_masks,
            "text_masks":input_masks,
            "image_mask_labels":image_labels,
            "image_mask":image_mask,

        }


    def forward(self, batch) -> dict:
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret
        infer = self.infer(batch)
        for task in self.tasks:
            ret.update(task(self,batch, infer))

        return ret

    def get_val_metric(self):
        metric = {}
        score = -1
        for name, task in zip(self.current_tasks, self.tasks):
            acc_handle = getattr(task, "val_accuracy")
            tmp = acc_handle.compute()
            if score == -1:
                score = tmp
            metric.update({name:tmp})
            acc_handle.reset()
        return score, metric
            