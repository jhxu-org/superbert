train = [
        # dict(
        #     type="vqav2",
        #     split="train",
        #     max_text_len=60,
        #     data_dir="/search/gpu3/superbert",
        #     image_only=False,
        #     cache_dir="/search/gpu2/xujianhua/cache",
        #     pipeline=dict(type="pixelbert_randaug",size=224),
        #     mlm=True,
        #     whole_word_masking=True,
        #     mlm_prob=0.15
        # ),
        # dict(
        #     type="sbu",
        #     split="train",
        #     max_text_len=60,
        #     data_dir="/search/gpu3/superbert",
        #     image_only=False,
        #     cache_dir="/search/gpu2/xujianhua/cache",
        #     pipeline=dict(type="pixelbert_randaug",size=224),
        #     mlm=True,
        #     whole_word_masking=True,
        #     mlm_prob=0.15

        # ),
        dict(
            type="vg",
            split="train",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0,
        ),
        dict(
            type="f30k",
            split="train",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0

        ),
        dict(
            type="coco",
            split="train",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0
        ),
]

val = [
        # dict(
        #     type="vqav2",
        #     split="val",
        #     max_text_len=60,
        #     data_dir="/search/gpu3/superbert",
        #     image_only=False,
        #     cache_dir="/search/gpu2/xujianhua/cache",
        #     pipeline=dict(type="pixelbert_randaug",size=224),
        #     mlm=True,
        #     whole_word_masking=True,
        #     mlm_prob=0.15
        # ),
        # dict(
        #     type="sbu",
        #     split="val",
        #     max_text_len=60,
        #     data_dir="/search/gpu3/superbert",
        #     image_only=False,
        #     cache_dir="/search/gpu2/xujianhua/cache",
        #     pipeline=dict(type="pixelbert_randaug",size=224),
        #     mlm=True,
        #     whole_word_masking=True,
        #     mlm_prob=0.15

        # ),
        dict(
            type="vg",
            split="val",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0
        ),
        dict(
            type="f30k",
            split="val",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0
        ),
        dict(
            type="coco",
            split="val",
            max_text_len=60,
            data_dir="/search/gpu3/superbert",
            image_only=False,
            cache_dir="/search/gpu2/xujianhua/cache",
            pipeline=dict(type="pixelbert_randaug",size=224),
            mlm=True,
            whole_word_masking=True,
            mlm_prob=0.15,
            draw_false_image=1,
            draw_false_text=0

        )
]


tasks = [
    dict(type="mlm", weight=1),
    dict(type="itm_wpa", weight=1),

]

model = dict(type="superbert")
# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2,
# )
# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


model_name_or_path="/search/gpu2/xujianhua/code/pretrain/vilt_my/checkpoints/bert-base-uncased"
bert_model="bert-base-uncased"
max_iters=2000
max_img_seq_length=-1
img_feature_dim=768
img_feature_type="patch"
vit = ""
mask_image=False
mask_text=False
use_layernorm=True
add_norm_before_transformer=True
drop_out=0.1
drop_rate=drop_out
use_b = 1
textb_sample_mode=1
texta_false_prob=0.0
config_name=""
cache_dir="/search/gpu2/xujianhua/cache"
load_path=""
# max_seq_length=60
learning_rate=1e-5
weight_decay=0.01
scheduler="linear"
samples_per_gpu=25
workers_per_gpu=4
optimizer = "adamw"
adam_epsilon=1e-8
do_lower_case = True
on_memory=True
max_grad_norm=1.0
warmup_steps=2000
gradient_accumulation_steps=1
use_img_layernorm = 0
img_layer_norm_eps=1e-12
log_period=10
ckpt_period=5000
val_check_interval = 1.0
num_train_epochs=20
fp16=False
adjust_dp=True
save_epoch=1
save_after_epoch=0