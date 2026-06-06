_base_ = ['./segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py']

# 1. 你的真实数据集路径
data_root = 'D:/Github/Datasets/cityscapes/cityscapes/'

# 2. 严格统一裁剪尺寸 (适配 RTX 4060)
crop_size = (512, 1024) 

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(data_preprocessor=data_preprocessor)

# 3. 强制覆盖底层流水线，锁死 crop_size
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # 👈 强制规定切图尺寸
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 4. 重新绑定 dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        pipeline=val_pipeline)
)
test_dataloader = val_dataloader

# 5. 训练与保存策略
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1000)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=1,
        save_best='mIoU'
    )
)

load_from = None
resume = False