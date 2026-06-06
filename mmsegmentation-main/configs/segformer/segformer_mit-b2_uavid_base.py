_base_ = [
    '../_base_/models/segformer_mit-b2.py',
    '../_base_/datasets/uavid.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

num_classes = 8

# 1. 修复预训练权重与测试模式 (添加 pretrained 和 mode='slide')
model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth',
    data_preprocessor=dict(size=(512, 512)),
    decode_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)) # 滑窗测试，步长通常设为尺寸的2/3
)

# 2. 提升 Batch Size 以稳定梯度 (4060显存跑512x512切图，bs设为4没问题)
train_dataloader = dict(batch_size=4, num_workers=2, persistent_workers=True)
val_dataloader = dict(batch_size=1, num_workers=2, persistent_workers=True)
test_dataloader = val_dataloader

# 3. 修复迭代步数 (1万步太少，提升到至少 40000)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50), # 缩短日志打印间隔，方便观察
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=1,
        save_best='mIoU',
        save_last=True,
        rule='greater',
        out_dir='./work_dirs/segformer_b2_uavid/checkpoints'
    )
)

# 4. 彻底修复优化器 (必须用 AdamW)
optim_wrapper = dict(
    _delete_=True, # 删除 base 里的 SGD
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# 5. 修复学习率调度器 (对应 max_iters)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

work_dir = './work_dirs/segformer_b2_uavid'
out_dir = work_dir + '/predictions'