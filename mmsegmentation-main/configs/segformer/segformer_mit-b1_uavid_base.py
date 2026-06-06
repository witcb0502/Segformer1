_base_ = [
    '../_base_/models/segformer_mit-b1.py',
    '../_base_/datasets/uavid.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

num_classes = 8

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    decode_head=dict(num_classes=num_classes)
)

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False
)

test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=2000,
    val_interval=500
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100000),
    checkpoint=dict(
    type='CheckpointHook',
    by_epoch=False,
    interval=1000,
    max_keep_ckpts=1,
    save_best='mIoU',
    save_last=True,
    rule='greater',
    out_dir='./work_dirs/segformer_b1_uavid/checkpoints'
)
)

optim_wrapper = dict(
    optimizer=dict(lr=0.00006)
)

# 相应修改工作目录为B1
work_dir = './work_dirs/segformer_b1_uavid'

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

log_config = dict(
    interval=10,  # 每10步记录一次日志
    hooks=[
        dict(type='TextLoggerHook'),  # 文本日志
        dict(type='TensorBoardLoggerHook', log_dir='./work_dirs/segformer_b1_uavid/tensorboard_logs')  # TensorBoard日志
    ]
)