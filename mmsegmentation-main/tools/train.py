# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 写入工具
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# 假设 model_output 是模型的输出，target 是真实标签
def compute_loss(model_output, target):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model_output, target)
    return loss
def compute_accuracy(model_output, target):
    _, predicted = torch.max(model_output, 1)  # 获取每个样本的预测类别
    correct = (predicted == target).sum().item()  # 比较预测与真实标签
    accuracy = correct / target.size(0)  # 计算准确度
    return accuracy
# 假设预测的结果是一个 N x H x W 的 Tensor
def generate_predicted_image(model_output):
    # 假设预测的类别已经通过 argmax 得到
    _, predicted = torch.max(model_output, 1)
    
    # 将预测结果转化为对应的颜色映射
    # 这里我们用 0-255 的颜色进行映射
    color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])  # 一个简单的颜色映射例子
    predicted_color = color_map[predicted.cpu().numpy()]  # 转为 numpy 格式
    
    # 将颜色映射后的结果保存为图像
    predicted_image = Image.fromarray(predicted_color[0])  # 选择第一个样本
    return predicted_image
def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 设定工作目录
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 启用混合精度训练（AMP）
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log('AMP training is already enabled in your config.', logger='current', level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', ('`--amp` is only supported when the optimizer wrapper type is '
                                                     f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # 恢复训练
    cfg.resume = args.resume

        # 创建模型训练的Runner
    runner = Runner.from_cfg(cfg)

    # 开始训练
    runner.train()


if __name__ == '__main__':
    main()