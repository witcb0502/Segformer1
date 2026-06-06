from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np
from PIL import Image

from mmseg.apis import init_model, inference_model


PALETTE = np.array([
    [0, 0, 0],          # 0 Background clutter
    [128, 0, 0],        # 1 Building
    [128, 64, 128],     # 2 Road
    [0, 128, 0],        # 3 Tree
    [128, 128, 0],      # 4 Low vegetation
    [64, 0, 128],       # 5 Moving car
    [192, 0, 192],      # 6 Static car
    [64, 64, 0],        # 7 Human
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """把 0-7 单通道标签转成彩色图。"""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, rgb in enumerate(PALETTE):
        color[mask == class_id] = rgb

    return color


def make_overlay(image_rgb: np.ndarray, color_mask: np.ndarray, alpha=0.55) -> np.ndarray:
    """把彩色预测图叠加到原图上。"""
    return (image_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='权重文件路径')
    parser.add_argument('--image', required=True, help='验证集原图路径')
    parser.add_argument('--gt', required=True, help='验证集标签路径，使用 ann_index 的单通道标签')
    parser.add_argument('--out-dir', required=True, help='输出文件夹')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    gt_path = Path(args.gt)

    model = init_model(args.config, args.checkpoint, device=args.device)
    result = inference_model(model, str(image_path))

    pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

    image_rgb = np.array(Image.open(image_path).convert('RGB'))
    gt = np.array(Image.open(gt_path))

    if gt.ndim == 3:
        print('警告：你传入的 gt 是彩色标签，不是 ann_index 单通道标签。')
        print('建议使用 data/uavid/ann_index/val 下面的标签。')
        gt_color = gt[:, :, :3].astype(np.uint8)
    else:
        gt_color = colorize_mask(gt.astype(np.uint8))

    pred_color = colorize_mask(pred)
    pred_overlay = make_overlay(image_rgb, pred_color, alpha=0.55)

    shutil.copy2(image_path, out_dir / 'original.png')
    Image.fromarray(gt_color).save(out_dir / 'gt_color.png')
    Image.fromarray(pred_color).save(out_dir / 'pred_color.png')
    Image.fromarray(pred_overlay).save(out_dir / 'pred_overlay.png')

    print('可视化结果已保存到：', out_dir)


if __name__ == '__main__':
    main()