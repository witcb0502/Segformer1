import os
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from mmseg.apis import init_model, inference_model
from mmseg.datasets import get_dataset
from mmseg.utils import colorize_mask, overlay_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='权重文件路径')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--out-dir', required=True, help='保存预测图的文件夹')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--alpha', type=float, default=0.55)
    args = parser.parse_args()

    # 获取输入图像路径和输出路径
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)

    # 使用训练时的日志目录路径
    work_dir = Path("./work_dirs/segformer_b0_uavid")  # 修改为训练的工作目录路径
    out_dir = work_dir / "results"  # 将预测图保存到此目录下
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # 加载模型和权重
    model = init_model(args.config, args.checkpoint, device=args.device)
    result = inference_model(model, str(image_path))

    pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
    image_rgb = np.array(Image.open(image_path).convert('RGB'))

    pred_color = colorize_mask(pred)
    pred_overlay = overlay_image(image_rgb, pred_color, alpha=args.alpha)

    original_save_path = out_dir / f'{stem}_original.png'
    pred_save_path = out_dir / f'{stem}_pred_overlay.png'

    shutil.copy2(image_path, original_save_path)  # 保存原始图像
    Image.fromarray(pred_overlay).save(pred_save_path)  # 保存预测图

    print('预测完成')
    print('原图保存到:', original_save_path)
    print('预测图保存到:', pred_save_path)

if __name__ == "__main__":
    main()