from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# 原始彩色标签路径
SRC_TRAIN = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid\ann_dir\train")
SRC_VAL = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid\ann_dir\val")

# 转换后的单通道标签路径
OUT_TRAIN = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid\ann_index\train")
OUT_VAL = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid\ann_index\val")

OUT_TRAIN.mkdir(parents=True, exist_ok=True)
OUT_VAL.mkdir(parents=True, exist_ok=True)

# UAVid 官方 8 类颜色映射
# 这里的类别编号 0-7 是我们训练时用的索引
COLOR_TO_ID = {
    (0, 0, 0): 0,          # Background clutter
    (128, 0, 0): 1,        # Building
    (128, 64, 128): 2,     # Road
    (0, 128, 0): 3,        # Tree
    (128, 128, 0): 4,      # Low vegetation
    (64, 0, 128): 5,       # Moving car
    (192, 0, 192): 6,      # Static car
    (64, 64, 0): 7,        # Human
}


def convert_one(src_path: Path, out_path: Path):
    img = Image.open(src_path).convert("RGB")
    arr = np.array(img)

    h, w, _ = arr.shape
    out = np.full((h, w), 255, dtype=np.uint8)

    for color, class_id in COLOR_TO_ID.items():
        mask = np.all(arr == np.array(color, dtype=np.uint8), axis=-1)
        out[mask] = class_id

    unknown_pixels = np.sum(out == 255)
    if unknown_pixels > 0:
        print(f"[警告] {src_path.name} 有 {unknown_pixels} 个像素没有匹配到颜色，已设为 255 忽略标签")

    Image.fromarray(out).save(out_path.with_suffix(".png"))


def convert_folder(src_dir: Path, out_dir: Path):
    files = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))

    print(f"处理目录: {src_dir}")
    print(f"文件数量: {len(files)}")

    for p in tqdm(files):
        convert_one(p, out_dir / p.name)


convert_folder(SRC_TRAIN, OUT_TRAIN)
convert_folder(SRC_VAL, OUT_VAL)

print("\n转换完成！")
print("训练标签输出:", OUT_TRAIN)
print("验证标签输出:", OUT_VAL)