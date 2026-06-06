from pathlib import Path
from PIL import Image
import numpy as np

ann_dir = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid\ann_index\train")
files = list(ann_dir.glob("*.png")) + list(ann_dir.glob("*.jpg"))

if not files:
    print("没有找到标签文件")
    exit()

p = files[0]
img = Image.open(p)
arr = np.array(img)

print("标签文件:", p)
print("PIL mode:", img.mode)
print("shape:", arr.shape)
print("dtype:", arr.dtype)

if arr.ndim == 2:
    print("这是单通道标签")
    print("unique values:", np.unique(arr)[:50])
else:
    print("这是彩色标签，需要转换成 0-7 单通道标签")
    pixels = arr.reshape(-1, arr.shape[-1])
    colors = np.unique(pixels, axis=0)
    print("颜色数量:", len(colors))
    print("前 20 个颜色:")
    print(colors[:20])