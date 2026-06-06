import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. 对应你配置里的路径
img_dir = 'data/uavid/img_dir/train'
ann_dir = 'data/uavid/ann_index/train'

# 获取第一张图片名称
img_names = os.listdir(img_dir)
if not img_names:
    print("原图文件夹为空！请检查路径。")
    exit()

sample_img_name = img_names[0]
# 假设标签文件名和原图完全一样。如果你的标签有后缀，请修改下一行
sample_ann_name = sample_img_name 

img_path = os.path.join(img_dir, sample_img_name)
ann_path = os.path.join(ann_dir, sample_ann_name)

if not os.path.exists(ann_path):
    print(f"严重错误: 找不到对应的标签文件 {ann_path}")
    print("说明文件名没对上！")
    exit()

# 2. 读取图像和标签
img = Image.open(img_path)
lbl = Image.open(ann_path)
lbl_np = np.array(lbl)

print(f"--- 正在检查样本: {sample_img_name} ---")
print(f"标签图的维度 (Shape): {lbl_np.shape}")
print(f"标签图中包含的像素值 (Unique Values): {np.unique(lbl_np)}")

# 3. 诊断分析
if len(lbl_np.shape) == 3:
    print("\n🚨 警告: 你的标签图是 3 通道的 RGB 图像！")
    print("mmsegmentation 需要单通道的索引图 (0-7)。请先编写脚本将 RGB 标签转换为单通道 Index 标签。")
elif np.max(lbl_np) == 255 and len(np.unique(lbl_np)) > 10:
    print("\n🚨 警告: 你的标签图似乎没有被正确转换为 0-7 的类别索引，存在非法类别值！")
else:
    print("\n✅ 数据集格式检查通过！标签是单通道索引图，且文件名对应。")

# 4. 可视化确认有没有错位
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
# 乘以30是为了让0-7的灰度值在肉眼下能区分开来
plt.imshow(lbl_np * 30, cmap='gray') 
plt.title("Label (Index Map)")
plt.show()