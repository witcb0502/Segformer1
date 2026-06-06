import os
import numpy as np
from PIL import Image

# 重新校准的映射表：确保 0-7 的顺序与你的 config/class 定义严格一致
COLOR_MAP = {
    (128, 0, 0): 0,      # Building
    (128, 64, 128): 1,   # Road
    (0, 128, 0): 2,      # Tree
    (128, 128, 0): 3,    # Low vegetation
    (64, 0, 128): 4,     # Moving car
    (192, 0, 192): 5,    # Static car
    (64, 64, 0): 6,      # Human
    (0, 0, 0): 7         # Clutter/Background
}

def convert_labels(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"❌ 找不到输入路径: {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"正在处理 {input_dir}，共 {len(files)} 张图片...")
    
    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 默认全部填充为背景类 7 (Clutter)
        label_index = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 7
        
        for rgb, idx in COLOR_MAP.items():
            match = (img_np == np.array(rgb)).all(axis=2)
            label_index[match] = idx
            
        out_img = Image.fromarray(label_index, mode='L')
        out_img.save(os.path.join(output_dir, filename))
    print(f"🎯 {input_dir} 转换完毕！")

base_dir = r"D:\Github\Segformer\mmsegmentation-main\data\uavid"
convert_labels(os.path.join(base_dir, "ann_color", "train"), os.path.join(base_dir, "ann_index", "train"))
convert_labels(os.path.join(base_dir, "ann_color", "val"), os.path.join(base_dir, "ann_index", "val"))
print("================ 转换结束 ================")