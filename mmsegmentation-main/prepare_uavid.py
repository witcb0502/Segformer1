from pathlib import Path
import shutil

# 原始 UAVid 数据集路径
SRC_ROOT = Path(r"D:\Github\Segformer\uavid")

# 整理后放到 MMSegmentation 的 data/uavid
DST_ROOT = Path(r"D:\Github\Segformer\mmsegmentation-main\data\uavid")

IMG_TRAIN = DST_ROOT / "img_dir" / "train"
IMG_VAL = DST_ROOT / "img_dir" / "val"
ANN_TRAIN = DST_ROOT / "ann_dir" / "train"
ANN_VAL = DST_ROOT / "ann_dir" / "val"

for p in [IMG_TRAIN, IMG_VAL, ANN_TRAIN, ANN_VAL]:
    p.mkdir(parents=True, exist_ok=True)


def find_dirs(seq_dir: Path):
    img_dir = None
    label_dir = None

    for p in seq_dir.rglob("*"):
        if p.is_dir():
            name = p.name.lower()
            if "image" in name or "img" in name:
                img_dir = p
            if "label" in name or "mask" in name or "gt" in name:
                label_dir = p

    return img_dir, label_dir


def collect_images(folder: Path):
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(folder.glob(ext))
    return sorted(files)


def copy_split(split_folder_name: str, out_img: Path, out_ann: Path):
    split_dir = SRC_ROOT / split_folder_name

    if not split_dir.exists():
        print(f"[错误] 找不到目录: {split_dir}")
        return

    seq_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    print(f"\n处理 {split_folder_name}，序列数量: {len(seq_dirs)}")

    img_count = 0
    ann_count = 0

    for seq in seq_dirs:
        img_dir, label_dir = find_dirs(seq)

        if img_dir is None:
            print(f"[警告] {seq.name} 没找到图片文件夹")
            continue

        if label_dir is None:
            print(f"[警告] {seq.name} 没找到标签文件夹")

        img_files = collect_images(img_dir)

        for img_path in img_files:
            new_img_name = f"{seq.name}_{img_path.name}"
            shutil.copy2(img_path, out_img / new_img_name)
            img_count += 1

            if label_dir is not None:
                label_candidates = list(label_dir.glob(img_path.stem + ".*"))

                if label_candidates:
                    label_path = label_candidates[0]
                    new_label_name = f"{seq.name}_{label_path.name}"
                    shutil.copy2(label_path, out_ann / new_label_name)
                    ann_count += 1
                else:
                    print(f"[警告] {seq.name} 找不到标签: {img_path.name}")

    print(f"复制图片数量: {img_count}")
    print(f"复制标签数量: {ann_count}")


copy_split("uavid_train", IMG_TRAIN, ANN_TRAIN)
copy_split("uavid_val", IMG_VAL, ANN_VAL)

print("\n整理完成！")
print("输出目录:", DST_ROOT)