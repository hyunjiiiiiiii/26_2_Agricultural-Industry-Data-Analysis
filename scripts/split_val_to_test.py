import random
import shutil
from pathlib import Path

# =====================
# 설정
# =====================
DATASET_ROOT = Path("dataset")
VAL_RATIO_TO_TEST = 0.3
SEED = 42

IMG_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

# =====================
# 초기화
# =====================
random.seed(SEED)

img_val_dir = DATASET_ROOT / "images" / "val"
lbl_val_dir = DATASET_ROOT / "labels" / "val"

img_test_dir = DATASET_ROOT / "images" / "test"
lbl_test_dir = DATASET_ROOT / "labels" / "test"

img_test_dir.mkdir(parents=True, exist_ok=True)
lbl_test_dir.mkdir(parents=True, exist_ok=True)

# =====================
# val 이미지 목록 수집
# =====================
img_files = []
for ext in IMG_EXTS:
    img_files.extend(img_val_dir.glob(f"*{ext}"))

img_files = sorted(img_files)

assert len(img_files) > 0, "❌ val 이미지가 없습니다."

# =====================
# 랜덤 샘플링
# =====================
num_test = int(len(img_files) * VAL_RATIO_TO_TEST)
test_imgs = random.sample(img_files, num_test)

print(f"[INFO] Total val images : {len(img_files)}")
print(f"[INFO] Move to test     : {num_test}")
print(f"[INFO] Remain in val    : {len(img_files) - num_test}")

# =====================
# 이동
# =====================
missing_labels = 0

for img_path in test_imgs:
    name = img_path.stem
    lbl_path = lbl_val_dir / f"{name}.txt"

    # 이미지 이동
    shutil.move(str(img_path), img_test_dir / img_path.name)

    # 라벨 이동
    if lbl_path.exists():
        shutil.move(str(lbl_path), lbl_test_dir / lbl_path.name)
    else:
        missing_labels += 1

if missing_labels > 0:
    print(f"⚠️ Warning: {missing_labels} images had no label files")

print("Done: validation → test split completed.")
