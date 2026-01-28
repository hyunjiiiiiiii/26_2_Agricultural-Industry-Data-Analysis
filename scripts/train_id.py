"""
train / val / test 폴더에 있는 파일명을 id로 추출하여 CSV로 저장
"""

import csv
from pathlib import Path

# =========================
# 설정
# =========================
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "images"  # 필요 시 수정

SPLITS = ["val"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def save_ids(split: str):
    split_dir = DATASET_DIR / split
    if not split_dir.exists():
        print(f"[SKIP] 폴더 없음: {split_dir}")
        return

    ids = []

    for img_path in split_dir.iterdir():
        if img_path.suffix.lower() in IMAGE_EXTS:
            ids.append(img_path.stem)  # 확장자 제거한 파일명

    if not ids:
        print(f"[WARN] 이미지 없음: {split_dir}")
        return

    output_csv = DATASET_DIR / f"{split}_ids.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id"])
        for _id in sorted(ids):
            writer.writerow([_id])

    print(f"[OK] {split}: {len(ids)} ids saved → {output_csv}")


def main():
    for split in SPLITS:
        save_ids(split)

    print("\n ID CSV 생성 완료")


if __name__ == "__main__":
    main()
