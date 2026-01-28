"""
dataset_cls/ 폴더에서 클래스(disease)별 이미지 개수 집계

기본 폴더 구조(권장):
project_root/
  dataset_cls/
    train/
      b1/ xxx.jpg ...
      b2/ ...
    val/
      b1/ ...
      b2/ ...

실행:
  python scripts/check_cls_counts.py

옵션:
  --root dataset_cls   (기본값)
  --min_n 200          (기본값: 200, '너무 적은 클래스' 기준)
  --ext jpg jpeg png webp  (기본 확장자)
"""

import argparse
from pathlib import Path
from collections import Counter
from typing import List


def count_images_in_class_dir(class_dir: Path, exts: List[str]) -> int:
    """한 클래스 폴더 안 이미지 개수 세기"""
    n = 0
    for ext in exts:
        n += len(list(class_dir.glob(f"*.{ext}")))
    return n


def count_split(split_dir: Path, exts: List[str]) -> Counter:
    """split(train/val) 폴더 안 클래스별 이미지 개수 Counter로 반환"""
    counts = Counter()
    if not split_dir.exists():
        print(f"[ERROR] split dir not found: {split_dir}")
        return counts

    for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        counts[cls_dir.name] = count_images_in_class_dir(cls_dir, exts)

    return counts


def print_report(title: str, counts: Counter, min_n: int) -> None:
    """리포트 출력"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    if not counts:
        print("[WARN] no classes found.")
        return

    total = sum(counts.values())
    print(f"classes: {len(counts)} | total images: {total}")

    # 많은 순으로 출력
    for cls, n in counts.most_common():
        flag = " <LOW" if n < min_n else ""
        print(f"{cls:15s} : {n:7d}{flag}")

    # LOW 클래스 따로 요약
    low = [(cls, n) for cls, n in counts.items() if n < min_n]
    if low:
        low_sorted = sorted(low, key=lambda x: x[1])
        print(f"\n[LOW classes] (< {min_n}) -> {len(low_sorted)} classes")
        print(", ".join([f"{cls}({n})" for cls, n in low_sorted]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset_cls",
                        help="dataset_cls root folder name (default: dataset_cls)")
    parser.add_argument("--min_n", type=int, default=200,
                        help="threshold for LOW class (default: 200)")
    parser.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png", "webp"],
                        help="image extensions to count (default: jpg jpeg png webp)")
    args = parser.parse_args()

    # scripts/ 아래에 있으므로 프로젝트 루트는 parent.parent
    project_root = Path(__file__).parent.parent
    root_dir = project_root / args.root

    train_dir = root_dir / "train"
    val_dir = root_dir / "val"

    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] dataset_cls root: {root_dir}")
    print(f"[INFO] exts: {args.ext}")
    print(f"[INFO] LOW threshold: {args.min_n}")

    train_counts = count_split(train_dir, args.ext)
    val_counts = count_split(val_dir, args.ext)

    print_report("TRAIN SPLIT", train_counts, args.min_n)
    print_report("VAL SPLIT", val_counts, args.min_n)


if __name__ == "__main__":
    main()
