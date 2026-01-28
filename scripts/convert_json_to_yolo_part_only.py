"""
JSON → YOLO 변환 (PART ONLY, 3 classes)
- annotations.part 만 사용
- 정상(정상 폴더)은 객체 없음(negative): 빈 라벨(.txt) 생성
- 클래스 재매핑:
    병해(1) -> 0
    생리장해(2) -> 1
    작물보호제처리반응(3) -> 2
- 출력: dataset/labels_part_only_3cls/{train,val}
"""

import json
from pathlib import Path
from typing import Tuple

# tqdm 선택적
try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


# 원래 폴더명 -> 원래 상위 클래스 ID (참고용)
CLASS_MAPPING = {
    "정상": 0,
    "병해": 1,
    "생리장해": 2,
    "작물보호제처리반응": 3,
}

# ✅ 3클래스 재매핑 (탐지 대상만)
# original_task_id -> new_class_id
TASK_TO_3CLS = {
    1: 0,  # 병해
    2: 1,  # 생리장해
    3: 2,  # 작물보호제처리반응
}
NORMAL_TASK_ID = 0  # 정상은 negative (객체 없음)


def convert_bbox_to_yolo(
    x: float, y: float, w: float, h: float, img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """절대좌표 -> YOLO 정규화 (x_center, y_center, w, h)"""
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height

    # clip
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return x_center, y_center, width, height


def write_empty_label(output_dir: Path, stem: str) -> None:
    """빈 라벨 파일 생성(negative sample)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stem}.txt").write_text("", encoding="utf-8")


def process_json_file(json_path: Path, original_task_id: int, output_dir: Path) -> bool:
    """
    JSON 1개 처리:
    - 정상(original_task_id=0): 빈 라벨 생성
    - 나머지: part만 YOLO 라벨 생성 (3cls로 재매핑)
    """
    try:
        # 정상은 객체가 없다고 정의: 빈 라벨 생성 후 종료
        if original_task_id == NORMAL_TASK_ID:
            write_empty_label(output_dir, json_path.stem)
            return True

        # 3클래스 재매핑
        mapped_class = TASK_TO_3CLS.get(original_task_id)
        if mapped_class is None:
            # 혹시 모를 예외 케이스 방어
            write_empty_label(output_dir, json_path.stem)
            return True

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        desc = data.get("description", {})
        ann = data.get("annotations", {})

        img_width = desc.get("width", 0)
        img_height = desc.get("height", 0)

        if img_width == 0 or img_height == 0:
            print(f"[WARN] 이미지 크기 없음: {json_path.name}")
            return False

        parts = ann.get("part", [])

        # part가 없으면(예: 병해인데 part 누락) -> 빈 라벨 생성(학습은 가능)
        if not parts:
            write_empty_label(output_dir, json_path.stem)
            return True

        yolo_lines = []
        for part in parts:
            x = float(part.get("x", 0))
            y = float(part.get("y", 0))
            w = float(part.get("w", 0))
            h = float(part.get("h", 0))

            if w <= 1 or h <= 1:
                continue

            xc, yc, wn, hn = convert_bbox_to_yolo(x, y, w, h, img_width, img_height)
            yolo_lines.append(f"{mapped_class} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{json_path.stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        return True

    except Exception as e:
        print(f"[ERROR] {json_path.name}: {e}")
        return False


def process_dataset(json_base_dir: Path, labels_output_dir: Path, split: str):
    print(f"\n{'='*60}")
    print(f"{split.upper()} PART-ONLY (3cls) 라벨 생성")
    print(f"{'='*60}")

    total, success = 0, 0

    for class_name, original_task_id in CLASS_MAPPING.items():
        class_json_dir = json_base_dir / class_name
        if not class_json_dir.exists():
            print(f"[SKIP] 폴더 없음: {class_json_dir}")
            continue

        json_files = list(class_json_dir.glob("*.json"))
        iterator = tqdm(json_files, desc=class_name) if HAS_TQDM else json_files

        for json_path in iterator:
            total += 1
            if process_json_file(json_path, original_task_id, labels_output_dir):
                success += 1

    print(f"\n총 JSON: {total}")
    print(f"성공: {success}")
    print(f"출력 위치: {labels_output_dir}")


def main():
    # 이 파일을 프로젝트 루트(= dataset 폴더가 있는 위치)에 두는 걸 권장
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"

    json_train_dir = dataset_dir / "meta" / "json" / "TL5_토마토"
    json_val_dir = dataset_dir / "meta" / "json" / "VL5_토마토"

    labels_train = dataset_dir / "labels_part_only" / "train"
    labels_val = dataset_dir / "labels_part_only" / "val"

    process_dataset(json_train_dir, labels_train, "train")
    process_dataset(json_val_dir, labels_val, "val")

    print("\n PART-ONLY (3cls) 라벨 생성 완료!")


if __name__ == "__main__":
    main()
