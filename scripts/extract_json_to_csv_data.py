"""
JSON 메타데이터에서 필요한 컬럼만 추출하여 CSV로 저장
- 추출 컬럼: task, area, disease, grow, risk
- 입력 경로:
    dataset/meta/json/TL5_토마토/{정상,병해,생리장해,작물보호제처리반응}/*.json
    dataset/meta/json/VL5_토마토/{정상,병해,생리장해,작물보호제처리반응}/*.json
- 출력 파일:
    dataset/metadata_train.csv
    dataset/metadata_val.csv

"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

# JSON이 들어있는 클래스 폴더명
CLASS_FOLDERS = ["정상", "병해", "생리장해", "작물보호제처리반응"]


def parse_json_min(json_path: Path) -> Optional[Dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        desc = data.get("description", {})
        ann = data.get("annotations", {})

        image_name = desc.get("image")
        image_id = Path(image_name).stem if image_name else None

        row = {
            "image_id": image_id,
            "task": desc.get("task"),      # 0~3
            "area": ann.get("area"),
            "disease": ann.get("disease"),
            "grow": ann.get("grow"),
            "risk": ann.get("risk"),
        }

        # 최소 유효성 체크: task가 없으면 스킵(데이터 깨짐 방지)
        if row["task"] is None:
            print(f"[SKIP] task 없음: {json_path}")
            return None

        return row

    except Exception as e:
        print(f"[ERROR] {json_path.name}: {e}")
        return None


def process_split(json_root: Path) -> List[Dict]:
    rows: List[Dict] = []

    if not json_root.exists():
        print(f"[WARN] JSON 루트 폴더 없음: {json_root}")
        return rows

    for class_name in CLASS_FOLDERS:
        class_dir = json_root / class_name
        if not class_dir.exists():
            print(f"[SKIP] 폴더 없음: {class_dir}")
            continue

        json_files = list(class_dir.glob("*.json"))
        for jp in json_files:
            row = parse_json_min(jp)
            if row is not None:
                rows.append(row)

    return rows


def save_csv(rows: List[Dict], output_path: Path) -> None:
    """rows를 CSV로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WARN] 저장할 데이터 없음: {output_path}")
        return

    # 고정 컬럼 순서
    fieldnames = ["image_id", "task", "area", "disease", "grow", "risk"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CSV 저장 완료: {output_path} (rows={len(rows)})")


def main():
    # 이 파일은 scripts/ 아래에 있으므로, 프로젝트 루트는 parent.parent
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"

    train_json_root = dataset_dir / "meta" / "json" / "TL5_토마토"
    val_json_root   = dataset_dir / "meta" / "json" / "VL5_토마토"

    train_rows = process_split(train_json_root)
    val_rows   = process_split(val_json_root)

    save_csv(train_rows, dataset_dir / "metadata_train.csv")
    save_csv(val_rows, dataset_dir / "metadata_val.csv")

    print("\n Done!")


if __name__ == "__main__":
    main()
