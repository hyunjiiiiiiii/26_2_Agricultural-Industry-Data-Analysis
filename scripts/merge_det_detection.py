import argparse
import pandas as pd
from pathlib import Path


TASKB_FEATURES = [
    "id",
    "p_b2_max",
    "p_b3_max",
    "p_b6_max",
    "p_b7_max",
    "p_b8_max",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="features_images.csv 경로"
    )
    ap.add_argument(
        "--output",
        type=str,
        default="val_detection.csv",
        help="Task B feature CSV 출력 경로"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # 1. CSV 로드
    df = pd.read_csv(in_path)

    # 2. 필요한 컬럼 존재 여부 확인
    missing = [c for c in TASKB_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. Feature 선택
    df_taskb = df[TASKB_FEATURES].copy()

    # 4. 타입 정리 (안전장치)
    int_cols = [
        "has_any_det",
        "has_phys_det",
        "has_other_det",
        "num_phys_boxes",
    ]
    for c in int_cols:
        df_taskb[c] = df_taskb[c].fillna(0).astype(int)

    float_cols = [
        "yolo_conf_max",
        "stn_pmax_max",
        "stn_entropy_mean",
        "p_b2_max",
        "p_b3_max",
        "p_b6_max",
        "p_b7_max",
        "p_b8_max",
    ]
    for c in float_cols:
        df_taskb[c] = df_taskb[c].fillna(0.0).astype(float)

    # 5. sanity check (확률 범위)
    prob_cols = [
        "yolo_conf_max",
        "stn_pmax_max",
        "p_b2_max",
        "p_b3_max",
        "p_b6_max",
        "p_b7_max",
        "p_b8_max",
    ]
    for c in prob_cols:
        df_taskb[c] = df_taskb[c].clip(0.0, 1.0)

    # 6. 저장
    df_taskb.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Task B feature saved to: {out_path}")
    print(f"   rows: {len(df_taskb)}, cols: {len(df_taskb.columns)}")


if __name__ == "__main__":
    main()
