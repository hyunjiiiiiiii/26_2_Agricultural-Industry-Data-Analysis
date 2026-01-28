import argparse
import pandas as pd
from pathlib import Path


# ✅ features_images.csv 에서 가져올 컬럼들
TASKB_FEATURES = [
    "id",
    "stn_pmax_max",
    "stn_entropy_mean",
    "p_b2_max",
    "p_b3_max",
    "p_b6_max",
    "p_b7_max",
    "p_b8_max",
]

def read_table(path: Path) -> pd.DataFrame:
    """csv/xlsx 자동 로드"""
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="features_images.csv 경로")
    ap.add_argument("--output_data", type=str, required=True, help="id가 이미 들어있는 데이터(csv/xlsx)")
    ap.add_argument("--id_col", type=str, default="id", help="output_data에서 id 컬럼명 (default: id)")
    ap.add_argument("--output", type=str, default="val_taskb_matched.csv", help="저장 경로")
    ap.add_argument(
        "--how", type=str, default="inner", choices=["inner", "left"],
        help="merge 방식: inner=교집합만, left=output_data 기준 유지 (default: inner)"
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    outdata_path = Path(args.output_data)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if not outdata_path.exists():
        raise FileNotFoundError(f"Output-data file not found: {outdata_path}")

    # 1) 로드
    if in_path.suffix.lower() in [".xlsx", ".xls"]:
        df_feat = pd.read_excel(in_path)
    else:
        df_feat = pd.read_csv(in_path)

    df_out = read_table(outdata_path)

    # 2) id 컬럼 체크
    if "id" not in df_feat.columns:
        raise ValueError("features_images.csv 에 'id' 컬럼이 없습니다.")
    if args.id_col not in df_out.columns:
        raise ValueError(f"output_data 에 '{args.id_col}' 컬럼이 없습니다. (지금 컬럼들: {list(df_out.columns)[:30]}...)")

    # 3) 필요한 feature 컬럼 존재 여부
    missing = [c for c in TASKB_FEATURES if c not in df_feat.columns]
    if missing:
        raise ValueError(f"features_images.csv 에 필요한 컬럼이 없습니다: {missing}")

    # 4) output_data의 id만 남기기 (필터)
    df_out = df_out.copy()
    df_out[args.id_col] = df_out[args.id_col].astype(str).str.strip()

    df_feat = df_feat.copy()
    df_feat["id"] = df_feat["id"].astype(str).str.strip()

    out_ids = set(df_out[args.id_col].dropna().unique().tolist())
    df_feat = df_feat[df_feat["id"].isin(out_ids)].copy()

    # 5) TaskB feature만 선택
    df_taskb = df_feat[TASKB_FEATURES].copy()

    # 6) 타입 정리(안전)
    float_cols = [c for c in TASKB_FEATURES if c != "id"]
    for c in float_cols:
        df_taskb[c] = pd.to_numeric(df_taskb[c], errors="coerce").fillna(0.0).astype(float)

    # 확률/점수 클립(필요 시)
    prob_cols = ["stn_pmax_max","p_b2_max","p_b3_max","p_b6_max","p_b7_max","p_b8_max"]
    for c in prob_cols:
        if c in df_taskb.columns:
            df_taskb[c] = df_taskb[c].clip(0.0, 1.0)

    # 7) id로 merge (output_data에 있는 라벨/메타 + taskB feature 합치기)
    # output_data의 id_col 이름을 'id'로 맞춰 merge
    df_out_renamed = df_out.rename(columns={args.id_col: "id"})
    merged = df_out_renamed.merge(df_taskb, on="id", how=args.how)

    # 8) 저장
    merged.to_csv(out_path, index=False, encoding="utf-8")

    # 9) 리포트
    print(f"✅ saved: {out_path}")
    print(f"   output_data rows: {len(df_out)}")
    print(f"   matched feature rows: {len(df_taskb)}")
    print(f"   merged rows: {len(merged)} / cols: {len(merged.columns)}")

    # 매칭 안 된 id 개수(참고)
    if args.how == "left":
        miss = merged["stn_pmax_max"].isna().sum() if "stn_pmax_max" in merged.columns else 0
        print(f"   (left merge) rows with missing features: {miss}")

if __name__ == "__main__":
    main()
