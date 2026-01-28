import argparse
import json
from pathlib import Path
import pandas as pd


def load_json_area_grow(json_path: Path):
    """json에서 annotations.area, annotations.grow 추출"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ann = data.get("annotations", {})
    area = ann.get("area", None)
    grow = ann.get("grow", None)
    return area, grow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, default="val_ids.csv")
    ap.add_argument("--vl5", type=str, default="dataset/meta/json/VL5_토마토",
                    help="VL5_토마토 JSON 루트 폴더")
    ap.add_argument("--out", type=str, default="val_meta.csv", help="출력 CSV")
    ap.add_argument("--id_col", type=str, default="id", help="CSV에서 id 컬럼명")
    ap.add_argument("--suffix", type=str, default="", help="id 뒤에 붙은 확장자 제거용(보통 불필요)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    vl5_root = Path(args.vl5)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not vl5_root.exists():
        raise FileNotFoundError(f"VL5 root not found: {vl5_root}")

    df = pd.read_csv(csv_path)
    if args.id_col not in df.columns:
        raise ValueError(f"CSV에 '{args.id_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    # id 정리 (혹시 .jpg 같은 게 섞여있을 때 대비)
    ids = df[args.id_col].astype(str).tolist()
    cleaned_ids = []
    for s in ids:
        s2 = s.strip()
        s2 = s2.replace("\\", "/")
        s2 = Path(s2).stem  # "aaa.jpg" -> "aaa"
        if args.suffix and s2.endswith(args.suffix):
            s2 = s2[: -len(args.suffix)]
        cleaned_ids.append(s2)

    # 결과 컬럼 준비
    area_list = [None] * len(cleaned_ids)
    grow_list = [None] * len(cleaned_ids)
    found_list = [0] * len(cleaned_ids)  # json 찾았는지 표시

    # VL5_토마토 내부 하위 폴더(정상/병해/생리장해/작물보호제처리반응) 포함해서 탐색
    # id.json을 직접 찾기 위해 glob 사용 (필요한 id만)
    for i, cid in enumerate(cleaned_ids):
        # 1) 가장 흔한 위치: VL5_토마토/<클래스폴더>/<id>.json
        #    클래스폴더를 모르니 **로 recursive glob**
        matches = list(vl5_root.rglob(f"{cid}.json"))

        if not matches:
            found_list[i] = 0
            continue

        # 중복 매칭이 있을 수 있으니 첫 번째 사용(보통 1개)
        json_path = matches[0]
        try:
            area, grow = load_json_area_grow(json_path)
            area_list[i] = area
            grow_list[i] = grow
            found_list[i] = 1
        except Exception as e:
            # 파싱 실패 시 None 유지
            found_list[i] = 0

    # DF에 컬럼 추가
    df["area"] = area_list
    df["grow"] = grow_list
    df["meta_found"] = found_list  # 1이면 매핑 성공, 0이면 못 찾음/실패

    # 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 요약
    total = len(df)
    found = int(df["meta_found"].sum())
    print(f"[OK] saved: {out_path}")
    print(f"[INFO] rows: {total}")
    print(f"[INFO] meta_found: {found}/{total} (not found: {total - found})")


if __name__ == "__main__":
    main()
