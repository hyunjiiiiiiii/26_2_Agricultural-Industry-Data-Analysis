"""
dataset_cls 자동 생성 스크립트 (GT part 기반, 생리장해 disease 세부분류용)

- 입력:
  dataset/meta/json/TL5_토마토/생리장해/*.json
  dataset/meta/json/VL5_토마토/생리장해/*.json
  dataset/images/train/**.jpg
  dataset/images/val/**.jpg

- 출력:
  dataset_cls/train/<disease_code>/*.jpg
  dataset_cls/val/<disease_code>/*.jpg

설명:
- JSON의 annotations.part[] 박스를 사용해 원본 이미지를 crop하여 patch 이미지를 생성
- patch는 분류 모델(STN+classifier) 학습용 데이터
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List

# Pillow 사용
from PIL import Image

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


# =========================
# 설정값 (필요시 여기만 수정)
# =========================
PHYS_FOLDER_NAME = "생리장해"   # JSON 폴더명
OUTPUT_ROOT_NAME = "dataset_cls"
OUT_SIZE = 224                 # patch 최종 resize 크기 (224 추천)
PADDING_RATIO = 0.15           # bbox 주변 padding 비율 (0.10~0.25 추천)
MIN_BOX_SIZE = 20              # 너무 작은 박스는 제외(픽셀 기준)
JPEG_QUALITY = 95              # 저장 품질


def find_image(images_base_dir: Path, image_filename: str) -> Optional[Path]:
    """
    images_base_dir 아래에서 image_filename을 찾는다.
    - images_base_dir 바로 아래에 있거나,
    - 하위 폴더에도 있을 수 있어서 탐색
    """
    p = images_base_dir / image_filename
    if p.exists():
        return p

    # 하위 폴더 탐색(1-depth)
    for sub in images_base_dir.iterdir():
        if sub.is_dir():
            pp = sub / image_filename
            if pp.exists():
                return pp
    return None


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def pad_bbox(
    x: float, y: float, w: float, h: float,
    img_w: int, img_h: int,
    pad_ratio: float
) -> Tuple[int, int, int, int]:
    """
    bbox에 padding을 주고 이미지 범위로 clip한 (x1,y1,x2,y2) 반환
    """
    # padding (bbox 크기 기준)
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x1 = int(x) - pad_w
    y1 = int(y) - pad_h
    x2 = int(x + w) + pad_w
    y2 = int(y + h) + pad_h

    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 1, img_w)      # x2/y2는 crop에서 끝좌표라 img_w/img_h까지 허용
    y2 = clamp(y2, 1, img_h)

    # 이상한 경우 보정
    if x2 <= x1 + 1:
        x2 = clamp(x1 + 2, 2, img_w)
    if y2 <= y1 + 1:
        y2 = clamp(y1 + 2, 2, img_h)

    return x1, y1, x2, y2


def parse_json(json_path: Path) -> Optional[Tuple[str, str, List[dict]]]:
    """
    JSON에서 (image_filename, disease_code, part_list) 추출
    - 생리장해 disease(b*)를 분류 라벨로 사용
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        desc = data.get("description", {})
        ann = data.get("annotations", {})

        image_filename = desc.get("image")
        disease_code = ann.get("disease")  # 예: b1, b6, b8 ...
        parts = ann.get("part", [])

        if not image_filename or not disease_code:
            return None

        # part가 없으면 patch 생성 불가 -> 스킵
        if not parts:
            return None

        return image_filename, disease_code, parts

    except Exception as e:
        print(f"[ERROR] JSON 파싱 실패: {json_path.name} ({e})")
        return None


def save_patch(
    img: Image.Image,
    crop_box: Tuple[int, int, int, int],
    out_path: Path,
    out_size: int
) -> None:
    """
    crop -> resize -> save
    """
    patch = img.crop(crop_box)
    patch = patch.resize((out_size, out_size), resample=Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    patch.save(out_path, format="JPEG", quality=JPEG_QUALITY)


def process_split(
    split_name: str,
    json_root: Path,
    images_root: Path,
    out_root: Path,
) -> None:
    """
    split_name: train/val
    json_root: dataset/meta/json/TL5_토마토 or VL5_토마토
    images_root: dataset/images/train or val
    out_root: dataset_cls/train or val
    """
    class_dir = json_root / PHYS_FOLDER_NAME
    if not class_dir.exists():
        print(f"[WARN] JSON 폴더 없음: {class_dir}")
        return

    json_files = list(class_dir.glob("*.json"))
    it = tqdm(json_files, desc=f"{split_name} json") if HAS_TQDM else json_files

    made = 0
    skipped = 0

    for jp in it:
        parsed = parse_json(jp)
        if parsed is None:
            skipped += 1
            continue

        image_filename, disease_code, parts = parsed
        img_path = find_image(images_root, image_filename)
        if img_path is None:
            skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                img_w, img_h = im.size

                for i, part in enumerate(parts):
                    x = float(part.get("x", 0))
                    y = float(part.get("y", 0))
                    w = float(part.get("w", 0))
                    h = float(part.get("h", 0))

                    # 너무 작은 박스 제외
                    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                        continue

                    x1, y1, x2, y2 = pad_bbox(x, y, w, h, img_w, img_h, PADDING_RATIO)

                    # 저장 파일명: 원본stem + part index
                    stem = Path(image_filename).stem
                    out_path = out_root / disease_code / f"{stem}_p{i}.jpg"

                    save_patch(im, (x1, y1, x2, y2), out_path, OUT_SIZE)
                    made += 1

        except Exception as e:
            print(f"[ERROR] 이미지 처리 실패: {img_path.name} ({e})")
            skipped += 1
            continue

    print(f"\n[{split_name}] patches made: {made}, skipped_json_or_img: {skipped}")
    print(f"[{split_name}] output: {out_root}")


def main():
    # scripts/ 아래에 파일이 있으므로 프로젝트 루트는 parent.parent
    project_root = Path(__file__).parent.parent

    dataset_dir = project_root / "dataset"

    # 입력 경로
    train_json_root = dataset_dir / "meta" / "json" / "TL5_토마토"
    val_json_root   = dataset_dir / "meta" / "json" / "VL5_토마토"

    train_images_root = dataset_dir / "images" / "train"
    val_images_root   = dataset_dir / "images" / "val"

    # 출력 경로
    out_base = project_root / OUTPUT_ROOT_NAME
    out_train = out_base / "train"
    out_val   = out_base / "val"

    out_base.mkdir(parents=True, exist_ok=True)

    print("=== dataset_cls 생성 시작 (GT part 기반) ===")
    print(f"OUT_SIZE={OUT_SIZE}, PADDING_RATIO={PADDING_RATIO}, MIN_BOX_SIZE={MIN_BOX_SIZE}")
    print(f"Output root: {out_base}")

    process_split("train", train_json_root, train_images_root, out_train)
    process_split("val",   val_json_root,   val_images_root,   out_val)

    print("\n Done. dataset_cls/ 생성 완료")


if __name__ == "__main__":
    main()
