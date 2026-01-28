import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


# -------------------------
# STN 모듈 (학습 코드와 동일)
# -------------------------
class STN(nn.Module):
    def __init__(self, input_size: int = 224):
        super().__init__()
        self.input_size = input_size

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        if input_size != 224:
            raise ValueError("현재 STN 구현은 imgsz=224 기준 고정입니다.")

        # 224 -> conv7:218 -> pool:109 -> conv5:105 -> pool:52, ch=10
        fc_in = 10 * 52 * 52
        self.fc_loc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)
        )

        # identity init
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x = nn.functional.grid_sample(x, grid, align_corners=False)
        return x


class STNClassifier(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True, imgsz: int = 224):
        super().__init__()
        self.stn = STN(input_size=imgsz)

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        self.backbone = net
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.stn(x)
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits


# -------------------------
# 유틸
# -------------------------
def list_images(src: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if src.is_file() and src.suffix.lower() in exts:
        return [src]
    files = []
    for p in src.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def clip_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def pad_box(x1, y1, x2, y2, pad_ratio: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    return clip_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# 메인
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, required=True, help="이미지 폴더 또는 단일 이미지 경로")
    ap.add_argument("--yolo", type=str, required=True, help="YOLO best.pt 경로")
    ap.add_argument("--stn", type=str, required=True, help="STN best.pth 경로")
    ap.add_argument("--out", type=str, default="outputs/taskB_features", help="출력 폴더")
    ap.add_argument("--conf", type=float, default=0.12, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.6, help="YOLO NMS IoU threshold")
    ap.add_argument("--pad", type=float, default=0.25, help="bbox padding ratio")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference imgsz")
    ap.add_argument("--phys_id", type=int, default=1, help="YOLO physiological class id (너는 1)")
    ap.add_argument("--save_crops", action="store_true", help="crop 이미지를 저장할지")
    ap.add_argument("--device", type=str, default="", help="YOLO device: ''=auto, 'cpu' or '0' 등")
    ap.add_argument("--stn_device", type=str, default="", help="STN device: ''=auto, 'cpu' or 'cuda'")
    ap.add_argument("--empty_value", type=float, default=0.0, help="탐지없음(STN skip)일 때 확률/통계값 채우는 값(0.0 추천)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # 1) 이미지 목록
    images = list_images(Path(args.source))
    if not images:
        raise FileNotFoundError(f"No images found at {args.source}")

    # 2) YOLO 로드
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    # 3) STN 로드 (탐지된 케이스만 돌릴 거지만 모델은 미리 로드)
    if args.stn_device.lower() == "cpu":
        stn_device = torch.device("cpu")
    elif args.stn_device.lower() == "cuda":
        stn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        stn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.stn, map_location="cpu")
    classes = ckpt.get("classes", None)
    if classes is None:
        raise RuntimeError("STN checkpoint에 'classes'가 없습니다.")
    num_classes = len(classes)

    train_args = ckpt.get("args", {})
    backbone = train_args.get("backbone", "resnet18")
    pretrained = bool(train_args.get("pretrained", True))

    stn_model = STNClassifier(num_classes=num_classes, backbone=backbone, pretrained=pretrained, imgsz=224)
    stn_model.load_state_dict(ckpt["model_state"], strict=True)
    stn_model.to(stn_device)
    stn_model.eval()

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if pretrained else transforms.Lambda(lambda x: x),
    ])

    # CSV 경로
    boxes_csv = out_dir / "features_boxes.csv"
    images_csv = out_dir / "features_images.csv"

    # 박스 단위 컬럼(탐지된 박스만)
    box_fields = [
        "id", "image_path", "image_name",
        "box_id",
        "yolo_conf", "yolo_cls",
        "x1", "y1", "x2", "y2",
        "crop_w", "crop_h",
        "stn_top1", "stn_top1_prob",
        "stn_entropy",
    ] + [f"p_{c}" for c in classes]

    # 이미지 단위 컬럼(모든 이미지)
    img_fields = [
        "id", "image_path", "image_name",
        "has_phys_det",          # 0/1
        "num_phys_boxes",
        "yolo_conf_max", "yolo_conf_mean",
        "stn_valid",             # 0/1 (STN을 실제로 수행했는지)
        "stn_pmax_max", "stn_pmax_mean",
        "stn_entropy_mean",
    ] + [f"p_{c}_max" for c in classes] + [f"p_{c}_mean" for c in classes]

    if args.save_crops:
        crops_dir = out_dir / "crops_phys"
        ensure_dir(crops_dir)

    img_rows: List[Dict[str, Any]] = []
    box_rows: List[Dict[str, Any]] = []

    # 4) 이미지별 처리
    for idx, im_path in enumerate(images, start=1):
        im_bgr = cv2.imread(str(im_path))
        if im_bgr is None:
            print(f"[WARN] failed to read: {im_path}")
            continue

        h, w = im_bgr.shape[:2]
        img_id = im_path.stem

        # YOLO 예측
        results = yolo.predict(
            source=str(im_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False
        )
        r = results[0]

        # 기본: 탐지 없음 row (STN 완전 skip)
        empty_row = {
            "id": img_id,
            "image_path": str(im_path),
            "image_name": im_path.name,
            "has_phys_det": 0,
            "num_phys_boxes": 0,
            "yolo_conf_max": args.empty_value,
            "yolo_conf_mean": args.empty_value,
            "stn_valid": 0,
            "stn_pmax_max": args.empty_value,
            "stn_pmax_mean": args.empty_value,
            "stn_entropy_mean": args.empty_value,
        }
        for c in classes:
            empty_row[f"p_{c}_max"] = args.empty_value
            empty_row[f"p_{c}_mean"] = args.empty_value

        if r.boxes is None or len(r.boxes) == 0:
            img_rows.append(empty_row)
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        keep = np.where(clss == args.phys_id)[0].tolist()
        if len(keep) == 0:
            img_rows.append(empty_row)
            continue

        # 여기부터 "phys 탐지 있음" → STN 수행
        yolo_conf_list = []
        stn_pmax_list = []
        ent_list = []
        p_stack = []

        for b_i, j in enumerate(keep):
            x1, y1, x2, y2 = xyxy[j]
            x1, y1, x2, y2 = pad_box(x1, y1, x2, y2, args.pad, w, h)

            crop = im_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]

            if args.save_crops:
                save_name = f"{img_id}_box{b_i:02d}.jpg"
                cv2.imwrite(str((out_dir / "crops_phys" / save_name)), crop)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)

            x = tf(crop_rgb).unsqueeze(0).to(stn_device)

            with torch.no_grad():
                logits = stn_model(x).squeeze(0).detach().cpu().numpy()

            probs = softmax_np(logits)
            top1_idx = int(probs.argmax())
            top1_cls = classes[top1_idx]
            top1_prob = float(probs[top1_idx])
            entropy = float(-(probs * np.log(probs + 1e-12)).sum())

            # 박스 row
            row = {
                "id": img_id,
                "image_path": str(im_path),
                "image_name": im_path.name,
                "box_id": b_i,
                "yolo_conf": float(confs[j]),
                "yolo_cls": int(clss[j]),
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "crop_w": int(crop_w), "crop_h": int(crop_h),
                "stn_top1": top1_cls,
                "stn_top1_prob": top1_prob,
                "stn_entropy": entropy,
            }
            for c_i, c in enumerate(classes):
                row[f"p_{c}"] = float(probs[c_i])
            box_rows.append(row)

            # 이미지 집계
            yolo_conf_list.append(float(confs[j]))
            stn_pmax_list.append(top1_prob)
            ent_list.append(entropy)
            p_stack.append(probs)

        # 유효 박스가 하나도 없으면(empty 취급)
        if len(p_stack) == 0:
            img_rows.append(empty_row)
            continue

        P = np.stack(p_stack, axis=0)  # (B, C)
        p_max = P.max(axis=0)
        p_mean = P.mean(axis=0)

        img_row = {
            "id": img_id,
            "image_path": str(im_path),
            "image_name": im_path.name,
            "has_phys_det": 1,
            "num_phys_boxes": int(len(p_stack)),
            "yolo_conf_max": float(np.max(yolo_conf_list)),
            "yolo_conf_mean": float(np.mean(yolo_conf_list)),
            "stn_valid": 1,
            "stn_pmax_max": float(np.max(stn_pmax_list)),
            "stn_pmax_mean": float(np.mean(stn_pmax_list)),
            "stn_entropy_mean": float(np.mean(ent_list)),
        }
        for c_i, c in enumerate(classes):
            img_row[f"p_{c}_max"] = float(p_max[c_i])
            img_row[f"p_{c}_mean"] = float(p_mean[c_i])

        img_rows.append(img_row)

        if idx % 50 == 0:
            print(f"[INFO] processed {idx}/{len(images)} images...")

    # 5) CSV 저장
    with open(boxes_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=box_fields)
        wri.writeheader()
        for r in box_rows:
            wri.writerow({k: r.get(k, "") for k in box_fields})

    with open(images_csv, "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=img_fields)
        wri.writeheader()
        for r in img_rows:
            wri.writerow({k: r.get(k, 0) for k in img_fields})

    print("\n Done.")
    print(f"- images(all): {images_csv}")
    print(f"- boxes(det only): {boxes_csv}")
    if args.save_crops:
        print(f"- crops: {out_dir / 'crops_phys'}")


if __name__ == "__main__":
    main()
