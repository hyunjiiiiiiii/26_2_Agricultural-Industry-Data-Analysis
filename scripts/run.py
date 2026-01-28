import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO


# -------------------------
# (예시) STN + ResNet18 래퍼
# -------------------------
# ⚠️ IMPORTANT:
# - 너의 학습 코드에서 사용한 모델 클래스/구조가 다르면 이 부분만 맞춰줘야 함.
# - 아래는 "checkpoint에 model_state_dict만 저장"된 흔한 케이스 기준.
class STNResNet18(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # 여기에 너의 STN 모듈 + resnet18 정의가 있어야 함
        # >>> 너의 학습 코드 구조로 교체 필요 <<<
        # 아래는 placeholder (실제론 너의 STN+resnet18 구현을 그대로 가져와야 함)
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        # 실제로는: x -> STN -> ResNet18
        return self.backbone(x)


def load_stn_model(ckpt_path: str, device: str, num_classes: int) -> nn.Module:
    model = STNResNet18(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    # 다양한 저장 포맷 대응
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # 모델 state dict로 바로 저장된 경우
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # key prefix 정리(필요할 때만)
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def xyxy_to_int(xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy.tolist()
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", required=True, help="YOLOv11 best.pt path")
    ap.add_argument("--stn-ckpt", required=True, help="STN+ResNet18 checkpoint path")
    ap.add_argument("--image-dir", required=True, help="folder with images")
    ap.add_argument("--out-csv", required=True, help="output csv path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--phys-class-ids", default="0", help="comma-separated YOLO class ids considered 'phys'")
    ap.add_argument("--topk", type=int, default=1, help="how many top boxes to crop for STN (default 1)")
    ap.add_argument("--save", action="store_true",
                help="save YOLO prediction images with bboxes")
    ap.add_argument("--save-txt", action="store_true",
                help="save YOLO prediction labels (txt)")
    args = ap.parse_args()

    device = args.device
    phys_ids = set(int(x.strip()) for x in args.phys_class_ids.split(",") if x.strip() != "")

    # Load models
    yolo = YOLO(args.yolo_weights)
    stn_model = load_stn_model(args.stn_ckpt, device=device, num_classes=args.num_classes)

    # Image preprocess for STN/ResNet
    tfm = transforms.Compose([
        transforms.ToTensor(),
        # 학습 시 normalize 썼으면 동일하게 맞춰야 함
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_dir = Path(args.image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "has_any_det",
                "has_phys_det",
                "num_boxes",
                "yolo_conf_max",
                "yolo_phys_conf_max",
                "stn_top1_class",
                "stn_top1_prob",
            ],
        )
        writer.writeheader()

        # YOLO batch inference (Ultralytics가 내부적으로 배치 처리 가능)
        results = yolo.predict(
            source=[str(p) for p in img_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        for r in results:
            path = r.path
            im = cv2.imread(path)
            if im is None:
                continue
            h, w = im.shape[:2]

            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                row = {
                    "image_path": path,
                    "has_any_det": 0,
                    "has_phys_det": 0,
                    "num_boxes": 0,
                    "yolo_conf_max": 0.0,
                    "yolo_phys_conf_max": 0.0,
                    "stn_top1_class": -1,
                    "stn_top1_prob": 0.0,
                }
                writer.writerow(row)
                continue

            confs = boxes.conf.detach().cpu().numpy()
            clss = boxes.cls.detach().cpu().numpy().astype(int)
            xyxys = boxes.xyxy.detach().cpu().numpy()

            has_any_det = 1
            yolo_conf_max = float(confs.max())

            phys_mask = np.array([c in phys_ids for c in clss], dtype=bool)
            has_phys_det = int(phys_mask.any())
            yolo_phys_conf_max = float(confs[phys_mask].max()) if has_phys_det else 0.0

            # --- STN crop: topk by confidence (전체 중 상위 topk) ---
            order = np.argsort(-confs)
            top_idxs = order[: max(1, args.topk)]

            # top1 결과만 저장(필요하면 topk 다 저장하도록 확장 가능)
            stn_top1_class = -1
            stn_top1_prob = 0.0

            # 첫 번째 박스로만 STN 수행 (기본)
            idx0 = int(top_idxs[0])
            x1, y1, x2, y2 = xyxy_to_int(xyxys[idx0], w, h)
            crop = im[y1:y2, x1:x2]

            # crop이 너무 작으면 그냥 스킵
            if crop.size > 0:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # 학습 입력 사이즈에 맞춰 resize 필요 (예: 224)
                crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                x = tfm(crop_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = stn_model(x)
                    prob = torch.softmax(logits, dim=1)
                    pmax, pred = torch.max(prob, dim=1)
                    stn_top1_class = int(pred.item())
                    stn_top1_prob = float(pmax.item())

            row = {
                "image_path": path,
                "has_any_det": has_any_det,
                "has_phys_det": has_phys_det,
                "num_boxes": int(len(confs)),
                "yolo_conf_max": yolo_conf_max,
                "yolo_phys_conf_max": yolo_phys_conf_max,
                "stn_top1_class": stn_top1_class,
                "stn_top1_prob": stn_top1_prob,
            }
            writer.writerow(row)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
