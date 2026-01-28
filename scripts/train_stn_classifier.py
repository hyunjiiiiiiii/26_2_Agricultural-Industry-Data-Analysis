"""
STN + ResNet 분류 학습 스크립트 (dataset_cls 기반)

폴더 구조:
project_root/
  dataset_cls/
    train/
      b1/ xxx.jpg ...
      b2/ ...
    val/
      b1/ ...
      b2/ ...
  scripts/
    train_stn_classifier.py

실행 예시:
  python scripts/train_stn_classifier.py --data dataset_cls --run stn_resnet18

불균형 대응(추천):
  python scripts/train_stn_classifier.py --data dataset_cls --run stn_resnet18 \
    --pretrained --epochs 30 --batch 64 --lr 1e-4 \
    --min_per_class 50 --oversample --class_weight
"""

import argparse
import json
import os
import time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# -------------------------
# STN 모듈 (Affine STN)
# -------------------------
class STN(nn.Module):
    """
    간단한 affine STN.
    입력: (B, 3, H, W)
    출력: (B, 3, H, W) (정렬/왜곡 보정된 이미지)
    """
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

        # feature map 크기 계산 (input_size 기준)
        # Conv7 -> (H-6), Pool2 -> /2
        # Conv5 -> (..-4), Pool2 -> /2
        # 224 -> conv7:218 -> pool:109 -> conv5:105 -> pool:52
        # channel 10 => 10*52*52
        fc_in = 10 * 52 * 52 if input_size == 224 else None
        if fc_in is None:
            # input_size 변경 시 대략적으로 계산해도 되지만, 여기선 안전하게 forward에서 한 번 계산 가능
            # 다만 안정성을 위해 224로 쓰는 걸 추천.
            raise ValueError("STN은 기본 imgsz=224 사용을 권장합니다.")

        self.fc_loc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.ReLU(True),
            nn.Linear(64, 6)  # 2x3 affine
        )

        # 초기값: identity transform
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
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class ReIndexDataset(Dataset):
    """
    ImageFolder의 라벨(id)이 원래 class_to_idx 기준이므로,
    keep_classes만 남기고 0..K-1로 재인덱싱하기 위한 래퍼.
    """
    def __init__(self, subset, old_to_new: Dict[int, int]):
        self.subset = subset
        self.old_to_new = old_to_new

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y_old = self.subset[idx]
        y_new = self.old_to_new[y_old]
        return x, y_new


def compute_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(targets)
    w = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        w[c] = 1.0 / max(counts.get(c, 1), 1)
    w = w / w.mean()  # 평균 1로 정규화
    return w


def make_sampler(targets: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = Counter(targets)
    class_w = {c: 1.0 / max(counts.get(c, 1), 1) for c in range(num_classes)}
    sample_w = [class_w[t] for t in targets]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    """
    반환: (val_loss, val_acc)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def pick_keep_classes(train_dir: Path, min_per_class: int) -> List[str]:
    """
    train split에서 클래스별 이미지 개수 확인 후,
    min_per_class 미만인 클래스는 제외.
    """
    keep = []
    for cls_dir in sorted([d for d in train_dir.iterdir() if d.is_dir()]):
        n = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            n += len(list(cls_dir.glob(ext)))
        if n >= min_per_class:
            keep.append(cls_dir.name)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="dataset_cls", help="dataset_cls root folder")
    ap.add_argument("--run", type=str, default="stn_resnet18", help="run name under runs_cls/")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--min_per_class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.imgsz != 224:
        raise ValueError("현재 STN 구현은 imgsz=224 기준으로 고정되어 있습니다. (권장: 224)")

    set_seed(args.seed)

    project_root = Path(__file__).parent.parent
    data_root = project_root / args.data
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    out_dir = project_root / "runs_cls" / args.run
    ensure_dir(out_dir)

    # 클래스 필터링
    keep_classes = pick_keep_classes(train_dir, args.min_per_class)
    if not keep_classes:
        raise RuntimeError("No classes meet min_per_class. Lower --min_per_class or check dataset_cls/train.")

    print(f"[INFO] keep_classes ({len(keep_classes)}): {keep_classes}")

    # Transform
    train_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.05)], p=0.7),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        # 정규화(선택). pretrained 쓰면 넣는 게 일반적
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if args.pretrained else transforms.Lambda(lambda x: x),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) if args.pretrained else transforms.Lambda(lambda x: x),
    ])

    # ImageFolder 로드
    base_train = datasets.ImageFolder(train_dir, transform=train_tf)
    base_val = datasets.ImageFolder(val_dir, transform=val_tf)

    # keep_classes만 남기는 subset 인덱스
    keep_set = set(keep_classes)
    train_indices = [i for i, (p, y) in enumerate(base_train.samples) if base_train.classes[y] in keep_set]
    val_indices = [i for i, (p, y) in enumerate(base_val.samples) if base_val.classes[y] in keep_set]

    train_subset = torch.utils.data.Subset(base_train, train_indices)
    val_subset = torch.utils.data.Subset(base_val, val_indices)

    # old idx -> new idx (0..K-1)
    old_to_new = {base_train.class_to_idx[c]: i for i, c in enumerate(keep_classes)}
    new_to_class = {i: c for c, i in old_to_new.items()}

    train_ds = ReIndexDataset(train_subset, old_to_new)
    val_ds = ReIndexDataset(val_subset, old_to_new)

    # targets 추출(oversample/weight용)
    train_targets_new: List[int] = []
    for i in train_indices:
        _, y_old = base_train.samples[i]
        train_targets_new.append(old_to_new[y_old])

    num_classes = len(keep_classes)

    # DataLoader
    sampler = None
    shuffle = True
    if args.oversample:
        sampler = make_sampler(train_targets_new, num_classes)
        shuffle = False
        print("[INFO] Using WeightedRandomSampler (oversampling).")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # model
    model = STNClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        imgsz=args.imgsz
    ).to(device)

    # loss
    ce_weight = None
    if args.class_weight:
        ce_weight = compute_class_weights(train_targets_new, num_classes).to(device)
        print("[INFO] Using class weights in CrossEntropyLoss.")
    criterion = nn.CrossEntropyLoss(weight=ce_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # logging
    results_csv = out_dir / "results.csv"
    mapping_json = out_dir / "class_mapping.json"
    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump({"classes": keep_classes, "old_to_new": {str(k): v for k, v in old_to_new.items()}},
                  f, ensure_ascii=False, indent=2)
    print(f"[INFO] saved class mapping: {mapping_json}")

    if not results_csv.exists():
        results_csv.write_text("epoch,train_loss,train_acc,val_loss,val_acc,elapsed_sec\n", encoding="utf-8")

    best_val_acc = -1.0
    best_path = out_dir / "best.pth"
    last_path = out_dir / "last.pth"

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        # save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": keep_classes,
            "args": vars(args),
        }, last_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": keep_classes,
                "args": vars(args),
            }, best_path)

        # log
        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{elapsed:.2f}\n")

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"best_val_acc={best_val_acc:.4f} | "
              f"epoch_time={elapsed:.1f}s")

    total = time.time() - start_time
    print(f"\n✅ Training done. total_time={total/60:.1f} min")
    print(f"best: {best_path}")
    print(f"last: {last_path}")
    print(f"logs: {results_csv}")


if __name__ == "__main__":
    main()
