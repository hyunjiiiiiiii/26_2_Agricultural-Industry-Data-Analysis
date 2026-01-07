"""
데이터 준비 스크립트
- 라벨 파일 검증 및 변환
- 데이터셋 분할 (train/val/test)
- YOLO 형식으로 데이터 구조 정리
"""

import os
import shutil
import random
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2
import numpy as np


class DataPreparer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.classes = self.config['classes']['names']
        
    def validate_labels(self, images_dir, labels_dir):
        """라벨 파일 검증"""
        print("라벨 파일 검증 중...")
        images = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
        labels = list(Path(labels_dir).glob("*.txt"))
        
        valid_pairs = []
        invalid_images = []
        invalid_labels = []
        
        for img_path in tqdm(images):
            label_path = Path(labels_dir) / (img_path.stem + ".txt")
            
            if not label_path.exists():
                invalid_images.append(img_path)
                continue
            
            # 라벨 파일 검증
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            raise ValueError(f"Invalid label format: {line}")
                        class_id, x, y, w, h = map(float, parts)
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            raise ValueError(f"Invalid bbox coordinates: {line}")
            except Exception as e:
                invalid_labels.append((label_path, str(e)))
                continue
            
            valid_pairs.append((img_path, label_path))
        
        print(f"\n검증 완료:")
        print(f"  유효한 이미지-라벨 쌍: {len(valid_pairs)}")
        print(f"  라벨 없는 이미지: {len(invalid_images)}")
        print(f"  잘못된 라벨: {len(invalid_labels)}")
        
        if invalid_labels:
            print("\n잘못된 라벨 파일:")
            for label, error in invalid_labels[:10]:
                print(f"  {label}: {error}")
        
        return valid_pairs
    
    def split_dataset(self, data_pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        """데이터셋 분할"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율 합이 1이어야 합니다"
        
        random.seed(seed)
        random.shuffle(data_pairs)
        
        total = len(data_pairs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data_pairs[:train_end]
        val_data = data_pairs[train_end:val_end]
        test_data = data_pairs[val_end:]
        
        print(f"\n데이터셋 분할:")
        print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
        print(f"  Val: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
        print(f"  Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def copy_to_processed(self, data_pairs, split_name, output_dir):
        """처리된 데이터로 복사"""
        images_dir = Path(output_dir) / "images"
        labels_dir = Path(output_dir) / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, label_path in tqdm(data_pairs, desc=f"{split_name} 복사 중"):
            shutil.copy2(img_path, images_dir / img_path.name)
            shutil.copy2(label_path, labels_dir / label_path.name)
    
    def prepare_pretrain_data(self):
        """Pretrain 데이터 준비"""
        print("=" * 50)
        print("Pretrain 데이터 준비 (AI Hub 토마토 데이터)")
        print("=" * 50)
        
        images_dir = self.data_config['pretrain']['images']
        labels_dir = self.data_config['pretrain']['labels']
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"경고: {images_dir} 또는 {labels_dir}가 존재하지 않습니다.")
            return
        
        # 라벨 검증
        valid_pairs = self.validate_labels(images_dir, labels_dir)
        
        if not valid_pairs:
            print("유효한 데이터가 없습니다.")
            return
        
        # 데이터 분할
        train_data, val_data, test_data = self.split_dataset(valid_pairs)
        
        # 처리된 데이터로 복사
        base_dir = Path(self.data_config['processed']['train']).parent
        self.copy_to_processed(train_data, "train", base_dir / "train")
        self.copy_to_processed(val_data, "val", base_dir / "val")
        self.copy_to_processed(test_data, "test", base_dir / "test")
        
        print("\nPretrain 데이터 준비 완료!")
    
    def prepare_finetune_data(self):
        """Fine-tune 데이터 준비"""
        print("=" * 50)
        print("Fine-tune 데이터 준비 (방울토마토 소량 데이터)")
        print("=" * 50)
        
        images_dir = self.data_config['finetune']['images']
        labels_dir = self.data_config['finetune']['labels']
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"경고: {images_dir} 또는 {labels_dir}가 존재하지 않습니다.")
            return
        
        # 라벨 검증
        valid_pairs = self.validate_labels(images_dir, labels_dir)
        
        if not valid_pairs:
            print("유효한 데이터가 없습니다.")
            return
        
        # 소량 데이터이므로 train/val만 분할 (test는 선택사항)
        random.seed(42)
        random.shuffle(valid_pairs)
        
        split_idx = int(len(valid_pairs) * 0.8)
        train_data = valid_pairs[:split_idx]
        val_data = valid_pairs[split_idx:]
        
        print(f"\nFine-tune 데이터 분할:")
        print(f"  Train: {len(train_data)}")
        print(f"  Val: {len(val_data)}")
        
        # 처리된 데이터로 복사 (별도 디렉토리)
        finetune_dir = Path("data/processed/finetune")
        self.copy_to_processed(train_data, "train", finetune_dir / "train")
        self.copy_to_processed(val_data, "val", finetune_dir / "val")
        
        print("\nFine-tune 데이터 준비 완료!")


if __name__ == "__main__":
    preparer = DataPreparer()
    
    # Pretrain 데이터 준비
    preparer.prepare_pretrain_data()
    
    # Fine-tune 데이터 준비
    preparer.prepare_finetune_data()

