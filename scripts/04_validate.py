"""
검증 및 에러 분석 스크립트
- Class별 mAP 계산
- Confusion Matrix 생성
- 미탐지 분석
"""

import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from collections import defaultdict


def validate_model(config_path="config.yaml", weights_path=None):
    """모델 검증 및 에러 분석"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if weights_path is None:
        weights_path = Path(config['paths']['weights_dir']) / "finetune" / "best.pt"
    
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {weights_path}")
    
    # 모델 로드
    model = YOLO(str(weights_path))
    
    # 검증 데이터 경로
    val_config = config['validation']
    data_config = config['data']
    
    # Fine-tune 데이터셋 사용
    dataset_yaml_path = "data/dataset_finetune.yaml"
    if not Path(dataset_yaml_path).exists():
        # Pretrain 데이터셋 사용
        dataset_yaml_path = "data/dataset_pretrain.yaml"
    
    print("=" * 50)
    print("모델 검증 시작")
    print("=" * 50)
    print(f"가중치: {weights_path}")
    print(f"데이터셋: {dataset_yaml_path}")
    print("=" * 50)
    
    # 검증 실행
    results = model.val(
        data=dataset_yaml_path,
        imgsz=config['model']['finetune']['imgsz'],
        conf=val_config['conf_threshold'],
        iou=val_config['iou_threshold'],
        max_det=val_config['max_det'],
        save_json=val_config['save_json'],
        save_hybrid=val_config['save_hybrid'],
    )
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("검증 결과")
    print("=" * 50)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    if hasattr(results.box, 'maps'):
        print("\nClass별 mAP50-95:")
        class_names = config['classes']['names']
        for i, (class_name, map_value) in enumerate(zip(class_names, results.box.maps)):
            print(f"  {class_name}: {map_value:.4f}")
    
    # Confusion Matrix 생성
    create_confusion_matrix(model, dataset_yaml_path, config)
    
    # 미탐지 분석
    analyze_missed_detections(model, dataset_yaml_path, config)
    
    return results


def create_confusion_matrix(model, dataset_yaml_path, config):
    """Confusion Matrix 생성"""
    print("\n" + "=" * 50)
    print("Confusion Matrix 생성 중...")
    print("=" * 50)
    
    # 검증 실행하여 confusion matrix 데이터 수집
    results = model.val(
        data=dataset_yaml_path,
        imgsz=config['model']['finetune']['imgsz'],
        conf=config['validation']['conf_threshold'],
        iou=config['validation']['iou_threshold'],
    )
    
    # Confusion Matrix 저장 경로
    confusion_dir = Path(config['paths']['confusion_dir'])
    confusion_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion Matrix 플롯이 있다면 저장
    if hasattr(results, 'confusion_matrix'):
        confusion_path = confusion_dir / "confusion_matrix.png"
        if Path(confusion_path).exists():
            print(f"Confusion Matrix 저장: {confusion_path}")
    
    print("Confusion Matrix 생성 완료")


def analyze_missed_detections(model, dataset_yaml_path, config):
    """미탐지 분석"""
    print("\n" + "=" * 50)
    print("미탐지 분석 중...")
    print("=" * 50)
    
    from ultralytics.data import YOLODataset
    
    # 데이터셋 로드
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    val_path = Path(dataset_config['path']) / dataset_config['val']
    
    if not val_path.exists():
        print(f"검증 데이터 경로를 찾을 수 없습니다: {val_path}")
        return
    
    # 이미지 파일 목록
    image_files = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))
    
    missed_detections = defaultdict(list)
    class_names = config['classes']['names']
    
    print(f"검증 이미지 수: {len(image_files)}")
    
    for img_path in image_files[:100]:  # 샘플링 (전체 분석 시 제거)
        # 추론
        results = model.predict(
            str(img_path),
            imgsz=config['model']['finetune']['imgsz'],
            conf=config['validation']['conf_threshold'],
            iou=config['validation']['iou_threshold'],
            verbose=False
        )
        
        # 라벨 로드
        label_path = val_path.parent / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        # Ground truth 로드
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    gt_boxes.append((class_id, float(parts[1]), float(parts[2]), 
                                   float(parts[3]), float(parts[4])))
        
        # 예측 결과
        pred_boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                pred_boxes.append((class_id, conf, box.xywhn[0].tolist()))
        
        # 미탐지 분석 (간단한 버전)
        detected_classes = set([p[0] for p in pred_boxes])
        gt_classes = set([g[0] for g in gt_boxes])
        
        for gt_class in gt_classes:
            if gt_class not in detected_classes:
                missed_detections[class_names[gt_class]].append(str(img_path))
    
    # 결과 출력
    print("\n미탐지 통계:")
    for class_name, missed_list in missed_detections.items():
        print(f"  {class_name}: {len(missed_list)}개 이미지에서 미탐지")
        if len(missed_list) > 0:
            print(f"    예시: {missed_list[0]}")
    
    # 결과 저장
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "missed_detections.txt", 'w', encoding='utf-8') as f:
        for class_name, missed_list in missed_detections.items():
            f.write(f"{class_name}:\n")
            for img_path in missed_list:
                f.write(f"  {img_path}\n")
            f.write("\n")
    
    print(f"\n미탐지 분석 결과 저장: {results_dir / 'missed_detections.txt'}")


if __name__ == "__main__":
    import sys
    
    weights_path = sys.argv[1] if len(sys.argv) > 1 else None
    validate_model(weights_path=weights_path)

