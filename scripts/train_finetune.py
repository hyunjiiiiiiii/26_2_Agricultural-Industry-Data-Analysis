"""
Fine-tune 학습 스크립트
방울토마토 소량 데이터로 도메인 적응 학습
작은 병반 대응을 위한 고해상도 및 타일링 적용
"""

import os
from pathlib import Path
import yaml
from ultralytics import YOLO


def train_finetune(config_path="config.yaml"):
    """Fine-tune 학습 실행"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']['finetune']
    data_config = config['data']
    classes_config = config['classes']
    
    # Pretrain 가중치 로드
    weights_path = model_config['weights']
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Pretrain 가중치를 찾을 수 없습니다: {weights_path}")
    
    print(f"Pretrain 가중치 로드: {weights_path}")
    model = YOLO(weights_path)
    
    # Fine-tune 데이터 경로 설정
    finetune_dir = Path("data/processed/finetune")
    train_path = finetune_dir / "train"
    val_path = finetune_dir / "val"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Fine-tune 데이터를 찾을 수 없습니다: {train_path} 또는 {val_path}")
    
    # YOLO 데이터셋 설정 파일 생성
    dataset_yaml = {
        'path': str(finetune_dir.absolute()),
        'train': "train/images",
        'val': "val/images",
        'names': {i: name for i, name in enumerate(classes_config['names'])},
        'nc': classes_config['num_classes']
    }
    
    dataset_yaml_path = "data/dataset_finetune.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print("=" * 50)
    print("Fine-tune 학습 시작 (작은 병반 대응)")
    print("=" * 50)
    print(f"Pretrain 가중치: {weights_path}")
    print(f"모델: {model_config['model_size']}")
    print(f"Epochs: {model_config['epochs']}")
    print(f"이미지 크기: {model_config['imgsz']} (고해상도)")
    print(f"배치 크기: {model_config['batch']}")
    print(f"디바이스: {model_config['device']}")
    print(f"타일링: {config['augmentation']['tile_size']} (overlap: {config['augmentation']['tile_overlap']})")
    print("=" * 50)
    
    # 학습 실행 (고해상도, 작은 객체 대응)
    results = model.train(
        data=dataset_yaml_path,
        epochs=model_config['epochs'],
        imgsz=model_config['imgsz'],
        batch=model_config['batch'],
        device=model_config['device'],
        workers=model_config['workers'],
        patience=model_config['patience'],
        save_period=model_config['save_period'],
        project=config['paths']['weights_dir'],
        name="finetune",
        exist_ok=True,
        # Fine-tune을 위한 낮은 학습률
        lr0=config['training']['lr0'] * 0.1,  # Pretrain보다 낮은 학습률
        lrf=config['training']['lrf'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['warmup_epochs'],
        warmup_momentum=config['training']['warmup_momentum'],
        warmup_bias_lr=config['training']['warmup_bias_lr'],
        box=config['training']['box'],
        cls=config['training']['cls'],
        dfl=config['training']['dfl'],
        # 작은 객체 대응을 위한 강화된 증강
        mosaic=config['augmentation']['mosaic'],
        mixup=config['augmentation']['mixup'],
        copy_paste=config['augmentation']['copy_paste'],
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        shear=config['augmentation']['shear'],
        perspective=config['augmentation']['perspective'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        # 작은 객체 탐지를 위한 추가 설정
        close_mosaic=10,  # 마지막 10 epoch은 mosaic 비활성화
    )
    
    # 최종 가중치를 weights/finetune으로 복사
    weights_dir = Path(config['paths']['weights_dir']) / "finetune"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    if best_weight.exists():
        import shutil
        shutil.copy2(best_weight, weights_dir / "best.pt")
        print(f"\n최종 가중치 저장: {weights_dir / 'best.pt'}")
    
    print("\nFine-tune 학습 완료!")
    return results


if __name__ == "__main__":
    train_finetune()

