"""
Pretrain 학습 스크립트
AI Hub 토마토 데이터로 YOLOv11 모델 pretrain
"""

import os
from pathlib import Path
import yaml
from ultralytics import YOLO


def train_pretrain(config_path="config.yaml"):
    """Pretrain 학습 실행"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']['pretrain']
    data_config = config['data']
    classes_config = config['classes']
    
    # 모델 초기화
    model = YOLO(f"{model_config['model_size']}.pt")
    
    # 데이터 경로 설정
    train_path = Path(data_config['processed']['train'])
    val_path = Path(data_config['processed']['val'])
    
    # YOLO 데이터셋 설정 파일 생성
    dataset_yaml = {
        'path': str(train_path.parent.absolute()),
        'train': f"train/images",
        'val': f"val/images",
        'test': f"test/images" if Path(data_config['processed']['test']).exists() else None,
        'names': {i: name for i, name in enumerate(classes_config['names'])},
        'nc': classes_config['num_classes']
    }
    
    # test 경로가 없으면 제거
    if dataset_yaml['test'] is None:
        del dataset_yaml['test']
    
    dataset_yaml_path = "data/dataset_pretrain.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print("=" * 50)
    print("Pretrain 학습 시작")
    print("=" * 50)
    print(f"모델: {model_config['model_size']}")
    print(f"Epochs: {model_config['epochs']}")
    print(f"이미지 크기: {model_config['imgsz']}")
    print(f"배치 크기: {model_config['batch']}")
    print(f"디바이스: {model_config['device']}")
    print(f"클래스: {classes_config['names']}")
    print("=" * 50)
    
    # 학습 실행
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
        name="pretrain",
        exist_ok=True,
        # 하이퍼파라미터
        lr0=config['training']['lr0'],
        lrf=config['training']['lrf'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['warmup_epochs'],
        warmup_momentum=config['training']['warmup_momentum'],
        warmup_bias_lr=config['training']['warmup_bias_lr'],
        box=config['training']['box'],
        cls=config['training']['cls'],
        dfl=config['training']['dfl'],
        # 데이터 증강
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
    )
    
    # 최종 가중치를 weights/pretrain으로 복사
    weights_dir = Path(config['paths']['weights_dir']) / "pretrain"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    best_weight = Path(results.save_dir) / "weights" / "best.pt"
    if best_weight.exists():
        import shutil
        shutil.copy2(best_weight, weights_dir / "best.pt")
        print(f"\n최종 가중치 저장: {weights_dir / 'best.pt'}")
    
    print("\nPretrain 학습 완료!")
    return results


if __name__ == "__main__":
    train_pretrain()

