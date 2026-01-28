"""
추론 스크립트
- 단일 이미지 추론
- 영상 추론
- 배치 추론
"""

import os
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import cv2


def inference_image(model, image_path, output_dir, config):
    """단일 이미지 추론"""
    print(f"이미지 추론: {image_path}")
    
    results = model.predict(
    source=str(image_path),
    imgsz=config['inference']['imgsz'],
    conf=config['inference']['conf_threshold'],
    iou=config['inference']['iou_threshold'],
    max_det=config['inference']['max_det'],
    save=True,
    save_txt=True, 
    project=output_dir,
    name="inference",
    exist_ok=True,
    )
    
    return results


def inference_video(model, video_path, output_dir, config):
    """영상 추론"""
    print(f"영상 추론: {video_path}")
    
    results = model.predict(
        source=str(video_path),
        imgsz=config['inference']['imgsz'],
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        max_det=config['inference']['max_det'],
        save=True,
        project=output_dir,
        name="inference",
        exist_ok=True,
    )
    
    return results


def inference_batch(model, images_dir, output_dir, config):
    """배치 추론"""
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    
    print(f"배치 추론: {len(image_files)}개 이미지")
    
    results = model.predict(
        source=str(images_dir),
        imgsz=config['inference']['imgsz'],
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        max_det=config['inference']['max_det'],
        save=True,
        project=output_dir,
        name="inference",
        exist_ok=True,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 추론 스크립트")
    parser.add_argument("--weights", type=str, required=True, help="모델 가중치 경로")
    parser.add_argument("--source", type=str, required=True, help="입력 소스 (이미지/영상/디렉토리)")
    parser.add_argument("--output", type=str, default="results/inference", help="출력 디렉토리")
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    
    args = parser.parse_args()
    
    # 설정 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 모델 로드
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {args.weights}")
    
    model = YOLO(args.weights)
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 소스 타입 확인
    source_path = Path(args.source)
    
    if source_path.is_file():
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 이미지
            inference_image(model, source_path, output_dir, config)
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            # 영상
            inference_video(model, source_path, output_dir, config)
        else:
            print(f"지원하지 않는 파일 형식: {source_path.suffix}")
    elif source_path.is_dir():
        # 디렉토리 (배치 추론)
        inference_batch(model, source_path, output_dir, config)
    else:
        print(f"소스를 찾을 수 없습니다: {args.source}")
    
    print(f"\n추론 완료! 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()

