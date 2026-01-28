"""
JSON 형식을 YOLO 형식으로 변환하는 스크립트
- dataset/meta/json/ 폴더의 JSON 파일들을 읽어서
- dataset/labels/ 폴더에 YOLO 형식의 .txt 파일로 변환
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

# tqdm을 선택적 의존성으로 처리
try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdm이 없으면 일반 함수로 대체
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


# 클래스 매핑: 폴더명 -> 클래스 ID
CLASS_MAPPING = {
    "정상": 0,
    "병해": 1,
    "생리장해": 2,
    "작물보호제처리반응": 3
}


def convert_bbox_to_yolo(x: float, y: float, w: float, h: float, 
                        img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    절대 좌표를 YOLO 정규화 좌표로 변환
    
    Args:
        x: 바운딩 박스 왼쪽 상단 x 좌표 (절대)
        y: 바운딩 박스 왼쪽 상단 y 좌표 (절대)
        w: 바운딩 박스 너비 (절대)
        h: 바운딩 박스 높이 (절대)
        img_width: 이미지 너비
        img_height: 이미지 높이
    
    Returns:
        (x_center, y_center, width, height) 정규화된 좌표 (0-1 사이)
    """
    # 중심 좌표 계산
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    
    # 정규화된 너비와 높이
    width = w / img_width
    height = h / img_height
    
    # 범위 검증 및 클리핑
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def process_json_file(json_path: Path, class_id: int, output_dir: Path, 
                     images_base_dir: Path) -> bool:
    """
    단일 JSON 파일을 처리하여 YOLO 형식의 .txt 파일 생성
    
    Args:
        json_path: JSON 파일 경로
        class_id: 클래스 ID
        output_dir: 출력 디렉토리 (labels/train 또는 labels/val)
        images_base_dir: 이미지 파일이 있는 기본 디렉토리
    
    Returns:
        성공 여부 (bool)
    """
    try:
        # JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 정보 추출
        description = data.get('description', {})
        img_width = description.get('width', 0)
        img_height = description.get('height', 0)
        image_filename = description.get('image', '')
        
        if img_width == 0 or img_height == 0:
            print(f"경고: {json_path}에서 이미지 크기 정보가 없습니다.")
            return False
        
        if not image_filename:
            print(f"경고: {json_path}에서 이미지 파일명이 없습니다.")
            return False
        
        # 이미지 파일 존재 확인 (선택사항, 경고만 출력)
        image_path = images_base_dir / image_filename
        if not image_path.exists():
            # 하위 폴더에서 찾기 시도
            found = False
            for subdir in images_base_dir.iterdir():
                if subdir.is_dir():
                    potential_path = subdir / image_filename
                    if potential_path.exists():
                        found = True
                        break
            
            if not found:
                print(f"경고: 이미지 파일을 찾을 수 없습니다: {image_filename}")
                # 이미지가 없어도 라벨 파일은 생성 (이미지가 나중에 추가될 수 있음)
        
        # 어노테이션 추출
        annotations = data.get('annotations', {})
        bbox_list = annotations.get('bbox', [])
        part_list = annotations.get('part', [])
        
        # YOLO 형식 라벨 생성
        yolo_lines = []
        
        # bbox 처리
        for bbox in bbox_list:
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            w = bbox.get('w', 0)
            h = bbox.get('h', 0)
            
            if w > 0 and h > 0:
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    x, y, w, h, img_width, img_height
                )
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # part 처리
        for part in part_list:
            x = part.get('x', 0)
            y = part.get('y', 0)
            w = part.get('w', 0)
            h = part.get('h', 0)
            
            if w > 0 and h > 0:
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    x, y, w, h, img_width, img_height
                )
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 출력 파일 경로 생성
        json_stem = json_path.stem
        output_file = output_dir / f"{json_stem}.txt"
        
        # 라벨 파일 저장
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))
            if yolo_lines:  # 마지막 줄에 개행 추가하지 않음
                f.write('\n')
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"오류: JSON 파싱 실패 - {json_path}: {e}")
        return False
    except Exception as e:
        print(f"오류: {json_path} 처리 중 예외 발생: {e}")
        return False


def process_dataset(json_base_dir: Path, images_base_dir: Path, 
                   labels_output_dir: Path, split: str = "train"):
    """
    전체 데이터셋 처리
    
    Args:
        json_base_dir: JSON 파일이 있는 기본 디렉토리 (TL5_토마토 또는 VL5_토마토)
        images_base_dir: 이미지 파일이 있는 기본 디렉토리 (images/train 또는 images/val)
        labels_output_dir: 라벨 출력 디렉토리 (labels/train 또는 labels/val)
        split: 데이터셋 분할 이름 ("train" 또는 "val")
    """
    print(f"\n{'='*60}")
    print(f"{split.upper()} 데이터셋 처리 시작")
    print(f"{'='*60}")
    
    if not json_base_dir.exists():
        print(f"경고: JSON 디렉토리가 존재하지 않습니다: {json_base_dir}")
        return
    
    # 클래스 폴더 순회
    total_files = 0
    success_count = 0
    fail_count = 0
    
    for class_folder_name, class_id in CLASS_MAPPING.items():
        class_json_dir = json_base_dir / class_folder_name
        
        if not class_json_dir.exists():
            print(f"경고: 클래스 폴더가 존재하지 않습니다: {class_json_dir}")
            continue
        
        print(f"\n클래스 '{class_folder_name}' (ID: {class_id}) 처리 중...")
        
        # JSON 파일 목록 가져오기
        json_files = list(class_json_dir.glob("*.json"))
        
        if not json_files:
            print(f"  JSON 파일이 없습니다: {class_json_dir}")
            continue
        
        # 각 JSON 파일 처리
        json_iter = tqdm(json_files, desc=f"  {class_folder_name}") if HAS_TQDM else json_files
        for idx, json_path in enumerate(json_iter, 1):
            total_files += 1
            if process_json_file(json_path, class_id, labels_output_dir, images_base_dir):
                success_count += 1
            else:
                fail_count += 1
            
            # tqdm이 없을 때 진행 상황 표시 (100개마다)
            if not HAS_TQDM and idx % 100 == 0:
                print(f"    진행 중: {idx}/{len(json_files)} 파일 처리 완료")
    
    print(f"\n{'='*60}")
    print(f"{split.upper()} 데이터셋 처리 완료")
    print(f"{'='*60}")
    print(f"전체 파일: {total_files}")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")


def main():
    """메인 함수"""
    # 프로젝트 루트 디렉토리
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"
    
    # 경로 설정
    json_train_dir = dataset_dir / "meta" / "json" / "TL5_토마토"
    json_val_dir = dataset_dir / "meta" / "json" / "VL5_토마토"
    
    images_train_dir = dataset_dir / "images" / "train"
    images_val_dir = dataset_dir / "images" / "val"
    
    labels_train_dir = dataset_dir / "labels" / "train"
    labels_val_dir = dataset_dir / "labels" / "val"
    
    # Train 데이터셋 처리
    if json_train_dir.exists():
        process_dataset(
            json_base_dir=json_train_dir,
            images_base_dir=images_train_dir,
            labels_output_dir=labels_train_dir,
            split="train"
        )
    else:
        print(f"경고: Train JSON 디렉토리가 존재하지 않습니다: {json_train_dir}")
    
    # Val 데이터셋 처리
    if json_val_dir.exists():
        process_dataset(
            json_base_dir=json_val_dir,
            images_base_dir=images_val_dir,
            labels_output_dir=labels_val_dir,
            split="val"
        )
    else:
        print(f"경고: Val JSON 디렉토리가 존재하지 않습니다: {json_val_dir}")
    
    print("\n모든 변환 작업이 완료되었습니다!")
    print(f"라벨 파일 위치:")
    print(f"  Train: {labels_train_dir}")
    print(f"  Val: {labels_val_dir}")


if __name__ == "__main__":
    main()
