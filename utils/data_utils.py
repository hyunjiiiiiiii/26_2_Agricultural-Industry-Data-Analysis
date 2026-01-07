"""
데이터 유틸리티 함수
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def resize_with_padding(image, target_size, fill_value=114):
    """패딩을 추가하여 이미지 리사이즈"""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 패딩 추가
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[fill_value] * image.shape[2]
    )
    
    return padded, scale, (left, top)


def tile_image(image, tile_size, overlap=0.25):
    """이미지를 타일로 분할 (작은 객체 탐지용)"""
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    positions = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 타일 좌표
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # 타일 추출
            tile = image[y1:y2, x1:x2]
            
            # 타일 크기가 충분한 경우만 추가
            if tile.shape[0] >= tile_size * 0.5 and tile.shape[1] >= tile_size * 0.5:
                # 타일 크기로 패딩
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    tile = cv2.copyMakeBorder(
                        tile, 0, tile_size - tile.shape[0],
                        0, tile_size - tile.shape[1],
                        cv2.BORDER_CONSTANT, value=[114, 114, 114]
                    )
                
                tiles.append(tile)
                positions.append((x1, y1, x2, y2))
    
    return tiles, positions


def merge_tile_predictions(tile_predictions, positions, original_size, conf_threshold=0.25):
    """타일 예측 결과를 원본 이미지 크기로 병합"""
    h, w = original_size
    all_boxes = []
    
    for (pred, conf, cls), (x1, y1, x2, y2) in zip(tile_predictions, positions):
        if conf < conf_threshold:
            continue
        
        # 타일 좌표를 원본 이미지 좌표로 변환
        tile_h, tile_w = pred.shape[:2] if hasattr(pred, 'shape') else (1280, 1280)
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        if isinstance(pred, np.ndarray) and len(pred) == 4:
            # YOLO 형식 (x_center, y_center, width, height) - 정규화됨
            x_center, y_center, box_w, box_h = pred
            
            # 타일 내 픽셀 좌표
            px = x_center * tile_w
            py = y_center * tile_h
            pw = box_w * tile_w
            ph = box_h * tile_h
            
            # 원본 이미지 좌표로 변환
            orig_x = x1 + px - pw / 2
            orig_y = y1 + py - ph / 2
            orig_w = pw
            orig_h = ph
            
            # 정규화
            orig_x_norm = orig_x / w
            orig_y_norm = orig_y / h
            orig_w_norm = orig_w / w
            orig_h_norm = orig_h / h
            
            all_boxes.append({
                'bbox': [orig_x_norm, orig_y_norm, orig_w_norm, orig_h_norm],
                'conf': conf,
                'cls': cls
            })
    
    # NMS 적용 (중복 제거)
    return apply_nms(all_boxes, iou_threshold=0.45)


def apply_nms(boxes, iou_threshold=0.45):
    """Non-Maximum Suppression 적용"""
    if not boxes:
        return []
    
    # Confidence로 정렬
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        # IoU가 높은 박스 제거
        boxes = [box for box in boxes 
                if calculate_iou(current['bbox'], box['bbox']) < iou_threshold]
    
    return keep


def calculate_iou(box1, box2):
    """IoU 계산 (정규화된 좌표)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 박스 좌표 변환
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    # 교집합
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 합집합
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

