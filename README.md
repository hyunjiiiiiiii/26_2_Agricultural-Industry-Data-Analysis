# 농산업 데이터 분석 대회: 방울토마토 영향 결핍 객체 탐지

이 프로젝트는 YOLOv11 모델을 사용하여 방울토마토 영향 결핍 객체 탐지 시스템입니다.

## 프로젝트 구조

```
.
├── data/                    # 데이터 디렉토리
│   ├── raw/                 # 원본 데이터
│   │   ├── pretrain/        # AI Hub 토마토 데이터 (pretrain용)
│   │   └── finetune/        # 방울토마토 소량 데이터 (fine-tune용)
│   └── processed/           # 처리된 데이터
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/                 # 실행 스크립트
│   ├── 01_prepare_data.py   # 데이터 준비
│   ├── 02_train_pretrain.py # Pretrain 학습
│   ├── 03_train_finetune.py # Fine-tune 학습
│   ├── 04_validate.py       # 검증 및 에러 분석
│   └── 05_inference.py      # 추론
├── utils/                   # 유틸리티 함수
│   └── data_utils.py        # 데이터 처리 유틸리티
├── notebooks/               # Jupyter 노트북
├── weights/                 # 모델 가중치
├── results/                 # 결과 파일
│   ├── logs/
│   ├── plots/
│   └── confusion/
├── config.yaml             # 프로젝트 설정
└── requirements.txt        # 패키지 의존성
```

## 설치 및 환경 설정

### 1. 환경 설정

```bash
# 가상환경 생성 (Mac)
python -m venv venv
source venv/bin/activate

# 패키지 설치 (YOLOv11 포함)
pip install -r requirements.txt
# 또는 직접 설치
pip install ultralytics>=8.0.0
```

**YOLOv11 설치**: `pip install ultralytics` 명령어로 설치하면 YOLOv11을 포함한 모든 YOLO 모델을 사용할 수 있습니다. Ultralytics 패키지가 YOLOv8, YOLOv9, YOLOv10, YOLOv11 등 여러 버전을 지원합니다.

### 2. Windows GPU 환경 (Jupyter Lab Server)

Windows GPU 서버에서 Jupyter Lab을 실행하고, Mac에서 접속하여 학습을 진행합니다.

```bash
# Windows 서버에서
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Mac에서 접속
# 브라우저에서 http://[서버IP]:8888 접속
```

## 사용 방법

### 1. 데이터 준비

#### 데이터 구조
- **Pretrain 데이터 (AI Hub 토마토)**: `data/raw/pretrain/images/`, `data/raw/pretrain/labels/`
- **Fine-tune 데이터 (방울토마토)**: `data/raw/finetune/images/`, `data/raw/finetune/labels/`

#### 라벨 형식 (YOLO)
```
class_id x_center y_center width height
```
모든 좌표는 정규화된 값 (0~1)입니다.

#### 데이터 준비 실행
```bash
python scripts/01_prepare_data.py
```

이 스크립트는:
- 라벨 파일 검증
- 데이터셋 분할 (train/val/test)
- 처리된 데이터로 복사

### 2. Pretrain 학습 (AI Hub 토마토 데이터)

```bash
python scripts/02_train_pretrain.py
```

또는 Jupyter Notebook에서:
```python
from scripts.train_pretrain import train_pretrain
train_pretrain()
```

학습된 가중치는 `weights/pretrain/best.pt`에 저장됩니다.

### 3. Fine-tune 학습 (방울토마토 소량 데이터)

```bash
python scripts/03_train_finetune.py
```

Fine-tune은:
- Pretrain 가중치를 로드
- 고해상도 (1280px)로 학습
- 작은 병반 대응을 위한 증강 적용

학습된 가중치는 `weights/finetune/best.pt`에 저장됩니다.

### 4. 검증 및 에러 분석

```bash
python scripts/04_validate.py --weights weights/finetune/best.pt
```

이 스크립트는:
- Class별 mAP 계산
- Confusion Matrix 생성
- 미탐지 분석

결과는 `results/` 디렉토리에 저장됩니다.

### 5. 추론

#### 단일 이미지
```bash
python scripts/05_inference.py --weights weights/finetune/best.pt --source path/to/image.jpg
```

#### 영상
```bash
python scripts/05_inference.py --weights weights/finetune/best.pt --source path/to/video.mp4
```

#### 배치 추론
```bash
python scripts/05_inference.py --weights weights/finetune/best.pt --source path/to/images/
```

## 설정 파일 (config.yaml)

주요 설정 항목:

- **모델 설정**: 모델 크기, 학습 epoch, 이미지 크기, 배치 크기
- **데이터 증강**: 작은 객체 대응을 위한 증강 설정
- **학습 하이퍼파라미터**: 학습률, 모멘텀, 가중치 감쇠 등
- **검증/추론 설정**: Confidence threshold, IoU threshold 등

자세한 내용은 `config.yaml` 파일을 참조하세요.

## 작은 병반 대응 전략

1. **고해상도 학습**: 1280px 이미지 크기 사용
2. **타일링**: 큰 이미지를 타일로 분할하여 작은 객체 탐지
3. **데이터 증강**: Mosaic, Mixup 등 강화된 증강 적용
4. **Fine-tune**: 소량 데이터로 도메인 적응

## 주의사항

- Mac에서는 GPU 학습이 불가능하므로, Windows GPU 서버에서 Jupyter Lab을 통해 학습을 진행해야 합니다.
- 데이터 경로는 절대 경로 또는 프로젝트 루트 기준 상대 경로를 사용하세요.
- 가중치 파일은 `.gitignore`에 포함되어 있으므로, 별도로 백업하세요.

## 라이선스

이 프로젝트는 연구 및 교육 목적으로 사용됩니다.

