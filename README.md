# CVDLens

AI 기반 색각이상 보정 웹 애플리케이션 — 졸업작품

색각이상(Color Vision Deficiency)을 가진 사용자가 카메라로 사진을 찍거나 이미지를 업로드하면, AI 모델이 색 대비를 재구성하여 보정된 이미지를 제공합니다.

---

## 주요 기능

- **카메라 촬영 보정** — 카메라로 찍은 사진을 AI가 실시간 색 보정
- **이미지 업로드 보정** — JPG/PNG 업로드 후 원본/보정 슬라이더 비교
- **Ishihara 진단** — 색각이상 자가 진단 테스트
- **3종 CVD 지원** — Protanopia(제1색맹) / Deuteranopia(제2색맹) / Tritanopia(제3색맹)
- **회원 기반 결과 저장** — 진단 이력 관리

---

## 기술 스택

| 영역 | 스택 |
|------|------|
| 프론트엔드 | Next.js 16, Tailwind CSS v4 |
| 인증 | NextAuth.js v5 (JWT) |
| 데이터베이스 | PostgreSQL |
| AI 추론 서버 | FastAPI, ONNX Runtime |
| 모델 | MobileNetV2 U-Net (smp), PyTorch Lightning |
| 학습 데이터 | COCO 2017 + Daltonize 알고리즘 |
| 배포 | Docker Compose |

---

## 프로젝트 구조

```
graduation_project/
├── cvd-lens/               # Next.js 웹 애플리케이션
│   ├── app/                # 페이지 및 컴포넌트
│   ├── inference/          # FastAPI 추론 서버
│   │   ├── main.py
│   │   └── Dockerfile
│   ├── docker/
│   │   └── init.sql        # DB 초기화 스크립트
│   ├── docker-compose.yml
│   └── Dockerfile
└── model-training/         # 모델 학습 코드
    ├── model/              # 네트워크, 손실함수, Lightning 모듈
    ├── data/               # Dataset, DataLoader
    ├── train.py
    ├── export_onnx.py
    └── CVDLens_Train_Kaggle.ipynb
```

---

## 모델

- **아키텍처**: `smp.Unet(encoder=MobileNetV2, in_channels=4, classes=3)`
- **입력**: RGB 3채널 + CVD 타입 채널 → `(B, 4, 256, 256)`
- **출력**: 보정된 RGB → `(B, 3, 256, 256)`
- **손실함수**: L1(1.0) + SSIM(0.5) + Perceptual/VGG(0.1)
- **학습**: Kaggle GPU (T4 / P100), COCO 2017 10K장

---

## 실행 방법

### 사전 준비

Kaggle에서 학습 완료 후 `cvdlens_fp32.onnx`를 다운로드하여 아래 경로에 배치:

```
model-training/outputs/onnx/cvdlens_fp32.onnx
```

### Docker로 실행

```bash
cd cvd-lens
docker compose up --build
```

| 서비스 | 주소 |
|--------|------|
| 웹 앱 | http://localhost:3000 |
| 추론 서버 | http://localhost:8000 |
| DB | localhost:5432 |

### 개발 모드

```bash
# 추론 서버 (별도 터미널)
cd cvd-lens/inference
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Next.js
cd cvd-lens
npm install
npm run dev
```

---

## 모델 학습

Kaggle 노트북(`model-training/CVDLens_Train_Kaggle.ipynb`) 사용:

1. Kaggle → New Notebook → Import → 노트북 업로드
2. Input 데이터셋 추가:
   - `awsaf49/coco-2017-dataset`
   - `cvdlens-code` (model-training/ 폴더 zip 업로드)
3. Accelerator → GPU T4 x2 또는 P100 선택
4. 셀 순서대로 실행

---

## 개발 환경

- Python 3.11
- Node.js 20
- Docker Desktop
