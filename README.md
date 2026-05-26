# CVDLens

AI 기반 색각이상 보정 웹 애플리케이션 및 선택적 색상 보정 연구 프로젝트입니다.

- 색각이상(Color Vision Deficiency, CVD)을 가진 사용자가 이미지, 카메라 사진, 영상을 입력하면 색상 혼동이 발생할 수 있는 영역을 중심으로 보정 결과를 제공합니다.
- 초기 모델은 Daltonization 결과를 직접 학습하는 방식으로 구현했고, 현재는 이를 Phase 1 베이스라인으로 두고 Phase 2에서 선택적 Daltonization 구조로 확장하고 있습니다.
- 핵심 기능: `이미지 보정`, `카메라 보정`, `영상 보정`, `이시하라 검사`, `보정 기록 저장`, `선택적 색상 보정 연구`

## 개발 기간

- 2026.03 ~ 진행 중

## 시작 가이드

### 배포 및 시연 주소

- [CVDLens Web App](https://d3kvjz20-3000.jpe1.devtunnels.ms/)
- [Frontend 배포 주소](https://cvd-lens.vercel.app)
- [Backend API 서버](https://cvd-lens.onrender.com)
- [Health Check](https://cvd-lens.onrender.com/health)

> dev tunnel 주소는 실행 환경에 따라 바뀔 수 있습니다.

### 프로젝트 실행 방법

```bash
cd cvd-lens
npm install
npm run dev
```

### FastAPI 추론 서버 실행

```bash
cd cvd-lens/inference
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 환경 변수

```text
DATABASE_URL=
NEXTAUTH_SECRET=
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 프로젝트 목적

- **색각이상 사용자 보조**
  - 적색맹, 녹색맹, 청색맹 사용자가 일상 이미지에서 색상 차이를 더 쉽게 구분할 수 있도록 보정 결과를 제공합니다.

- **전역 색상 변환의 한계 개선**
  - 기존 Daltonization은 이미지 전체에 색상 변환을 적용하기 때문에 원본 색감이 과도하게 훼손될 수 있습니다.
  - 본 프로젝트는 색상 혼동 가능성이 높은 영역을 중심으로 보정하는 선택적 보정 구조를 목표로 합니다.

- **AI 기반 선택적 보정 구조 제안**
  - Phase 1에서는 CVDLens 모델이 Daltonization 결과를 근사하도록 학습했습니다.
  - Phase 2에서는 Attention U-Net이 색상 혼동 영역 마스크를 예측하고, Daltonize 결과를 해당 영역에만 선택적으로 적용합니다.

## 주요 기능

### `이미지 보정`

- JPG/PNG 이미지를 업로드하여 색각 유형별 보정 결과를 생성합니다.
- 원본과 보정 결과를 슬라이더로 비교할 수 있습니다.
- 로그인한 사용자는 보정 결과를 기록으로 저장할 수 있습니다.

### `카메라 보정`

- 웹 카메라로 사진을 촬영한 뒤 색각 유형에 맞게 보정합니다.
- 촬영 원본과 보정 결과를 비교할 수 있습니다.
- 카메라 보정 결과도 목록에 저장할 수 있습니다.

### `영상 보정`

- MP4/MOV 영상을 업로드하면 프레임 단위로 색상 보정을 수행합니다.
- 보정된 영상은 H.264 MP4 형식으로 저장 및 다운로드할 수 있습니다.

### `이시하라 검사`

- 실제 이시하라 도판 이미지를 기반으로 간단한 색각 검사를 제공합니다.
- 매번 일부 도판을 랜덤으로 선택하여 검사를 진행합니다.
- 검사 결과는 참고용이며, 정확한 진단은 안과 전문의 상담이 필요합니다.

### `보정 기록`

- 이미지/카메라 보정 결과를 DB에 저장합니다.
- 저장된 보정 결과를 목록에서 다시 확인할 수 있습니다.

## 연구 구조

### Phase 1. CVDLens Baseline

기존 CVDLens 모델은 원본 이미지와 CVD 타입을 입력받아 Daltonization 결과 이미지를 직접 생성하도록 학습했습니다.

```text
Input  = Original RGB + CVD type
Target = Daltonize(original)
Output = CVDLens(original, cvd_type)
```

이 구조는 웹 서비스에 적용하기 쉽고 ONNX 추론이 가능하지만, 연구 관점에서는 "기존 Daltonize 알고리즘을 신경망이 모방한 것"이라는 한계가 있습니다. 따라서 Phase 1 모델은 최종 기여가 아니라 비교용 베이스라인으로 사용합니다.

### Phase 2. Selective Daltonization

Phase 2는 보정 이미지를 직접 생성하는 대신, 색각 이상자가 혼동할 가능성이 높은 영역을 예측하는 마스크 모델을 학습합니다.

```text
Original
  ├─ CVD Simulation → Difference Map → Pseudo Mask
  └─ Daltonize      → Corrected Candidate

Attention U-Net → Mask M

Final Output = M * Daltonized + (1 - M) * Original
```

| 구분 | Phase 1 | Phase 2 |
| --- | --- | --- |
| AI가 학습하는 것 | Daltonize 결과 이미지 | 색상 혼동 영역 마스크 |
| Daltonize 역할 | 학습 정답 이미지 생성 | 추론 시 보정 후보 생성 |
| pseudo-label | Daltonize(original) | `abs(CVD_sim(original) - original)` |
| 최종 출력 | 모델이 RGB 직접 생성 | 원본과 Daltonize 결과를 마스크로 합성 |
| 연구 의의 | 알고리즘 모방 베이스라인 | 필요한 영역만 선택적으로 보정 |

## Phase 2 모델

### `Attention U-Net Mask Predictor`

- **입력**: 8채널 `(B, 8, 256, 256)`
  - Original RGB: 3채널
  - CVD Simulation RGB: 3채널
  - Difference Map: 1채널
  - CVD Type Channel: 1채널
- **출력**: 1채널 soft mask `(B, 1, 256, 256)`
- **모델**: MobileNetV2 encoder 기반 U-Net + scSE decoder attention
- **Loss**: BCE Loss + Dice Loss
- **pseudo-label 생성**: CVD simulation 결과와 원본 이미지의 차이를 이용해 혼동 영역 마스크 생성

## Kaggle 학습 방법

Phase 2는 Kaggle에서 별도 학습 프로젝트로 실행합니다.

1. Kaggle에서 새 Notebook 프로젝트를 생성합니다.
2. `model-training.zip`을 Kaggle Dataset으로 업로드한 뒤 Notebook에 추가합니다.
3. COCO 2017 데이터셋을 Notebook에 추가합니다.
4. GPU를 켠 뒤 `CVDLens_Mask_Train_Kaggle.ipynb`를 실행합니다.
5. 학습이 끝나면 마스크 모델을 ONNX로 변환합니다.

직접 실행할 경우:

```bash
python train_mask.py \
  --coco-dir /path/to/coco/images \
  --batch-size 64 \
  --max-epochs 100 \
  --num-train 30000 \
  --num-val 3000 \
  --num-test 2000 \
  --lr 1e-3
```

ONNX 변환:

```bash
python export_mask_onnx.py --ckpt outputs/checkpoints_mask/best.ckpt
```

생성 파일:

```text
outputs/onnx_mask/cvdlens_mask_fp32.onnx
```

## 평가 지표

### Phase 1 평가

Phase 1에서는 모델이 Daltonization 결과를 얼마나 잘 근사하는지 확인하기 위해 PSNR/SSIM을 사용했습니다.

- COCO val 3000장
- 3개 CVD 타입
- 총 9000개 샘플
- 평균 PSNR: 42.34 dB
- 평균 SSIM: 0.9933

### Phase 2 평가 계획

- **Mask IoU**
  - pseudo confusion mask와 예측 mask의 일치도를 평가합니다.
- **CIEDE2000**
  - 보정 전후 색상 구별 가능성 변화를 평가합니다.
- **Color Discriminability**
  - CVD simulation 이후 색상 대비 개선 정도를 평가합니다.
- **시각화 비교**
  - 원본, Daltonize 전체 적용, 선택적 Daltonize 결과를 비교합니다.

## 디렉토리 구조

```text
graduation_project/
├── cvd-lens/
│   ├── app/
│   │   ├── correction/        # 카메라 / 이미지 / 영상 보정
│   │   ├── corrections/       # 보정 기록
│   │   ├── history/           # 진단 기록
│   │   ├── ishihara/          # 이시하라 검사
│   │   ├── components/
│   │   └── api/
│   ├── inference/
│   │   ├── main.py            # FastAPI 추론 서버
│   │   └── model/             # ONNX 모델 파일
│   └── public/
│       └── ishihara/          # 실제 이시하라 도판 이미지
└── model-training/
    ├── data/
    │   ├── dataset.py         # Phase 1 학습 데이터셋
    │   └── mask_dataset.py    # Phase 2 마스크 데이터셋
    ├── model/
    │   ├── network.py         # Phase 1 CVDLens 모델
    │   ├── lit_module.py      # Phase 1 Lightning 모듈
    │   ├── losses.py          # Phase 1 loss
    │   ├── mask_network.py    # Phase 2 Attention U-Net
    │   ├── mask_losses.py     # BCE + Dice loss
    │   └── mask_lit_module.py # Phase 2 Lightning 모듈
    ├── train.py
    ├── train_mask.py
    ├── export_onnx.py
    └── export_mask_onnx.py
```

## 구현 과정

### 1. Daltonization 기반 Phase 1 모델 구현

- COCO 이미지에 Daltonization 알고리즘을 적용하여 학습 쌍을 생성했습니다.
- MobileNetV2 U-Net이 원본 이미지와 CVD 타입을 입력받아 보정 이미지를 생성하도록 학습했습니다.
- ONNX로 변환하여 FastAPI 서버에서 추론할 수 있도록 구성했습니다.

### 2. Phase 1 한계 분석

- Phase 1 모델은 높은 PSNR/SSIM을 보였지만, 이는 Daltonization 결과를 잘 근사했다는 의미입니다.
- 즉, 기존 알고리즘을 모방했다는 한계가 있어 최종 연구 기여로는 부족하다고 판단했습니다.

### 3. Phase 2 선택적 보정 구조 설계

- Daltonize는 보정 후보 생성기로 사용합니다.
- Attention U-Net은 CVD simulation 기반 혼동 영역 마스크를 예측합니다.
- 최종 결과는 원본과 Daltonize 결과를 마스크 기반으로 합성합니다.

### 4. 웹앱 기능 확장

- 이미지 보정, 카메라 보정, 영상 보정 기능을 추가했습니다.
- 보정 결과를 저장하고 다시 확인할 수 있는 기록 기능을 구현했습니다.
- 실제 이시하라 도판 이미지를 사용하도록 검사 기능을 개선했습니다.

## 트러블슈팅

### Kaggle 학습 타임아웃

- **문제**
  - Kaggle Notebook에서 긴 학습 중 `CellTimeoutError`가 발생했습니다.
- **해결**
  - PyTorch Lightning checkpoint를 저장하고, 다음 세션에서 `last.ckpt`를 업로드하여 이어서 학습했습니다.

### Daltonization 모방 한계

- **문제**
  - Phase 1은 Daltonization 결과를 직접 학습했기 때문에 "기존 알고리즘을 신경망이 따라 한 것"이라는 한계가 있었습니다.
- **해결**
  - Phase 2에서 Daltonize는 보정 후보 생성기로만 사용하고, Attention U-Net이 혼동 영역 마스크를 예측하도록 역할을 분리했습니다.

### 영상 보정 재생 문제

- **문제**
  - OpenCV 기본 MP4 코덱으로 저장한 영상이 브라우저에서 바로 재생되지 않는 문제가 있었습니다.
- **해결**
  - ffmpeg를 사용해 H.264 MP4 형식으로 변환하도록 수정했습니다.

## 기술 스택

### Frontend

<img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=white"> <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white">

### Backend

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white"> <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">

### AI / Training

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white"> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">

### Deploy / Tools

<img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white"> <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"> <img src="https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white">

## 커밋 컨벤션

| 커밋 유형 | 설명 |
| --- | --- |
| init | 프로젝트 시작 |
| feat | 기능 추가 |
| fix | 버그 수정 |
| docs | 문서 수정 |
| style | 코드 포맷팅, UI 스타일 수정 |
| refactor | 코드 리팩토링 |
| chore | 설정, 패키지, 빌드 관련 수정 |
| remove | 파일 또는 코드 삭제 |
| rename | 파일 또는 폴더명 수정 |