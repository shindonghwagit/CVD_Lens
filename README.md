# CVDLens

AI 기반 색각이상 보정 웹 애플리케이션 및 선택적 색상 보정 연구 프로젝트입니다.

CVDLens는 색각이상(Color Vision Deficiency, CVD)을 가진 사용자가 이미지, 카메라 사진, 영상을 입력하면 색상 혼동이 발생할 수 있는 영역을 중심으로 보정 결과를 제공합니다. 초기 모델은 Daltonization 결과를 직접 학습하는 방식이었고, 현재는 이를 Phase 1 베이스라인으로 두고 Phase 2에서 선택적 Daltonization 구조로 확장하고 있습니다.

---

## 서비스 주소

개발 및 시연용 웹앱은 아래 주소에서 확인할 수 있습니다.

```text
Web App: https://d3kvjz20-3000.jpe1.devtunnels.ms/
```

배포 환경을 사용하는 경우:

```text
Frontend: https://cvd-lens.vercel.app
Backend:  https://cvd-lens.onrender.com
Health:   https://cvd-lens.onrender.com/health
```

> dev tunnel 주소는 실행 환경에 따라 바뀔 수 있습니다.

---

## 주요 기능

- **카메라 보정**: 웹 카메라로 촬영한 이미지를 보정하고 원본/보정 결과를 슬라이더로 비교
- **이미지 보정**: JPG/PNG 이미지를 업로드하여 색각 유형별 보정 결과 생성
- **영상 보정**: MP4/MOV 영상을 프레임 단위로 보정한 뒤 H.264 MP4로 저장
- **이시하라 검사**: 실제 이시하라 도판 이미지 기반의 간단한 색각 검사
- **보정 기록 저장**: 로그인 사용자의 이미지/카메라 보정 결과 저장 및 목록 조회
- **3종 색각 유형 지원**: 적색맹, 녹색맹, 청색맹

---

## 연구 구조

### Phase 1: CVDLens Baseline

기존 CVDLens 모델은 원본 이미지와 CVD 타입을 입력받아 Daltonization 결과 이미지를 직접 생성하도록 학습했습니다.

```text
Input  = Original RGB + CVD type
Target = Daltonize(original)
Output = CVDLens(original, cvd_type)
```

이 구조는 웹 서비스에 적용하기 쉽고 ONNX 추론이 가능하지만, 연구 관점에서는 "기존 Daltonize 알고리즘을 신경망이 모방한 것"이라는 한계가 있습니다. 따라서 Phase 1 모델은 최종 기여가 아니라 비교용 베이스라인으로 사용합니다.

### Phase 2: Selective Daltonization

현재 확장 중인 Phase 2는 보정 이미지를 직접 생성하는 대신, 색각 이상자가 혼동할 가능성이 높은 영역을 예측하는 마스크 모델을 학습합니다.

```text
Original
  ├─ CVD Simulation → Difference Map → pseudo mask
  └─ Daltonize      → corrected candidate

Attention U-Net → Mask M

Final Output = M * Daltonized + (1 - M) * Original
```

핵심 차이는 다음과 같습니다.

| 구분 | Phase 1 | Phase 2 |
|---|---|---|
| AI가 학습하는 것 | Daltonize 결과 이미지 | 색상 혼동 영역 마스크 |
| Daltonize 역할 | 학습 정답 이미지 생성 | 추론 시 보정 후보 생성 |
| pseudo-label | Daltonize(original) | `abs(CVD_sim(original) - original)` |
| 최종 출력 | 모델이 RGB 직접 생성 | 원본과 Daltonize 결과를 마스크로 합성 |
| 연구 의의 | 알고리즘 모방 베이스라인 | 필요한 영역만 선택적으로 보정 |

---

## Phase 2 모델

Phase 2는 Attention U-Net 기반 마스크 예측 모델입니다.

- **입력**: 8채널 `(B, 8, 256, 256)`
  - Original RGB: 3채널
  - CVD Simulation RGB: 3채널
  - Difference Map: 1채널
  - CVD Type Channel: 1채널
- **출력**: 1채널 soft mask `(B, 1, 256, 256)`
- **모델**: MobileNetV2 encoder 기반 U-Net + scSE decoder attention
- **Loss**: BCE Loss + Dice Loss
- **pseudo-label 생성**: CVD simulation 결과와 원본 이미지의 차이를 이용해 혼동 영역 마스크 생성

관련 파일:

```text
model-training/
├── data/
│   ├── dataset.py             # Phase 1 학습 데이터셋
│   └── mask_dataset.py        # Phase 2 마스크 데이터셋
├── model/
│   ├── network.py             # Phase 1 CVDLens 모델
│   ├── lit_module.py          # Phase 1 Lightning 모듈
│   ├── losses.py              # Phase 1 loss
│   ├── mask_network.py        # Phase 2 Attention U-Net
│   ├── mask_losses.py         # BCE + Dice loss
│   └── mask_lit_module.py     # Phase 2 Lightning 모듈
├── train.py                   # Phase 1 학습
├── export_onnx.py             # Phase 1 ONNX 변환
├── train_mask.py              # Phase 2 학습
├── export_mask_onnx.py        # Phase 2 ONNX 변환
└── CVDLens_Mask_Train_Kaggle.ipynb
```

---

## 기술 스택

| 영역 | 스택 |
|---|---|
| 프론트엔드 | Next.js, React, Tailwind CSS |
| 인증 | NextAuth.js |
| 데이터베이스 | PostgreSQL, Supabase |
| 추론 서버 | FastAPI, ONNX Runtime |
| 모델 학습 | PyTorch, PyTorch Lightning, segmentation-models-pytorch |
| 영상 처리 | OpenCV, ffmpeg |
| 배포 | Vercel, Render |

---

## 프로젝트 구조

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
    ├── model/
    ├── train.py
    ├── train_mask.py
    ├── export_onnx.py
    └── export_mask_onnx.py
```

---

## Kaggle 학습 흐름

Phase 2만 새로 학습하면 됩니다. Phase 1 CVDLens는 비교용 베이스라인으로 유지합니다.

1. `model-training.zip`을 Kaggle Dataset으로 업로드합니다.
2. COCO 이미지 데이터셋을 Kaggle Notebook에 추가합니다.
3. GPU를 켠 뒤 `CVDLens_Mask_Train_Kaggle.ipynb`를 실행합니다.
4. 학습이 끝나면 마스크 모델을 ONNX로 변환합니다.

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

이 모델은 최종 보정 이미지를 직접 생성하지 않고, Daltonize 결과를 어디에 얼마나 적용할지 결정하는 마스크를 출력합니다.

---

## 평가 지표

Phase 1에서는 모델이 Daltonization 결과를 얼마나 잘 근사하는지 확인하기 위해 PSNR/SSIM을 사용했습니다.

- COCO val 3000장, 3개 CVD 타입, 총 9000개 샘플
- 평균 PSNR: 42.34 dB
- 평균 SSIM: 0.9933

Phase 2에서는 다음 평가를 추가로 사용할 수 있습니다.

- Mask IoU: pseudo confusion mask와 예측 mask의 일치도
- CIEDE2000: 보정 전후 색상 구별 가능성 변화
- Color Discriminability: CVD simulation 이후 색상 대비 개선 정도
- 시각화 비교: 원본 / Daltonize 전체 적용 / 선택적 Daltonize 결과

---

## 로컬 실행

### FastAPI 추론 서버

```bash
cd cvd-lens/inference
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Next.js 웹앱

```bash
cd cvd-lens
npm install
npm run dev
```

기본 주소:

```text
웹앱: http://localhost:3000
추론 서버: http://localhost:8000
헬스 체크: http://localhost:8000/health
```

필요한 환경 변수:

```text
DATABASE_URL=
NEXTAUTH_SECRET=
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 발표용 요약

본 프로젝트는 색각이상 사용자를 위한 이미지 보정 시스템을 구현한다. 초기 Phase 1에서는 CNN 기반 CVDLens가 Daltonization 결과를 직접 근사하도록 학습했다. 그러나 이 방식은 기존 알고리즘 모방이라는 한계가 있어, Phase 2에서는 Attention U-Net이 CVD simulation 기반 혼동 영역 마스크를 예측하고, Daltonize 결과를 해당 영역에만 선택적으로 적용하는 구조로 확장하였다. 이를 통해 전역 색상 변환으로 인한 원본 색감 훼손을 줄이고, 혼동 가능성이 높은 영역 중심의 보정을 목표로 한다.
