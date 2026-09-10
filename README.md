# CVDLens

AI 기반 색각이상(Color Vision Deficiency, CVD) 보정 웹 애플리케이션 및 선택적 색상 보정 연구 프로젝트입니다.

- 색각이상을 가진 사용자가 이미지, 카메라 사진, 영상을 입력하면 색상 혼동이 발생하기 쉬운 영역을 중심으로 보정한 결과를 제공합니다.
- 이미지 전체를 일괄 변환하는 기존 Daltonization과 달리, **혼동 가능성이 높은 색만 골라서 그 색이 원래 갖는 밝기·질감은 유지한 채 구분 가능하게** 만드는 것을 목표로 합니다.
- 핵심 기능: `이미지 보정`, `카메라 보정`, `영상 보정`, `이시하라 검사`, `색각 교육`, `보정 기록 저장`

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
  - 적색맹(protan), 녹색맹(deutan), 청색맹(tritan) 사용자가 일상 이미지에서 색상 차이를 더 쉽게 구분할 수 있도록 보정 결과를 제공합니다.

- **전역 색상 변환의 한계 개선**
  - 기존 Daltonization은 이미지 전체에 색상 변환을 적용하기 때문에 원본 색감이 과도하게 훼손될 수 있습니다.
  - 본 프로젝트는 색상 혼동 가능성이 높은 픽셀에만 보정을 집중하는 **선택적 보정** 구조를 사용합니다.

- **알고리즘 모방이 아닌 지각 기반 학습**
  - Daltonization 결과 이미지를 정답으로 모방하도록 학습하지 않습니다.
  - 색각이상 시뮬레이션을 통해 "이 사용자에게 두 색이 얼마나 구분되는가"를 직접 손실 함수로 정의하고, 그 구분성을 높이도록 네트워크를 학습합니다.

## 주요 기능

### `이미지 보정`

- JPG/PNG 이미지를 업로드하여 색각 유형별 보정 결과를 생성합니다.
- 원본과 보정 결과를 슬라이더로 비교할 수 있습니다.
- 로그인한 사용자는 보정 결과를 기록으로 저장할 수 있습니다.

### `카메라 보정`

- 웹 카메라로 사진을 촬영한 뒤 색각 유형에 맞게 보정합니다.
- 촬영 원본과 보정 결과를 비교하고 목록에 저장할 수 있습니다.

### `영상 보정`

- MP4/MOV 영상을 업로드하면 프레임 단위로 색상 보정을 수행하고, H.264 MP4로 저장·다운로드할 수 있습니다.
- 적색맹/녹색맹/청색맹은 브라우저에서 **실시간**으로 카메라·영상 화면을 보정해 미리 볼 수 있습니다.
- 영상 보정 결과도 목록에 저장할 수 있습니다.

### `이시하라 검사`

- 공식 이시하라 38판 도판을 기반으로 색각 검사를 제공합니다.
- 정상판/분류판 응답 분포로 적색맹·녹색맹을 감별하고, 유형별 막대그래프로 결과를 시각화합니다.
- 검사 결과는 참고용이며, 정확한 진단은 안과 전문의 상담이 필요합니다.

### `색각 교육`

- 색각이상의 원리와 유형을 설명하는 가이드 페이지를 제공합니다.
- 클라이언트 측 유형 시뮬레이터로 정상 시야와 각 색각 유형의 시야를 비교해 볼 수 있습니다.

### `보정 기록`

- 이미지/카메라/영상 보정 결과를 DB에 저장하고, 목록에서 다시 확인할 수 있습니다.

## 보정 방식

CVDLens는 보정 이미지를 통째로 생성하지 않고, **원본에 더할 색상 델타(delta)** 를 예측한 뒤 혼동 가중치로 그 세기를 조절해 원본에 합성합니다.

```text
최종 출력 = clamp( 원본 + w · Δ )

Δ  : 네트워크가 예측하는 색상 보정 델타 (가시 부분공간으로 제한)
w  : 혼동 가중치 (색각 시뮬레이션 기반, 혼동되기 쉬운 픽셀일수록 1에 가까움)
```

- **혼동 가중치 `w`** — Lab ΔE 기반으로 "이 픽셀 색이 해당 색각 유형에게 얼마나 혼동되는가"를 계산합니다. 혼동이 적은 영역은 `w ≈ 0`이 되어 원본이 그대로 유지되므로, 보정이 필요한 색에만 선택적으로 적용됩니다.
- **가시 부분공간 제한** — 델타는 색각이상자가 실제로 구분할 수 있는 축(밝기 축·채도 축) 방향으로만 합성되어, 혼동선(confusion line)에 평행한 무의미한 변화 없이 구분성만 끌어올립니다.
- **모방이 아닌 구분성 최적화** — 손실 함수가 색각 시뮬레이션을 통과한 결과의 대비를 직접 평가하므로, 기존 Daltonize 알고리즘을 흉내 내는 것이 아니라 "그 사용자에게 잘 보이는가"를 기준으로 학습됩니다.

## 모델 구조

### `CVDCorrectionNet` (bilateral-grid 색상 보정 네트워크)

- **입력**: sRGB 이미지 `(B, 3, H, W)` + one-hot CVD 타입 `(B, 3)` + severity `(B, 1)`
- **출력**: 원본에 합성할 저주파 색상 델타 필드
- **구조**
  - MobileNetV3-Small 인코더 (ImageNet 정규화 입력)
  - CVD 타입·severity를 조건으로 하는 **FiLM** 변조를 2개 인코더 스테이지에 주입
  - 1×1 conv 헤드 → **bilateral grid** `(B, 2, D=8, 16, 16)` (밝기 델타 `d_lum`, 채도 델타 `d_c`)
  - 휘도 guide로 grid를 slice(trilinear)해 원해상도 `(d_lum, d_c)` 복원 → 가시 부분공간 합성으로 델타 생성
- **Speckle 억제** — bilateral-grid 슬라이싱 자체가 구조적 평활 prior로 작동해, 균일한 영역에서 얼룩(speckle) 없이 매끄러운 델타를 만듭니다.

## 학습

색각이상 시뮬레이션 기반의 **지각 손실**로 학습합니다. Daltonize 결과를 정답 이미지로 쓰지 않습니다.

- **손실 `CVDLossV2`** (`cvdlens_v2/losses.py`)
  - `L_contrast` — 색각 시뮬레이션을 통과한 결과에서 혼동 색쌍의 대비 부족분만 벌하는 one-sided·혼동 가중 대비 손실
  - `L_global` — 전역 색상 구분성(ΔE) 손실
  - `L_naturalness` — 원본 색감 보존 (LPIPS + 시뮬-L1)
  - `TV` — 델타 필드 평활 정규화
- **학습 환경**: Kaggle GPU, COCO 2017 이미지, 유형·severity 랜덤 배치, 20,000 step
- **체크포인트 선정**: 타입별 평균 혼동가중 대비비(ratio_w)와 do-nothing 보존성 게이트를 통과하는 체크포인트 중 최적점을 선정

```bash
py -m cvdlens_v2.train        # 학습
py -m cvdlens_v2.select_best  # 게이트 기반 최적 체크포인트 선정
py -m cvdlens_v2.export_onnx  # ONNX 변환 (타입별 self-contained 그래프)
```

## 추론 파이프라인 (`/infer`, `/infer/video`)

- **타입별 ONNX 모델** — 적색맹/녹색맹(p/d)은 학습된 `cvdlens_{p,d}.onnx`로 보정합니다. 입력은 `srgb(1,3,256,256)` + `severity(1,1)`.
- **델타 합성 · 원본 해상도 반환** — 모델은 256×256에서 보정하지만 서버는 보정 델타(`out − in`)만 뽑아 원본 해상도로 bilinear 업샘플한 뒤 원본 픽셀에 더합니다. 색보정은 저주파라 손실이 없고, 글자·경계 같은 고주파 디테일은 원본 그대로 유지됩니다.
- **letterbox 전처리** — center-crop 대신 aspect를 보존하는 letterbox로 256에 맞춰 화각이 잘리지 않습니다.
- **청색맹 hue 회전** — 청색맹(t)은 학습 모델이 파랑을 탈채도시키는 문제(물빠짐)가 있어, 채도·명도는 고정한 채 파랑·노랑 hue만 회전시키는 **채도보존 hue 회전**으로 보정합니다. 빨강·초록·회색은 hue 밴드·채도 하한 밖이라 건드리지 않아 선택성이 유지됩니다.
- **guided filter 후처리** — 합성된 델타를 원본을 guide로 하는 edge-aware guided filter로 다듬어, 경계는 선명하게·균일 영역의 저주파 델타 기복은 평탄하게 만듭니다. 모델은 그대로 두고 델타에만 적용합니다.
- **장변 2048 한도** — 원해상도 반환이라 응답이 커질 수 있어 입력 장변이 2048를 넘으면 다운스케일 후 처리합니다(`MAX_SIDE`).

## 평가

색각이상 시뮬레이션을 통과했을 때의 **구분성 회복**을 기준으로 평가합니다.

- **혼동 가중 대비비 (ratio_w / CRR)** — 보정 전후로 시뮬레이션 뷰에서 혼동 색쌍의 대비가 얼마나 회복되는지를 측정합니다. 타입별 평균 게이트(P ≥ 1.10, D ≥ 1.13, T ≥ 1.27)로 통과 여부를 판정합니다.
- **보존성(do-nothing anchor)** — 혼동이 적은 이미지에서는 보정을 거의 하지 않아야 하며, `|Δ| < 0.005`로 선택성을 검증합니다.
- **색상 구분성(ΔE) / naturalness** — 보정 후 혼동 색쌍의 CIEDE2000 증가와 피부색 등 무관 영역의 색 왜곡을 함께 확인합니다.
- **추론 검증** — `cd cvd-lens/inference && py validate_infer.py`로 (a) 잔글씨 선명도 유지, (b) 보정 방향 유지, (c) do-nothing 원해상도 `|Δ|` 최소를 확인합니다.

## 디렉토리 구조

```text
graduation_project/
├── cvd-lens/                     # 웹 애플리케이션 + 추론 서버
│   ├── app/
│   │   ├── correction/           # 카메라 / 이미지 / 영상 보정
│   │   ├── corrections/          # 보정 기록
│   │   ├── history/              # 진단 기록
│   │   ├── ishihara/             # 이시하라 검사 (38판)
│   │   ├── education/            # 색각 교육 + 유형 시뮬레이터
│   │   ├── components/
│   │   └── api/
│   ├── inference/
│   │   ├── main.py               # FastAPI 추론 서버 (델타 합성 · guided filter)
│   │   ├── guided.py             # guided filter 후처리
│   │   ├── config.py             # 추론 설정 (guided filter · JPEG 품질 등)
│   │   └── model/                # cvdlens_{p,d,t}.onnx
│   └── public/
│       └── ishihara/             # 이시하라 도판 이미지
└── cvdlens_v2/                   # 모델 · 손실 · 학습 · 평가 (연구 코드)
    ├── model.py                  # CVDCorrectionNet (bilateral-grid)
    ├── losses.py                 # CVDLossV2 (지각 대비 손실)
    ├── confusion.py              # 혼동 가중치 w
    ├── basis.py                  # 가시 부분공간 델타 합성
    ├── simulation.py             # 색각이상 시뮬레이션
    ├── train.py / kaggle_train.py
    ├── select_best.py            # 게이트 기반 체크포인트 선정
    └── export_onnx.py            # ONNX 변환
```

## 구현 과정

### 1. 지각 손실 프레임워크 설계

- Daltonize 결과를 모방하는 대칭 손실(L1/SSIM)은 항등변환(원본 그대로)이 최소가 되어 보정이 붕괴하는 문제가 있었습니다.
- 색각 시뮬레이션을 통과한 결과에서 **혼동 색쌍의 대비 부족분만** 벌하는 one-sided 혼동 가중 손실로 전환해 이 붕괴를 해결했습니다.

### 2. bilateral-grid 보정 네트워크 구현

- MobileNetV3-Small + FiLM 조건부 네트워크가 저해상 bilateral grid를 예측하고, 휘도 guide로 slice해 원해상도 델타를 복원하도록 구현했습니다.
- 델타를 가시 부분공간으로 제한해 혼동선에 무의미한 변화를 넣지 않고 구분성만 높였습니다.

### 3. 청색맹 보정 분리

- 학습 모델의 청색맹 보정이 파랑을 탈채도시키는 한계(파랑 회복↔채도 유지가 본질적으로 결합)를 확인하고, 청색맹만 채도보존 hue 회전 방식으로 분리했습니다.

### 4. 추론 파이프라인 최적화

- 256 보정 결과 대신 델타만 원해상도로 합성하도록 바꿔 디테일 손실과 색번짐을 없앴습니다.
- guided filter 후처리로 경계 선명도와 델타 균일도를 개선하고, 배포 서버 메모리 한도에 맞춰 계수 업샘플 방식의 fast guided filter로 OOM을 해결했습니다.

### 5. 웹앱 기능 확장

- 이미지·카메라·영상 보정, 실시간 브라우저 보정, 보정 기록 저장, 색각 교육, 이시하라 38판 검사를 구현했습니다.

## 트러블슈팅

### 학습 붕괴 (identity collapse)

- **문제** — 대칭 손실(L1/SSIM)에서는 원본을 그대로 두는 것이 손실 최소라 보정이 사라졌습니다.
- **해결** — 시뮬레이션 뷰의 대비 부족분만 벌하는 one-sided 혼동 가중 손실로 바꿔 보정이 살아나도록 했습니다.

### 청색맹 물빠짐

- **문제** — 학습 모델이 파랑을 구분 가능하게 만들면서 동시에 탈채도시켜(물빠짐) 색감이 손상됐습니다.
- **해결** — 청색맹은 채도·명도를 고정하고 파랑·노랑 hue만 회전시키는 방식으로 분리했습니다.

### 배포 서버 메모리(OOM)

- **문제** — 원해상도 color guided filter가 512MB 무료 서버에서 OOM으로 크래시했습니다.
- **해결** — guided filter 계수를 저해상도에서 구해 업샘플한 뒤 풀해상도 guide에 적용하는 fast 방식으로 메모리를 크게 줄였습니다.

### 영상 보정 재생 문제

- **문제** — OpenCV 기본 코덱 MP4가 브라우저에서 바로 재생되지 않고, 배포 서버에 ffmpeg가 없어 검은 화면으로 실패했습니다.
- **해결** — ffmpeg로 H.264 MP4 변환하도록 하고, ffmpeg를 PATH에서만 찾도록 수정했습니다.

## 기술 스택

### Frontend

<img src="https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"> <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=white"> <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white">

### Backend

<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"> <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white"> <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">

### AI / Training

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white"> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">

### Deploy / Tools

<img src="https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white"> <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"> <img src="https://img.shields.io/badge/Supabase-3FCF8E?style=for-the-badge&logo=supabase&logoColor=white">
