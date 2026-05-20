# CVDLens

AI 기반 색각이상 보정 웹 애플리케이션 — 졸업작품

색각이상(Color Vision Deficiency)을 가진 사용자가 카메라로 사진을 찍거나 이미지를 업로드하면, AI 모델이 색 대비를 재구성하여 보정된 이미지를 제공합니다.

---

## 주요 기능

- **카메라 촬영 보정** — 카메라로 찍은 사진을 AI가 색 보정
- **이미지 업로드 보정** — JPG/PNG 업로드 후 원본/보정 슬라이더 비교
- **Ishihara 진단** — 색각이상 자가 진단 테스트
- **진단 기록** — 이시하라 진단 이력 조회
- **3종 CVD 지원** — Protanopia(제1색맹) / Deuteranopia(제2색맹) / Tritanopia(제3색맹)
- **회원 기반 결과 저장** — 진단 및 보정 이력 관리

---

## 기술 스택

| 영역 | 스택 |
|------|------|
| 프론트엔드 | Next.js 16, Tailwind CSS v4 |
| 인증 | NextAuth.js v5 (JWT) |
| 데이터베이스 | PostgreSQL (Supabase) |
| AI 추론 서버 | FastAPI, ONNX Runtime |
| 모델 | MobileNetV2 U-Net (smp), PyTorch Lightning |
| 학습 데이터 | COCO 2017 + Daltonize 알고리즘 |

---

## 프로젝트 구조

```
graduation_project/
├── cvd-lens/               # Next.js 웹 애플리케이션
│   ├── app/                # 페이지 및 컴포넌트
│   │   ├── correction/     # 카메라/이미지 보정
│   │   ├── ishihara/       # 색각 진단
│   │   ├── history/        # 진단 기록
│   │   ├── login/
│   │   └── register/
│   ├── inference/          # FastAPI 추론 서버
│   │   ├── main.py
│   │   └── Dockerfile
│   └── lib/                # DB, 인증 설정
└── model-training/         # 모델 학습 코드
    ├── model/              # 네트워크, 손실함수, Lightning 모듈
    ├── data/               # Dataset, DataLoader
    ├── train.py
    └── export_onnx.py
```

---

## 모델

- **아키텍처**: `smp.Unet(encoder=MobileNetV2, in_channels=4, classes=3)`
- **입력**: RGB 3채널 + CVD 타입 채널 → `(B, 4, 256, 256)`
- **출력**: 보정된 RGB → `(B, 3, 256, 256)`
- **손실함수**: L1(1.0) + SSIM(0.5) + Perceptual/VGG(0.1)
- **학습**: Kaggle GPU (T4 x2), COCO 2017 30K장

---

## 실행 방법

### 사전 준비

1. Kaggle에서 학습 완료 후 ONNX 변환:
   ```bash
   python model-training/export_onnx.py
   ```

2. 생성된 `cvdlens_fp32.onnx`를 아래 경로에 배치:
   ```
   cvd-lens/inference/model/cvdlens_fp32.onnx
   ```

3. `cvd-lens/.env.local` 생성:
   ```
   DATABASE_URL=your_supabase_connection_string
   NEXTAUTH_SECRET=your_secret
   NEXTAUTH_URL=http://localhost:3000
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

### 추론 서버 실행

```bash
cd cvd-lens/inference
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 웹 앱 실행

```bash
cd cvd-lens
npm install
npm run dev
```

| 서비스 | 주소 |
|--------|------|
| 웹 앱 | http://localhost:3000 |
| 추론 서버 | http://localhost:8000 |

---

## 개발 환경

- Python 3.11
- Node.js 20
