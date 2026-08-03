# CVDLens Web App

색각이상 보정 웹 애플리케이션입니다. 전체 연구 구조와 Phase 2 선택적 Daltonization 설명은 루트 [README](../README.md)를 참고하세요.

## 서비스 주소

개발 및 시연용 웹앱:

- [CVDLens Web App](https://d3kvjz20-3000.jpe1.devtunnels.ms/)

배포 환경:

- [Frontend](https://cvd-lens.vercel.app)
- [Backend](https://cvd-lens.onrender.com)
- [Health Check](https://cvd-lens.onrender.com/health)

> dev tunnel 주소는 실행 환경에 따라 바뀔 수 있습니다.

## 실행 방법

```bash
npm install
npm run dev
```

FastAPI 추론 서버:

```bash
cd inference
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

필요한 환경 변수:

```text
DATABASE_URL=
NEXTAUTH_SECRET=
NEXTAUTH_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 추론 동작 (`/infer`, `/infer/video`)

- **원본 해상도 반환.** 모델은 256×256에서 보정하지만, 서버는 보정
  델타(`out − in`)만 뽑아 원본 해상도로 bilinear 업샘플한 뒤 원본
  픽셀에 가산합니다. 색보정은 저주파라 업샘플해도 손실이 없고,
  글자·경계 같은 고주파 디테일은 원본 그대로 유지됩니다. (이전에는
  256 결과를 그대로 반환해 프론트가 확대 표시 → 흐릿/색번짐.)
- **letterbox 전처리.** center-crop 대신 aspect를 보존하는 letterbox로
  256에 맞춥니다. 화각이 잘리지 않습니다.
- **장변 2048 한도.** 원본 해상도로 반환하므로 응답이 커질 수 있어,
  입력 장변이 2048를 넘으면 2048로 다운스케일 후 처리합니다
  (`MAX_SIDE`, `main.py`). 비디오는 인코더가 캡처 해상도에 맞춰져
  있어 프레임 원해상도를 그대로 유지합니다.
- **검증:** `cd inference && py validate_infer.py` — (a) 잔글씨 선명도
  유지(고주파 비율 > 0.95), (b) 1584 보정 방향 유지(기존 256 경로와
  cosine > 0.90), (c) do-nothing(1761) 원해상도 |Δ| < 0.005.
