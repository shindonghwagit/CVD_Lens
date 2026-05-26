# CVDLens Web App

색각이상 보정 웹 애플리케이션입니다. 전체 연구 구조와 Phase 2 선택적 Daltonization 설명은 루트 [README](../README.md)를 참고하세요.

## 서비스 주소

개발 및 시연용 웹앱:

```text
https://d3kvjz20-3000.jpe1.devtunnels.ms/
```

배포 환경:

```text
Frontend: https://cvd-lens.vercel.app
Backend:  https://cvd-lens.onrender.com
Health:   https://cvd-lens.onrender.com/health
```

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
