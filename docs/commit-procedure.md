# 커밋 · 배포 절차 메모

리포는 한 곳이지만 배포 타깃이 둘(Vercel = 프론트, Render = 추론 서버)이고,
학습/실험 산출물은 배포와 무관하다. push 전에 이 문서로 "이번 변경이 무엇을
재배포시키는가"를 먼저 판정한다.

## 1. 배포 영향 매트릭스

| 변경 경로 | 재배포 대상 | 비고 |
|---|---|---|
| `cvd-lens/**` (단, `inference/**` 제외) | **Vercel 재배포** | Next.js 프론트. `tsc` + `next build` 통과 필수 |
| `cvd-lens/inference/**` | **Render 재배포** | Python 추론 서버(FastAPI). Docker 빌드 |
| `outputs/**` | **배포 무관** | 실험 산출물·리포트 |
| `cvdlens_v2/**` | **배포 무관** | 학습/평가 코드 (배포 경로 아님) |
| `docs/**`, `paper/**`, 루트 `*.md` | **배포 무관** | 문서 |

주의: `cvd-lens/inference/**`는 `cvd-lens/**`의 하위이므로, inference를 건드리면
Vercel 빌드도 트리거될 수 있다(설정에 따라). inference만 바꾼 커밋은 Render만
확인하면 되지만, 같은 커밋에 프론트 변경이 섞이면 둘 다 확인.

## 2. push 전 체크

- [ ] **프론트 변경 시** `cd cvd-lens && npx tsc --noEmit` 통과
- [ ] **프론트 변경 시** `cd cvd-lens && npm run build` (next build) 통과
- [ ] `git status` clean (의도한 파일만 스테이징)
- [ ] 민감정보 없음 (`.env`, 토큰, DB 접속 문자열, 모델 서명 키 등 미포함)

## 3. push 후 확인

- [ ] Vercel 대시보드: 해당 커밋 빌드 **성공** (프론트 변경 시)
- [ ] Render 대시보드: 해당 커밋 빌드 **성공** (inference 변경 시)
- [ ] 실서비스 1분 스모크: **업로드 → 보정 1회** 정상 동작 (배포가 걸린 경우)

배포 무관 커밋(outputs/docs/paper/cvdlens_v2만)은 3단계 스킵 가능.
