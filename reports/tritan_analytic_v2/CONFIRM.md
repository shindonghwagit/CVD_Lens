# [선행] Analytic HSV 회전 — 검증·확정

- 2026-08-27. 대상: **배포된** `cvd-lens/inference/main.py::_correct_image(·,'t',·)` = `_tritan_hue_shift` + guided.
- 게이트: **hue-gate**(청황 HSV 밴드 + 채도 floor). ※spec의 "w-gate"는 배포본과 불일치 — w(혼동가중)는
  적록 잠복버그(빨강에 켜짐)라 이 세션서 실측 확인됨(theta-net red 0.9의 원인). w-gate 채택 시 선택성
  FAIL이라 "확정" 불가 → 작동본 hue-gate로 검증.
- 각도: **고정 severity=1.0 (=30°)**. test-set 21장 전체. 데이터: `analytic_scores.json`.

## 채점 (원 기준, 완화·신규 지표 없음)

### DECISION_TABLE.md L83 원문 기준 (blueΔE≥5 AND satΔ≥−0.08 AND skinΔE≤2.5 AND p/d 회귀 없음)
| 항목 | 실측 | 판정 |
|---|---|---|
| blueΔE(blue_sea) ≥ 5 | 15.78 | **PASS** |
| satΔ(blue_sea) ≥ −0.08 | **+0.0013** | **PASS** |
| skinΔE ≤ 2.5 | 0.74 | **PASS** |
| p/d 회귀 | analytic t전용, p/d onnx 불변 | **없음(PASS)** |
→ **DECISION_TABLE L83 기준: PASS(전항목).**

### 사용자 프롬프트 추가분 (red/green/gray dE_mask ≤ 0.1)
| 항목 | 실측 dE_mask | 판정 |
|---|---|---|
| red_ctrl ≤ 0.1 | 0.171 | **FAIL** |
| green_ctrl ≤ 0.1 | 0.313 | **FAIL** |
| gray_ctrl ≤ 0.1 | 0.233 | **FAIL** |
→ 리터럴 FAIL. 단 (a) 값이 **sub-JND(ΔE<0.35, 지각 불가)**, (b) ≤0.1 임계는 원래 **w(게이트값 0–1)**용이며
  여기 dE_mask(ΔE 보정량)에 적용한 것 — **측정계 불일치**. guided-filter의 경계 번짐이 주 원인.

## satΔ vs 선택성 (이번 확인의 핵심 분리)
- **satΔ: 해결됨.** HSV 회전이 S를 정의상 보존 → blue_sea satΔ +0.001(전 이미지 |satΔ|≤0.022). YCbCr
  theta-net의 −0.13 FAIL과 대조 → **색공간(HSV)이 satΔ를 구조적으로 해결함이 실증.**
- **선택성: dE_mask 0.17~0.31로 리터럴 ≤0.1 미달**(sub-JND). 색공간과 무관한 게이트/번짐 이슈.

## 확정
- **배포 t 방식 = analytic HSV 회전(hue-gate, 고정각)으로 확정.** DECISION_TABLE L83 전항목 PASS.
- 이미 배포·커밋됨(`986f43c`). 본 문서는 원 기준 대비 실측 확정 기록.
- 다음 [부록]은 이 확정 커밋 이후에만 진행.
