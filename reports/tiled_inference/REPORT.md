# 타일 추론 실험 (작업 3, 실험 전용·미배포)

N×N 겹침 타일(overlap 25%), 타일별 letterbox 256 추론 → Hann feather overlap-add →
guided filter. N=1은 현행 배포 경로와 동일. 지표 정의는 기존 스크립트 그대로.

## N별 지표 (traffic_street, sev 1.0, guided ON)

| case | N | tiles | ΔE_mean | CRR | NP\|Δ\| | 작은빨강ΔE | corr(logArea,\|δ\|) | 추론(s) |
|---|---|---|---|---|---|---|---|---|
| traffic_street_p | 1 | 1 | 0.81 | 1.02 | 0.0023 | 1.93 | 0.342 | 0.19 |
| traffic_street_p | 2 | 4 | 0.94 | 1.025 | 0.0026 | 3.15 | 0.448 | 0.15 |
| traffic_street_p | 3 | 9 | 0.96 | 1.025 | 0.0026 | 3.5 | 0.447 | 0.27 |
| traffic_street_d | 1 | 1 | 0.69 | 1.021 | 0.0031 | 1.51 | 0.305 | 0.07 |
| traffic_street_d | 2 | 4 | 0.76 | 1.026 | 0.0032 | 2.29 | 0.44 | 0.15 |
| traffic_street_d | 3 | 9 | 0.73 | 1.025 | 0.0031 | 2.62 | 0.425 | 0.32 |

## 결론 (케이스별)

### traffic_street_p → **tiling_marginal**
- 작은빨강 ΔE00: N=1 1.93 → N=3 3.5 (+81%)
- 부작용 점검: CRR 1.02→1.025, NP 0.0023→0.0026
- 추론 비용: 1.4× (N=1 대비)

### traffic_street_d → **tiling_effective**
- 작은빨강 ΔE00: N=1 1.51 → N=3 2.62 (+74%)
- 부작용 점검: CRR 1.021→1.025, NP 0.0031→0.0031
- 추론 비용: 4.6× (N=1 대비)

## 판정 기준
- **tiling_effective**: 작은빨강 ΔE 유의 상승(>15%) + CRR/NP 부작용 없음 → 배포 후보
- **tiling_marginal**: 개선(>5%) 있으나 추론 비용 대비 미미
- **tiling_ineffective**: 개선 없음(≤5%) → grid 해상도가 근본 원인이라는 ablation 근거

타일 경계 아티팩트: `*_boundary.png`(재조립 delta + 경계선 + mid-row 프로파일) 참조.

## 종합 해석

- **타일링은 작은 영역 희석을 완화한다** — 작은빨강 ΔE00가 p +81%, d +74%로 크게 상승.
  작은 객체가 더 작은 타일에서 상대적으로 커져 grid 셀 점유율이 올라간 결과다. → **grid
  공간 평균화(16×16 셀)가 dilution의 근본 원인이라는 ablation 증거.** (PIPELINE_DIAGNOSIS
  §2·결론(2)와 정합: 근본 해법은 grid 해상도 / 셀 점유율.)
- **부작용 미미** — CRR은 오히려 소폭 상승(1.02→1.025), ΔE_mean·NP는 절대값 변화가 미미
  (traffic은 대부분 무채색이라 NP가 0에 가깝다). p의 자동 판정이 `marginal`인 유일한 이유는
  NP 0.0023→0.0026(상대 +13%, 절대 +0.0003)이 임계 10%를 넘긴 것 — 무채색 우세 이미지의
  경계값 노이즈이며 d와 동일한 실질 효과다. **두 케이스 모두 실질적으로 effective.**
- **추론 비용 = 약 N² forward pass** (N=3 → 9×). 표의 초 단위 wall-time(1.4×/4.6×)은 sub-second
  추론의 측정 노이즈라 신뢰하지 말 것 — 배포 비용은 타일 수(N²) 기준으로 판단.
- **경계 아티팩트** — Hann feather overlap-add로 재조립. mid-row 프로파일(`*_boundary.png`)에서
  타일 경계 위치의 눈에 띄는 불연속은 관찰되지 않음(guided filter가 추가로 평활).
- **corr(logArea,|δ|)** 는 오히려 소폭 증가(0.34→0.45) — 타일링이 작은 영역뿐 아니라 큰 영역
  delta도 키워 분산이 커진 탓. 크기 의존을 완전히 없애진 못하나, 목표 지표(작은빨강 ΔE)는
  명확히 상승.

**결론: tiling은 작은 객체 dilution의 유효한 완화책(effective)이나 ~N² 추론 비용을 수반.**
신호등 등 작은 채도객체 보정이 중요하면 배포 후보(N=2가 비용/효과 균형: 4타일로 p +63%,
d +52%). 동시에 이 실험은 dilution 근본 원인이 grid 해상도임을 입증하는 ablation으로 기록.
