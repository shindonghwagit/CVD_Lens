# tritan 혼동 가중 w 정의 진단 (작업 2)

confusion.py 미변경, 실험 스크립트 내 변형만. W0=현행(Machado+현재임계값, 동일 코드경로).
W1=Brettel 시뮬. W2=Machado+타입별 캘리브(순 혼동색 w=1). W3=Brettel+캘리브.

**판정 기준:** 혼동영역 w_mean ≥ 0.5 & 무채색 w_mean ≤ 0.1 (핵심 이미지 ['blue_sea', 'traffic_street']).

## 결론: **채택=W1b**

H1 CONFIRMED (simulator). Machado-t under-moves real desaturated blue (pure blue dE=118 but sky/sea blue tiny), so W0 tritan w collapses on real blue. Brettel moves real blue properly (blue_sea conf_w 0.33→0.98). Fix = t-only Brettel sim + retuned thresholds (12.0, 30.0); p/d unchanged (Machado) → zero regression. Note: W2 (pure-color calibration) is counterproductive — pure ΔE76 is huge (86-118) so it RAISES the threshold and kills w.

- W0 대비 혼동영역 conf_w 개선: {'blue_sea': 0.633, 'traffic_street': 0.444}
- 엄격 기준(두 실사 모두 동시 충족) 통과 변형: 없음 (blue_sea는 W1b 단독 충족; traffic caveat 아래)
- p/d 훼손: **0.0** (W1b는 t 전용이라 p/d 미변경 → 회귀 0)
- traffic caveat: traffic_street (mostly-achromatic city, weak pale-blue signal) improves conf_w 0.571 vs W0 0.127 but achr sits ~0.13 (>0.1) — no single threshold satisfies both saturated sea-blue and pale sky-blue; blue_sea (diagnosed case) passes cleanly.

## 캘리브레이션 임계값 (순 혼동색 → w=1)

| type | machado low/high (conf dE) | brettel low/high (conf dE) |
|---|---|---|
| p | 12.9/77.4 ([87.0, 77.4]) | 11.02/66.1 ([94.7, 66.1]) |
| d | 13.25/79.51 ([85.8, 79.5]) | 11.31/67.88 ([94.7, 67.9]) |
| t | 17.2/86.0 ([118.4, 86.0]) | 0.0/0.01 ([0.0, 0.0]) |

## 이미지별 tritan w (혼동영역 / 무채색)

| image | conf/achr frac | W0 | W1 | W2 | W3 |
|---|---|---|---|---|---|
| blue_sea | 0.131/0.352 | 0.333/0.014 | 0.982/0.22 | 0.002/0.0 | 1.0/0.955 | 0.966/0.084 |
| traffic_street | 0.277/0.47 | 0.127/0.019 | 0.756/0.254 | 0.004/0.0 | 0.998/0.902 | 0.571/0.134 |
| tennis_proxy | 0.431/0.028 | 0.984/0.498 | 0.984/0.498 | 0.589/0.196 | 0.986/0.562 | 0.984/0.498 |

> 셀 = 혼동영역 w_mean / 무채색 w_mean. 굵은 목표: conf≥0.5, achr≤0.1.

## p/d 보존 점검 (혼동영역 conf_w, W0 대비 W2/W3 변화)

| image | type | W0 | W1 | W2 | W3 |
|---|---|---|---|---|---|
| blue_sea | p | nan | nan | nan | nan | nan |
| blue_sea | d | nan | nan | nan | nan | nan |
| traffic_street | p | 0.413 | 0.439 | 0.051 | 0.083 | 0.413 |
| traffic_street | d | 0.421 | 0.463 | 0.049 | 0.086 | 0.421 |
| tennis_proxy | p | 0.987 | 0.987 | 0.378 | 0.373 | 0.987 |
| tennis_proxy | d | 0.987 | 0.987 | 0.363 | 0.381 | 0.987 |

## 순색 패치 tritan w

| patch | W0 | W1 | W2 | W3 |
|---|---|---|---|---|
| blue | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| cyan | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 |
| yellow | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 |
| purple | 1.0 | 1.0 | 0.687 | 1.0 | 1.0 |
| gray | 0.0 | 0.0 | 0.0 | 0.013 | 0.0 |
| skin | 0.547 | 1.0 | 0.0 | 1.0 | 1.0 |

몽타주: `*_w.png` (이미지별 W0~W3 w 맵).
