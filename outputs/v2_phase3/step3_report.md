# Phase 2 Step 3 — CRR/NP Evaluation: CVDLens vs Daltonize

- Checkpoint: `model_best` (step 9000), severity 1.0
- Eval set: 60 images (top/mid/low 20×3), seed 20260803, pool 300
- Disjoint from Phase 1 bank (10) + held-out set (8)
- Fairness: both methods at 256², same simulator (machado, sev 1.0), same confusion weight w. Daltonize = `simulation.daltonize` (Brettel/error-shift; no external library).

## Metrics
- **CRR** (axis 1, ↑ better): `ratio_w` — confusion-weighted contrast ratio of sim(method) vs sim(original) on the CVD view (Phase 1 def).
- **NP** (axis 2, ↓ better): `|Δ|` mean sRGB change; `LPIPS` (VGG) perceptual distance to the original.
- Secondary: SI_uniform, corr_guide (Phase 1 logging set).

## Per-type summary (mean ± std)

| type | method | CRR ↑ | NP \|Δ\| ↓ | NP LPIPS ↓ | SI_uniform | corr_guide |
|---|---|---|---|---|---|---|
| p | cvdlens | 1.188 ± 0.444 | 0.0171 ± 0.0210 | 0.0423 ± 0.0498 | 0.371 | 0.465 |
| p | daltonize | 0.975 ± 0.328 | 0.0370 ± 0.0401 | 0.0738 ± 0.0646 | 0.211 | 0.351 |
| d | cvdlens | 1.159 ± 0.407 | 0.0193 ± 0.0218 | 0.0377 ± 0.0429 | 0.352 | 0.480 |
| d | daltonize | 0.924 ± 0.286 | 0.0241 ± 0.0270 | 0.0565 ± 0.0695 | 0.224 | 0.351 |
| t | cvdlens | 1.166 ± 0.414 | 0.0211 ± 0.0220 | 0.0573 ± 0.0629 | 0.284 | 0.530 |
| t | daltonize | 0.915 ± 0.279 | 0.0313 ± 0.0258 | 0.0688 ± 0.0451 | 0.214 | 0.474 |

> **SI_uniform이 CVDLens에서 더 높은 것은 결함이 아니라 구조 차이**다: daltonize는 전역에 가까운 저주파 색 이동이라 균일 영역의 delta가 거의 균일(SI_uniform 낮음)한 반면, CVDLens는 guide-aligned 국소 보정이라 균일-휘도 혼동 영역에서도 delta가 공간 구조를 가진다 — 이 고주파 성분이 잡음이 아니라 휘도 가이드와 정렬돼 있음은 corr_guide(CVDLens 0.47–0.53 > daltonize 0.35–0.47, Phase 1 검증셋 정의)가 뒷받침한다.

## Per-tier summary (confusion-mass stratum)

CRR (`ratio_w`) is only meaningful where the confusion weight w has mass. In the **low** tier (w̄≈0) the weighted ratio divides two near-zero quantities and is unstable — read the low tier as a **naturalness / do-nothing test** (|Δ|, LPIPS should be ~0), not as recovery.

| tier | method | CRR | NP \|Δ\| | NP LPIPS |
|---|---|---|---|---|
| top | cvdlens | 1.477 | 0.0424 | 0.0994 |
| top | daltonize | 1.042 | 0.0653 | 0.1286 |
| mid | cvdlens | 1.230 | 0.0142 | 0.0349 |
| mid | daltonize | 1.016 | 0.0227 | 0.0529 |
| low | cvdlens | 0.807 | 0.0010 | 0.0031 |
| low | daltonize | 0.756 | 0.0043 | 0.0176 |

## Paired comparison (CVDLens − Daltonize, Wilcoxon signed-rank)

Positive CRR diff = CVDLens recovers more. Negative NP diff = CVDLens damages less (better).

The **CRR** test population is restricted to the **top+mid tiers (n=40/type)** — the
w-meaningful strata — so the paired test matches the verdict criterion (low-tier
`ratio_w` is unstable at w̄≈0, see §Per-tier / §Limitations, and is excluded).
The **NP** tests stay on **all tiers (n=60/type)**, since |Δ|/LPIPS are well-defined
everywhere (the low tier is precisely where naturalness must hold).

| type | metric | population (n) | CVDLens | Daltonize | diff (mean) | p |
|---|---|---|---|---|---|---|
| p | CRR | top+mid (40) | 1.380 | 1.083 | +0.298 | 1.82e-11 |
| p | NP_delta | all (60) | 0.017 | 0.037 | -0.020 | 1.63e-11 |
| p | NP_lpips | all (60) | 0.042 | 0.074 | -0.031 | 1.09e-09 |
| d | CRR | top+mid (40) | 1.337 | 1.009 | +0.328 | 1.82e-12 |
| d | NP_delta | all (60) | 0.019 | 0.024 | -0.005 | 1.84e-04 |
| d | NP_lpips | all (60) | 0.038 | 0.056 | -0.019 | 4.98e-04 |
| t | CRR | top+mid (40) | 1.343 | 0.996 | +0.347 | 1.82e-12 |
| t | NP_delta | all (60) | 0.021 | 0.031 | -0.010 | 7.26e-07 |
| t | NP_lpips | all (60) | 0.057 | 0.069 | -0.012 | 8.22e-03 |

(The per-type CRR means in the table above are top+mid; the §Per-type summary table
reports the all-60 mean and so reads slightly lower — same rows, wider population.)

## Verdict

- **CRR (top+mid, w-meaningful)**: CVDLens **1.353** vs Daltonize **1.029** → recovery ≥ (동등 이상). (Daltonize sits near 1.0 — little net weighted-contrast gain on the CVD view.)
- **NP (all tiers)**: |Δ| CVDLens **0.0192** vs **0.0308**; LPIPS **0.0458** vs **0.0664** → damage lower (better), significant on all types (p<0.05).
- **Low tier do-nothing (naturalness)**: |Δ| CVDLens **0.0010** vs Daltonize **0.0043** — CVDLens leaves low-confusion images nearly untouched; daltonize still perturbs them.

**HYPOTHESIS SUPPORTED**: on w-meaningful images CVDLens recovers contrast at least as well as daltonize (in fact more), while damaging naturalness less across all tiers (significant on every type).

> Caveat: low-tier `ratio_w` is numerically unstable (w̄≈0) and is excluded from the CRR verdict; it is used only as a naturalness check.

## Scatter

![scatter](step3_scatter.png) — upper-right is good (high recovery, low damage).

## 평가의 한계 (Limitations)

본 절은 각주가 아닌 본문으로, 위 수치의 해석 범위를 명시한다.

- **시뮬레이터 기준 평가.** 모든 지표(CRR, NP)는 색각이상 당사자의 실제 지각이
  아니라 **시뮬레이션된 CVD 관점** 위에서 계산되었다. 회복(CRR)의 평가 시뮬레이터는
  Machado(sev 1.0), daltonize의 재색상 경로는 Brettel/error-shift 모델을 사용한다.
  즉 본 결과는 "이 모델들이 예측하는 CVD 관점에서 대비가 얼마나 회복/손상되는가"를
  측정하며, 실제 색각이상 관찰자의 지각과의 일치 여부는 **본 연구의 검증 범위 밖**이다
  (사용자 대상 심리물리 실험이 별도로 필요).
- **순환 의존성은 부분 완화.** 회복(Machado)과 daltonize(Brettel)가 서로 다른 모델을
  써서 "자기 시뮬레이터에 과적합된 회복"을 부분적으로 통제했으나, 두 모델 모두 동일한
  LMS-결손 가정을 공유하므로 완전한 독립은 아니다.
- **단일 severity 조건.** 평가는 완전 이색형(dichromat, sev 1.0)에서만 수행했고,
  경도~중등도 이상삼색형(anomalous trichromat)은 다루지 않았다.
- **low tier CRR 제외 (재확인).** low tier(w̄≈0)의 `ratio_w`는 두 near-zero 값의
  비율이라 수치적으로 불안정하므로 CRR 판정에서 제외했고(§Per-tier, §Verdict), 오직
  naturalness(do-nothing) 점검용으로만 읽는다.

## Case studies (5-panel witnesses)

세 케이스는 각각 최대 우세 / 최소 격차 / **CVDLens가 회복에서 지는** 케이스로,
평균 뒤에 숨는 편차를 본문에 노출하기 위한 것이다.

### 최대 CVDLens 우세 (회복 동등 이상 + 손상 낮음) — `000000111951` / Protanopia
CRR margin (CVDLens−Dalt) +0.417, |Δ| gap +0.1480, LPIPS gap +0.1852
![bigwin](step3_case_bigwin.png)

### 두 방법 간 최소 격차 — `000000404534` / Protanopia
CRR margin (CVDLens−Dalt) +0.023 (회복은 사실상 동률), 그러나 NP는 CVDLens가 크게 우세:
|Δ| gap +0.0178 (0.0064 vs 0.0241), LPIPS gap +0.0626 (0.0146 vs 0.0772). daltonize가
장면 전체를 청색으로 밀어내는 전역 이동을 쓰는 반면 CVDLens는 자연스러움을 보존한다
(lose 케이스와 stem 분리 — 서로 다른 이미지).
![similar](step3_case_similar.png)

### CVDLens가 회복에서 지는 케이스 — `000000183104` / Protanopia
CRR margin (CVDLens−Dalt) **-0.026** (daltonize가 이 이미지에서 대비를 더 회복).
단 NP는 여전히 CVDLens가 우세: |Δ| gap +0.0186, LPIPS gap +0.0346.
평균 CRR 우세(top+mid +0.30~0.35)에도 개별 이미지에서는 daltonize에 뒤지는 사례가
존재함을 명시한다.
![lose](step3_case_lose.png)
