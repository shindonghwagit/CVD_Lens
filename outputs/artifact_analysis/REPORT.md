# 보정 품질 이슈 조사 — Investigation #1: 균일 색 영역의 공간 불균일 아티팩트

- 대상: 배포 파이프라인(`cvd-lens/inference/main.py`)을 로컬 ONNX로 **동일 재현**
  (cap 2048 → letterbox 256 → per-type ONNX → delta content-box → bilinear
  upsample → composite). 웹 라운드트립 없음.
- 시뮬레이션 필요 시 Brettel-1997 NumPy(`cvdlens_v2.simulation`) 사용. daltonize import 없음.
- 스크립트: `cvdlens_v2/artifact_probe.py` · 산출물: `outputs/artifact_analysis/`

## 파이프라인에서 미리 확정한 사실 (코드 근거)

- 보정은 **256×256에서 계산**되고 delta만 native로 **bilinear 업샘플**된다
  (`main.py:_correct_image`). 따라서 native ΔE 맵의 얼룩은 256-space에서 발원한다.
- bilateral grid 공간 해상도 = **16×16**, guide(휘도)축 depth = **8**
  (`cvdlens_v2/model.py` `grid_spatial=16, grid_depth=8`). 256에서 grid cell 한 칸 =
  **256/16 = 16px**.
- delta = out − in을 원본에 더하므로 ΔE(원본, 보정본) ≈ |delta_full|의 색차. 즉 ΔE
  히트맵은 보정이 실제로 건드린 곳/양을 그대로 보여준다.

## 사전 등록 예상 (Pre-registered expectations) — 실행 전 기록

- **E1 (ΔE 불균일):** 균일 red 영역인데도 보정본 ΔE00가 공간적으로 출렁인다(얼룩).
  근거: 위 파이프라인 사실 + 사용자 관찰.
- **E2 (guide 리플):** 실사 STOP의 "균일" red 영역에서도 휘도 guide가 미세하게
  변동(조명·JPEG 텍스처)해 guide std > 0. 이 변동이 D=8 guide-bin 경계를 넘나들면
  인접 픽셀이 다른 D-slice를 표집.
- **E3 (grid 주기):** 얼룩의 공간 주기가 grid cell(256÷16=16px)과 정렬. 크롭 자기상관
  1st peak ≈ 16px 부근이면 grid 해상도 원인 확정.
- **E4 (합성 대조):** 완전 균일 red(합성 flat)는 backbone feature·guide가 상수라
  얼룩이 거의 없다(ΔE 공간표준편차 ≈ 0). 반면 미세 휘도 램프(synth_ramp)를 주면
  얼룩이 생긴다 → **얼룩은 입력 변동(guide/feature)이 grid를 통해 증폭된 결과**임을
  분리 입증.
- **불확실:** ΔE 주기가 정확히 16px가 아니라 그 배수/분수(guide-bin 전이 위치에
  따라)로 나타날 수 있음. peak가 8~32px 사이면 "grid 정렬"로 관대하게 판정하되
  수치 기록.

## 실행 결과 (Hit/Miss)

스크립트: `artifact_probe.py`(파이프라인+ΔE+guide+grid), `artifact_probe2.py`(기전).
수치: `probe_stats.json`, `mechanism_stats.json`. 그림: `*_probe.png`, `mechanism_stop_p.png`.

| 예상 | 판정 | 근거 수치 |
|---|---|---|
| E1 ΔE 불균일 | **HIT** | 균일 red 크롭(64,88–100,124): 원본 hue std **18.5°** → 보정본 hue std **140.2°**, 크롭 ΔE00 mean 21.6 **std 10.3** |
| E2 guide 리플 | **HIT** | "균일" red 크롭에서 guide range **0.577**(= D-bin 0.43→2.45, **약 2.0/8 bin**); 전체 STOP guide_std 0.82 |
| E3 grid 16px 주기 | **MISS(반증)** | 크롭 delta 행 자기상관에 16px 주기 peak **없음**. 아티팩트는 16×16 **공간격자 주기 banding이 아님** |
| E4 합성 대조 | **HIT(결정적)** | synth_flat(guide_std=**0.0000**): 내부 delta 균일, 내부 얼룩 없음(테두리 halo만). guide 변동이 있어야 내부 얼룩 발생 |

### 원인 확정 (root cause)

사용자의 "16×16 공간격자 한계" 가설은 **부분적으로만** 맞고, 지배적 기전은 다르다:

1. **guide/깊이축(D=8) 기반 색매핑이 채도 높은 red에서 과도하게 가파르다.**
   보정 색상은 휘도 guide의 **가파른(계단형에 가까운) 함수**다(`mechanism_stop_p.png`
   우하단 산점도: guide-bin 0.5→2.5에서 hue가 orange 40°→green 150°→magenta 340°로 점프).
   Red는 (i) 휘도가 낮아 guide가 압축된 구간이고 (ii) confusion weight w≈1로 최대 보정이
   적용되는 영역이라, 조명·JPEG·미세 텍스처로 인한 **작은 휘도 변동이 ~2개 depth bin을
   넘나들며 큰 hue 변동(18.5°→140°)으로 증폭**된다. 이것이 노랑↔주황↔갈색 얼룩의 정체다.
2. **입력 변동이 없으면 내부 얼룩도 없다.** 완전 균일 red(synth_flat)는 내부 delta가
   균일하다 → 얼룩은 격자 자체의 결함이 아니라 **입력 휘도 변동이 가파른 guide 매핑을
   통해 증폭된 결과**.
3. **별도 아티팩트 — 테두리 halo.** synth_flat에서도 256 이미지 경계에 강한 ΔE 띠가 있다
   (conv padding 경계 효과). 균일 입력에서도 나타나므로 위 얼룩과 원인이 다르다.
4. **보정 강도 자체가 매우 크다.** 완전 균일 saturated red의 평균 ΔE00가 **~31**(synth_flat).
   protan에서 red는 confusion 정중앙이라 w≈1·큰 delta가 정상 동작이나, 그 절대 크기가
   red→brown/orange 급변으로 지각된다 → 강도 기본값 하향(#4)의 직접 근거.

**함의(모델 수정 방향, 이번엔 구현 금지):** 공간 16×16을 키우는 것보다 **guide축 매핑의
평활화**(D 증가 또는 guide에 대한 Lipschitz/TV 정규화, 저휘도 채도영역의 매핑 기울기 제한)가
핵심. 테두리 halo는 padding/컨텍스트 처리 별도 과제. 이는 한계(limitations) 재료로 기록만.


---

# Investigation #2 — 비대상 영역 침범 (w-gate 선택성)

스크립트 `cvdlens_v2/gate_probe.py` · 그림 `gate2_*.png` · 수치 `gate_stats.json`.
w는 ONNX 그래프가 내부에서 쓰는 것과 동일한 식(`confusion.compute_confusion_weight`,
Machado sim + Lab ΔE threshold + blur)으로 재계산. 256 추론 공간에서 측정.

## 사전 등록 예상
- w-gate가 정상이면 무채색/저채도(sat<0.12) 픽셀의 |Δ|가 채도 높은 픽셀보다 현저히
  작고, corr(w, |Δ|) > 0.

## 결과 (Hit/Miss): **HIT (w-gate 선택성 정상)**

| 이미지·타입 | 무채색 |Δ| median | 무채색 |Δ| p90 | 채도 |Δ| median | corr(w,|Δ|) |
|---|---|---|---|---|
| stop · p | **0.000** | 0.027 | 0.212 | 0.83 |
| stop · d | **0.000** | 0.031 | 0.230 | 0.83 |
| sea · t | 4.9e-6 | 0.0019 | 0.0074 | 0.73 |

- 무채색 픽셀의 **|Δ| 중앙값이 0**(p/d) — w-gate가 무채색 이동을 실제로 억제한다.
  corr(w,|Δ|)=0.73–0.83로 강한 양의 상관. **선택성은 설계대로 작동.**
- 단, 무채색 p90 ≈ **0.027–0.031**의 꼬리가 있다: 채도영역에 **인접한** 무채색 픽셀
  일부가 움직인다. 이는 (i) sim ΔE가 밝은 회색에서 완전 0이 아닌 점 + (ii) **native
  해상도에서 delta를 bilinear 업샘플**할 때 채도영역 delta가 경계 너머로 번지는 것
  (256→2048은 8× 확대라 256의 1px 번짐이 native 8px)에서 온다. → "유리창 주황 끼"는
  **게이트 실패가 아니라 업샘플/경계 번짐**. 개선은 경계 인식 업샘플 or w로 delta_full을
  재마스킹(추론단, 별도 과제).

# Investigation #3 — Tritan 전역 desaturation

스크립트 `gate_probe.py`(실사 teal sea) + 합성 패치(`tritan_synth_patches.json`).

## 사전 등록 예상
- tritan의 w-gate 방향/basis가 p/d와 동일 로직이면, 파란 장면에서 tritan w가 높아
  (파랑=tritan confusion color) 큰 보정이 "설계대로" 걸린다. tritan만 방향벡터가
  잘못됐다면 basis 코드가 p/d와 다를 것.

## 결과 (Hit/Miss): **basis 정의 HIT, "tritan 버그" MISS — 설계대로 동작**

**basis/게이트 로직 대조 (코드):** `cvdlens_v2/basis.py` `_visible_basis`는 세 타입 모두
**동일한 SVD 로직**(Brettel 행렬 영공간=confusion 방향, 가시평면 직교기저)으로 유도된다.
타입별로 다른 것은 w 임계값뿐: p/d=(2,12), **t=(5,25)** (`confusion.py:18–22`). 방향벡터
정의 오류 없음.

**실사 teal sea:** tritan w_mean=**0.094**(p/d 0.60), NP|Δ|=**0.005**(최소). teal은
청록(blue-green)이라 tritan-가시평면에 가까워 w가 낮다 — 이 이미지만 보면 tritan은
**과보정이 아니라 최소보정**. (사용자 관찰과 표면상 상충 → 원인은 이미지 색조.)

**합성 패치로 색조 분리 (`tritan_synth_patches.json`):**

| 패치 | p w | d w | **t w** | t δ |
|---|---|---|---|---|
| pure_blue [.10,.20,.75] | 1.0 | .88 | **1.0** | 0.106 |
| sky_blue [.35,.55,.85] | .21 | .25 | **1.0** | **0.134** |
| teal [.10,.45,.50] | 1.0 | 1.0 | **.29** | 0.039 |
| tan_sand [.80,.70,.50] | .34 | .37 | **1.0** | 0.052 |
| cyan [.10,.7,.8] | 1.0 | 1.0 | **.62** | **0.239** |

- **순수/하늘 파랑에서 tritan w=1.0** → 큰 delta(0.11–0.13). 사용자의 "단색 blue →
  전역 물빠짐"은 **재현되며 설계대로**다: 파랑·노랑이 tritan의 confusion color라 파란
  장면 전체에서 w→1 → 최대 보정이 전역에 걸린다 (protan-on-red(#1)와 동일 현상).
- 내 첫 teal sea가 낮게 나온 건 청록이라서. "파도 노란 끼"는 tan/노랑도 tritan
  confusion(tan_sand w=1.0)이라 약간 노란 거품·모래 경계가 밀린 것.

## #1·#3 공통 결론
둘 다 **"대면적이 채도 높은 confusion color일 때 w가 1로 포화 → (guide축이 가파른)
보정이 전역/강하게 적용"** 이라는 동일 현상. protan=red, tritan=blue/yellow. severity
하향(#4)이 직접 완화책이고, 근본 개선은 #1의 guide축 매핑 평활화.
