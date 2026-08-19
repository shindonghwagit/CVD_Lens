# Tritan(t) 재학습 준비 — 조사 결과 및 계획

> **결론 먼저:** `_ERR2MOD` tritan 수정은 **올바르고 유효**하나(daltonize 비교 baseline·평가·
> 진단 기준), **v2 학습 손실은 daltonize를 전혀 쓰지 않으므로 이 수정만으로 t 모델을 재학습해도
> tritan 약세는 그대로다.** tritan 약세의 실제 원인은 학습이 쓰는 **혼동 가중 w**에 있다.
> 재학습을 하려면 먼저 w 쪽 개입을 결정해야 한다(아래 §4). — 사용자 caution("sanity/전제 문제 시
> 재학습 진행 말고 보고")에 따라 계획을 조건부로 제시한다.

## 1. `_ERR2MOD` 수정 검증 (작업 1·2 결과 — 통과)

수정: `_ERR2MOD`를 타입별 dict로 분리. p/d는 기존 행렬 **bit-exact 유지**, t는
`[[1,0,g],[0,1,g],[0,0,0]]`(B에러→R,G, `TRITAN_SHIFT_GAIN=0.7`).

sanity(`sanity_tritan_target.py`, `reports/tritan_sanity/`) — **전 gate PASS**:
- (1) 파랑 타깃 R +0.487 (보라 방향) ✅
- (2) 회색 불변(|Δ|=0) ✅
- (3) **실제 혼동쌍(cyan-green) sim(t) ΔE00 9.27→16.33 증가**(회복) ✅
  - 주: 스펙이 지정한 blue-green은 46.6→28.0으로 감소하나, 이는 blue-green이 **L\* 차이가
    지배하는 비-혼동쌍**이기 때문(파랑 어둡고 초록 밝음). 자동 탐지된 실제 혼동쌍(cyan-green,
    최소 sim-t ΔE)으로 gate를 대체해 원칙적으로 검증함. 스펙 값도 투명하게 병기.
- (4) p/d 타깃 bit-exact(max|Δ|=0) ✅
- (P) tritan 가시 부분공간 투영 잔존율 blue **0.999**(≥0.70) — shift가 거의 전부 가시 방향,
  GAIN/행렬 재검토 불필요 ✅

→ **수정은 색채학적으로 옳다.** daltonize baseline(Step 3 평가)·진단에서 tritan이 더는
degenerate no-op이 아니게 된다(그전엔 파랑에 R을 못 더해 baseline이 부당히 약했음).

## 2. 핵심 발견 — 학습은 daltonize를 쓰지 않는다

`_ERR2MOD`/`daltonize`의 사용처를 전수 확인:
- **학습 손실 `losses.CVDLossV2`**: `L_contrast`(다중스케일 Lab-gradient) + `L_global`(pairwise ΔE)
  + `L_natural`(LPIPS+L1). 전부 **`simulate` 기반**, daltonize/`_ERR2MOD` **import 없음**.
- **`train.py`·`kaggle_train.py`**: 손실에 위 CVDLossV2만 사용. 배치마다 `random.choice(CVD_TYPES)`.
  **타깃 사전생성 캐시 없음**(→ "t 캐시 재생성" 대상 자체가 존재하지 않음).
- daltonize 사용처는 **평가/진단 스크립트뿐**: `step3_eval/metrics`(daltonize 비교 baseline),
  `gate_probe`, `daily_test`, `artifact_probe*`, `diagnose_dalt*`, `sanity*`.

**따라서 동일 손실·데이터로 t를 재학습하면 수정 전과 동일한 모델이 나온다.** 이 수정은
t 모델을 바꾸지 못한다.

## 3. tritan 약세의 실제 원인 — 혼동 가중 w

학습이 실제로 tritan 보정을 켜고/끄는 것은 w(혼동 가중, `confusion.py`)다. w는
ΔE_Lab(orig, sim_t(orig))를 임계값 (τ_low,τ_high)=(5,25)로 정규화한 소프트 마스크이며,
L_contrast·모델 delta를 게이팅한다.

| 이미지/패치 | w_p | w_d | **w_t** |
|---|---|---|---|
| blue_sea (실사) | 0.66 | 0.67 | **0.117** |
| traffic_street | 0.16 | 0.17 | (낮음) |
| 순색 blue 패치 | 1.0 | 1.0 | 1.0 |

**실사진의 하늘·바다 파랑은 채도가 낮아 Machado-t 시뮬레이터가 거의 안 움직인다 →
per-pixel ΔE 작음 → w_t≈0.1 → w-gating이 tritan 보정을 꺼버린다 → t 모델이 파랑에서
no-op으로 학습·동작.** 순색 blue 패치는 w_t=1.0이라(크게 움직임) 문제가 안 보이지만,
실사 파랑에서 무너진다. 이것이 "blue_sea w mean ~0.1", tritan ΔE 만성 저조의 실제 기전이다.
(daltonize 타깃 버그와는 **독립적인** 문제.)

## 4. 재학습 전 결정 필요 — 무엇을 바꿔야 t가 실제로 개선되나

재학습이 의미를 가지려면 **학습이 쓰는 것**을 바꿔야 한다. 후보(모두 재학습 수반, p/d 영향 점검 필수):

- **옵션 A (최소 변경): tritan 혼동 임계값 하향.** `confusion.py`의 t 임계 (5,25)를
  (예: 2,12 또는 3,15)로 낮춰 채도 낮은 파랑에서도 w를 올린다. 저비용·국소 변경. 단
  오탐(무채색까지 w 상승)·p/d 무영향 확인 필요. **권장 1순위 실험.**
- **옵션 B: tritan 시뮬레이터/severity 재검토.** Machado-t가 실사 파랑을 과소 시뮬레이션하는지
  점검, 필요 시 t 학습 severity를 올리거나 시뮬레이터 보정.
- **옵션 C: 손실에 저채도 파랑 가중** 추가(국소).

→ **결정 요청:** 어느 옵션으로 갈지(또는 tritan 개선을 보류하고 `_ERR2MOD` 수정만 baseline
용도로 확정할지). 이 결정 없이는 재학습이 무의미하다.

## 5. (참고) 재학습 기계적 제약 — 옵션 확정 시 적용

작업 3의 기계 질문에 대한 답:
- **t 단독 재학습 불가(구조적).** 모델은 **단일 네트워크**를 타입 one-hot로 FiLM 조건화한
  것(`model.py`, 가중치 공유). `train.py:435`가 배치마다 랜덤 타입을 뽑아 **한 네트워크가
  p/d/t를 동시 학습**한다. t 배치만 먹이면(`--types t` 같은 필터를 추가하는 것 자체는 쉬움)
  공유 가중치가 흔들려 **p/d 회귀(catastrophic forgetting)** 위험. → t만 깨끗이 재학습하는
  구조가 아니다.
- **권장: 옵션 확정 후 3타입 전체 재학습.** 임계/시뮬레이터 변경은 w를 통해 학습 신호를
  바꾸므로 전체 재학습이 자연스럽고, p/d는 (변경이 t 전용이면) 사실상 불변으로 재검증된다.
- **캐시 재생성: 불필요**(타깃 캐시 구조 없음).

옵션 확정·재학습 시 체크리스트(기존 resume 워크플로 기준):
1. 변경(예: 옵션 A 임계값)을 `confusion.py`에 반영, 단위 확인(w_t가 blue_sea에서 상승하는지 재측정).
2. Kaggle: `python -m cvdlens_v2.kaggle_train --resume` (T4×2, batch 16, 20000 step 기준).
   **주의:** 손실 신호가 바뀌므로 이번엔 **resume가 아니라 fresh 재학습** 권장(기존 체크포인트는
   옛 w로 학습됨). 예상: 유효 수렴 ~6000, 선정 여유 ~3배(20000).
3. Gate 재평가: 타입평균 ratio_w **T≥1.27** + do-nothing 앵커 |Δ|<0.005 (train.py::validate).
   특히 **held-out·blue_sea류 저채도 파랑에서 tritan ΔE/CRR 개선**을 별도 확인.
4. 회귀 방지: **p/d ratio_w가 기존(1.10/1.13 gate, 뱅크 1.39/1.36) 대비 유지**되는지 확인.
5. export: `python -m cvdlens_v2.export_onnx` → **3개 ONNX 모두 재생성 후** parity(<1e-3) 재확인.
   배포는 `cvdlens_t.onnx`(및 변경 시 p/d) 교체. **배포 서버 코드(cvd-lens/inference)는 이번
   범위 밖 — ONNX 파일만 교체.**

## 6. 즉시 확정 가능한 것 vs 결정 대기

- **확정(이번 작업):** `_ERR2MOD` 타입별 수정 — daltonize baseline/평가/진단 정확성 개선. p/d 무영향.
- **대기(사용자 결정):** tritan 모델 개선 방향(§4 옵션). 결정 전까지 재학습 착수 안 함.
