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

→ **결정 완료(§7, 작업 2·3):** 옵션 B의 정제형 채택 — **tritan w를 Brettel 시뮬 +
임계값 (12,30)으로(=W1b, t 전용)**. H1(시뮬레이터) 확정. 상세는 §7.

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

- **확정:** `_ERR2MOD` 타입별 수정 — daltonize baseline/평가/진단 정확성 개선. p/d 무영향(커밋 3cc6c00).
- **확정(§7):** tritan w 정의 = Brettel 시뮬 + (12,30), t 전용. 재학습 대기(실행 금지).

---

## 7. w 진단 결과 (작업 2·3) — 원인 규명 및 채택 w 정의

근거: `reports/tritan_w_diagnosis/` (`w_diagnosis.json`, `loss_impact.json`, 몽타주).
모든 W0 기준선은 `compute_confusion_weight` 동일 코드경로로 재산출(기존 리포트 수치 미인용).

### 7.1 H1 확정 — Machado-t 시뮬레이터가 원인

w 변형 4종을 오프라인 비교(confusion.py 미변경, 실험 스크립트 내에서만):

| 변형 | 정의 | blue_sea conf_w / achr_w | traffic conf_w / achr_w |
|---|---|---|---|
| **W0** 현행 | Machado + (5,25) | 0.333 / 0.014 | 0.127 / 0.019 |
| **W1** | Brettel + (5,25) | **0.982** / 0.220 | 0.756 / 0.254 |
| W2 | Machado + 순색캘리브 | 0.002 / 0.0 | 0.004 / 0.0 |
| W3 | Brettel + 순색캘리브 | 1.0 / 0.955 | 0.998 / 0.902 |
| **W1b** 채택 | **Brettel + (12,30), t전용** | **0.966 / 0.084** | **0.571 / 0.134** |

- **H1 확정(시뮬레이터):** Brettel로 바꾸면 혼동영역 w가 급등(blue_sea 0.33→0.98). 핵심 기전:
  Machado-t는 **순색 파랑을 과대이동(dE=118)하나 실사 저채도 파랑은 과소이동**하는 반면,
  Brettel-t는 실사 파랑(sky/sea dE 33–37)을 제대로 움직인다. 실사는 저채도 파랑이라 Brettel이
  옳다.
- **H2(스케일) 기각:** W2(순색 w=1 캘리브)는 **역효과** — 순색 ΔE76가 86–118로 커서 임계값을
  되레 올려 w를 붕괴시킴(0.002). W3는 축퇴(Brettel 순색 dE≈0 → 임계 0 → 전부 w=1).
- **W1의 무채색 누수(achr 0.22)를 임계값 (12,30)으로 억제 = W1b.** blue_sea 0.966/0.084로 기준
  충족. traffic은 conf 0.571(4.5× 개선)이나 achr 0.134로 경미 초과 — 도시 무채색 우세 +
  옅은 하늘 파랑이라 단일 임계로 완벽 분리 불가(스캔 확인). blue_sea(진단 케이스)는 정상 통과.
- **p/d 회귀 0:** W1b는 t 전용(p/d는 Machado+(2,12) 그대로) → 구조적으로 p/d w 미변경.

### 7.2 손실 영향 (작업 3) — 재학습 유인 확인

현행 t 체크포인트(=배포 `cvdlens_t.onnx`, step 9000) 출력에 **같은 C_o/C_a, w만 교체**해
`L_contrast` 재계산:

| 이미지 | L_c (W0 → W1b) | 배율 | 혼동영역 weighted-deficit 배율 |
|---|---|---|---|
| blue_sea | 0.0256 → 0.103 | **×4.04** | ×2.45 |
| traffic_street | 0.0167 → 0.047 | ×2.82 | ×2.44 |
| tennis_proxy | 37.4 → 40.1 | ×1.07 (이미 포화) | ×1.11 |

→ 새 w에서 파랑 영역 손실 기여가 **2.4–4×** 커진다. 재학습 시 모델이 파랑을 움직일
gradient 유인이 실제로 생긴다는 정량 근거.

### 7.3 적용 지점 (재학습 시 — 아직 미적용, 실험 스크립트에만 있음)

**변경은 `confusion.py`의 t 분기에 국한**(p/d 미변경):
1. `_THRESHOLDS["t"]` = (5.0, 25.0) → **(12.0, 30.0)**.
2. t의 w 시뮬레이터를 **Brettel**로: `compute_confusion_weight`에서 `cvd_type=="t"`면
   `method="brettel"` (p/d는 machado 유지). severity는 t에서 무시(Brettel 고정 1.0).
   → 이 변경은 **공용 코드 머지 전 p/d w 불변을 단위 확인**하고 넣을 것(주의 준수).

### 7.4 업데이트된 재학습 체크리스트 (t 대상, 새 w — 실행 대기)

1. `confusion.py` t 분기에 §7.3 반영. **검증:** blue_sea w_t가 0.12→~0.9로 상승, **p/d w는
   bit 수준 불변**(diag_tritan_w W1b의 p/d=W0 재확인), 무채색 w_t ≤ ~0.13.
2. Fresh 재학습(옛 체크포인트는 옛 w로 학습됨): `python -m cvdlens_v2.kaggle_train` (resume 아님).
   단일 공유망이라 3타입 동시 학습이나, **t의 w만 바뀌므로 p/d 학습 신호는 불변 → p/d 보존**.
3. Gate 재평가: 타입평균 ratio_w **T≥1.27** + do-nothing 앵커. **추가:** blue_sea류 저채도
   파랑에서 tritan ΔE/CRR가 실제 상승하는지(작업 3의 손실 유인이 실현됐는지) held-out 확인.
4. p/d 회귀 방지: p/d ratio_w가 기존(gate 1.10/1.13, 뱅크 1.39/1.36) 유지 확인.
5. export: `python -m cvdlens_v2.export_onnx` → 3개 ONNX 재생성, parity(<1e-3). 배포는
   `cvdlens_t.onnx` 교체(+ p/d는 사실상 불변이나 재검증). **서버 코드 불변, ONNX만 교체.**

> 미해결 리스크: traffic류 옅은 하늘 파랑은 (12,30)으로도 achr 경미 초과(0.134). 필요 시
> t 임계값 미세조정 또는 무채색 억제(채도 게이트 추가)를 재학습 후 held-out 결과 보고 결정.

---

## 8. Kaggle 재학습 패키지 (작업 2) — 적용 완료, 학습은 사용자 실행

### 8.0 코드 반영 상태 (커밋 b2e2c70, main push 완료)
- `confusion.py` compute_confusion_weight: **명시적 `if cvd_type=="t"` 분기** → Brettel + (12,30).
  p/d는 Machado+(2,12) 그대로. 검증: p/d w bit-exact(회귀 0), t w==실험 W1b(0.00e+00).
- `model.py wrap_for_onnx._confusion_w`: t만 고정 Brettel+(12,30)로 미러 → **train forward w ==
  ONNX export w (p/d/t 전부 0.00e+00)**. 이게 없으면 재export가 machado-w로 게이팅되어 재학습이
  배포에 안 실림(이식 오류). t ONNX export OK(parity 1.55e-7), 단위테스트 전부 통과.

### 8.1 경로 최종 점검
- **새 confusion.py 반영: 자동.** `kaggle_train → train_main → train.py`의 `model(...)` forward가
  `model.py:318 compute_confusion_weight`를 호출 → 새 t-분기가 학습 w-gating에 그대로 적용.
  별도 조치 불필요.
- **타입 필터: 없음**(train.py:435 `random.choice(CVD_TYPES)`로 3타입 동시 학습). **추가 권장 안 함**(§8.2).

### 8.2 재학습 모드 — **전체 3타입 fresh 재학습 권장 (t 단독 필터 아님)**
- 모델은 **단일 공유망**(FiLM 타입 조건화). t 배치만 학습(--types t)하면 공유 가중치가 흔들려
  **p/d 망각(regression)** — "t 단독 학습이니 p/d 0"은 공유망에선 성립 안 함.
- **올바른 방법:** 3타입 전체 fresh 재학습. **w 변경이 t 전용**이므로 p/d의 학습 신호(=p/d w)는
  불변 → p/d 최적점 동일. 단 공유 가중치가 t의 새 gradient로 미세 이동하므로 **p/d는 gate
  유지(≈불변)이나 bit-0은 아님**(정직히 명시). t만 파랑 보정을 새로 학습.

### 8.3 Kaggle 업로드 체크리스트 (사용자가 올릴 것)
1. **COCO 데이터셋** 첨부: 공개 `coco-2017-dataset`(train2017 + val2017).
2. **레포 최신 main(b2e2c70)**: `!git clone` 또는 repo zip 업로드. **확인:**
   `git -C graduation_project log -1 --oneline` → `b2e2c70`, 그리고
   `grep _T_BRETTEL cvdlens_v2/confusion.py` 히트(= t-분기 반영 확인).
3. **Fresh 보장**: 새 세션의 `/kaggle/working`은 비어 있어 resume가 옛 체크포인트를 안 잡음
   (옛 체크포인트는 옛 w로 학습됨 → 반드시 fresh). 기존 `v2_phase1/`이 있으면 삭제.
4. **실행**: `!cd .../graduation_project && python -m cvdlens_v2.kaggle_train`
   (T4×2 DataParallel, batch 16, 20000 step, LPIPS off — 기존과 동일).
5. **모니터**: gate T mean ratio_w, 그리고 저채도 파랑(하늘/바다) 보정이 살아나는지 로그/중간
   val로 관찰. 완료 후 `model_best` .pt 다운로드.

### 8.4 Predict-then-verify (재학습 후 확인할 기대치 — 미리 기록)
기준선(재학습 전, 현행 배포 모델 + 신 w, `reports/tritan_retrain_eval/eval_before.json`):
blue_sea blueΔE **1.75**, traffic 1.02, tennis 10.69(포화 파랑은 이미 보정됨), skin 1.74.

| 항목 | 현행(before) | 재학습 후 예측 | 근거 |
|---|---|---|---|
| blue_sea t **w mean**(게이팅) | 0.117(구)→**0.663**(신, 이미 적용) | 0.663 유지 | confusion.py 확정 |
| blue_sea **저채도 파랑 보정** ΔE00 | 1.75 (구 모델 no-op) | **↑ ~5–10** | 손실기여 ×4(작업3), p/d 포화보정 6–8 유추 |
| traffic 하늘 파랑 ΔE00 | 1.02 | ↑ ~3–6 | w 0.06→0.31, 신호 ×2.8 |
| tennis(포화 파랑) ΔE00 | 10.69 (이미 보정) | ≈ 유지~소폭↑ | 포화 파랑은 구 w도 w=1 |
| gate **ratio_w T ≥ 1.27** | held-out 1.274(직전) | **PASS(유지~개선)** | 파랑 회복↑ → T 대비 개선 방향 |
| p/d ratio_w (뱅크 1.39/1.36, gate 1.10/1.13) | — | **gate 유지**, |Δ|≲0.02 | p/d 신호 불변, 공유망 미세 이동 |

> **판정:** 저채도 파랑 ΔE00가 유의 상승 + T gate 유지 + p/d gate 유지면 성공. p/d가 gate
> 아래로 떨어지면(공유망 망각) 회귀 — 보고 후 재검토(예: p/d 보호용 fine-tune/replay).

---

## 9. 재학습 후 평가 원커맨드 (작업 3) — 스크립트 준비 완료

새 `model_best.pt`를 받은 뒤 순서대로:

1. **gate 재평가**: (train.py::validate 로직) 새 체크포인트로 10장 뱅크 + held-out 게이트.
   `py -m cvdlens_v2.heldout_check`(경로/체크포인트 인자 확인) 또는 학습 로그의 마지막 val 블록.
2. **신규 ONNX export**: `py -m cvdlens_v2.export_onnx`(새 .pt 지정) → `cvdlens_{p,d,t}.onnx` 3개.
   **주의:** wrap_for_onnx가 t-분기를 포함하므로(§8.0) 새 t ONNX는 Brettel-w로 게이팅.
3. **parity check**: `py -m cvdlens_v2.parity_check`(또는 test_model 스모크) — PyTorch↔ONNX <1e-3.
4. **w맵+보정 몽타주 + 지표 (before/after 비교)**:
   `py -m cvdlens_v2.post_retrain_eval --model-dir <새 onnx 폴더> --tag after`
   → `reports/tritan_retrain_eval/` 에 blue_sea/tennis/traffic/skin 몽타주 + `eval_after.json`.
   **before(현행)**: `eval_before.json` 이미 생성됨 — after와 직접 대조.
5. **기존 daily_test 전체 지표**: 새 ONNX를 `cvd-lens/inference/model/`에 임시 배치 후
   로컬 추론 기반 지표(reeval_guided 등) 또는 daily_test(배포 교체 후). **ONNX 교체·배포는
   평가 결과 확인 후 별도 결정 — 이번엔 준비까지만.**

> post_retrain_eval는 지금(before) 실행해 기준선을 남겼고, 동일 스크립트를 새 ONNX로 재실행하면
> after가 나온다. ΔE00/w_t/CRR 정의는 기존 스크립트(daily_test, artifact_probe) 그대로.
