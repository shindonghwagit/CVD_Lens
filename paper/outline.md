# CVDLens 논문/보고서 골격 (outline)

> 상태 범례 — 각 소절에 표기:
> **[재조립]** 기존 문서/산출물을 편집·인용하면 되는 부분
> **[신규]** 새로 써야 하는 부분 (분석·서술)
> **[자료]** 인용할 기존 산출물 파일 경로

분량 목표는 학교 양식 확인 후 확정 (문서 하단 "미해결 질문" 참조).

---

## 1. 서론
- **[신규]** 문제 정의: 색각이상(CVD) 사용자가 색 대비를 잃는 상황, 재색상화의 목표.
- **[신규]** 기존 방법(daltonize류)의 한계: 고정 규칙, 자연스러움 손상, 저대비 영역
  불필요 변형. → Step 3 수치로 뒷받침(전방 참조).
- **[재조립]** 기여 요약: 학습 기반 재색상화 + 가시 부분공간 파라미터화 + w-gating +
  daltonize 대비 정량 우위(CRR ≥, NP ↓).

## 2. 관련 연구
- **[신규]** CVD 시뮬레이션 모델(평가·게이팅의 기반): Brettel-1997(error-shift 투영),
  Machado-2009(matrix). 본 연구는 이들을 **손실·혼동가중 w의 내부 모델**로 사용.
  - **[자료]** 구현 근거: `cvdlens_v2/simulation.py`, `cvdlens_v2/confusion.py`
- **[신규]** 재색상화 선행연구를 4계열로 정리하고 CVDLens의 좌표를 명시:
  1. **고전 daltonize (규칙 기반).** *Rathee & Mann 2022* — RGB→LMS 변환 후 결손 원뿔 정보
     삭제→역변환(Brettel) + 대비/블러/K-means 등 IPT. 논문 제목의 "CNN"은 **Ishihara 판 숫자
     인식(MNIST) = 자가진단 모듈**일 뿐, 보정 엔진은 비학습 daltonize. → 선택성·자연스러움 부재.
  2. **CNN이 손규칙을 모방 학습.** *Pendhari et al. 2024* — 타입별 CNN 오토인코더 3개가
     "적색→갈색 / 청색→보라 / (deutan)대비감소" 손규칙 출력을 페어 학습으로 모방. 규칙 자체가
     crude(지각·선택성 없음). Tkinter GUI + Ishihara. → 본 프로젝트 착안의 출발점이나 baseline.
  3. **학습 변환 + 지각모델 임베드(직계 선행).** *Orii et al.* — 다층 신경망 3블록:
     ①색변환 층(**학습**) + ②색맹 지각모델(고정) + ③색 구분모델(고정). sim·discrimination을
     고정해두고 **변환규칙만 학습**. **CVDLens 접근의 원형**(학습 변환 + 고정 CVD-sim을 손실에).
     CVDLens 확장: bilateral-grid 고해상, w-gating 선택성, daltonize 대비 CRR/NP, 웹 배포.
  4. **선택적·최적 보정(선택성 직계).** *Choi et al. 2019 (IEEE Access, 세종대)* — 기존법이
     **모든 색을 보정**해 정상시야 이질감을 유발한다고 비판하고, **혼동선(confusion-line) DB +
     region growing + 최소영역 보정 + 재보정 방지(collision 회피)**로 **혼동영역만 최소 변환**.
     **CVDLens의 w-gating이 이 선택성의 학습·미분가능 버전**(혼동가중 w=Brettel-sim ΔE 기반).
     단 Choi의 **이미지-전역 재보정 방지**는 CVDLens에 없는 요소(§7 향후과제로 흡수).
  5. **Ishihara 특화 IPT + 분류 평가.** *Akalın & Top 2025* — 3단계 IPT 필터를 Ishihara 판에
     적용, 색맹 시뮬 후 **MobileNetv2**로 분류해 객관 평가. Ishihara 국한 + 분류기 평가.
     CVDLens는 일반 이미지 + CRR/NP·daltonize 비교(더 일반적·지각적).
- **[신규] CVDLens의 위치 = Orii(학습 변환) × Choi(선택성)의 교차점을 현대화·배포.** 결손 축의
  성질에 맞춰 **축별 최적 방법**을 채택: **적록(p/d)=학습 지각 대비회복**(Orii 계열, 적록차를
  가시 청황/명도 축으로 매핑), **청황(t)=채도보존 해석적 hue 회전**(Choi/daltonize 계열, off-axis
  이동). analytic t는 최상위 저널(Choi)도 쓰는 정통 방식임을 근거로 정당화.
- **[신규]** HDRNet / bilateral grid 계열(저해상 계수 예측 → 고해상 적용) — p/d 아키텍처 착안점.
  - **[자료]** 논문 PDF: `논문 리뷰/` (Orii, Choi 2019, Akalın 2025, Pendhari 2024, Rathee 2022).

## 3. 방법
- **[재조립]** v1 실패 분석 = identity collapse의 수학적 원인:
  대칭 loss(L1-to-identity, SSIM, per-channel gradient matching)에서 identity가
  전역 최소 → |Δ|≈0.007–0.009, 보정 없음.
  - **[자료]** `outputs/v2_phase0/phase0_final_report.md` (Step 1~4 서사)
- **[재조립]** 가시 부분공간 파라미터화 + w-gating: 혼동 가중 w를 그래프 내부에서 계산,
  저혼동 영역 보호.
  - **[자료]** `cvdlens_v2/basis.py`, `cvdlens_v2/confusion.py`, `cvdlens_v2/model.py`
- **[재조립]** 아키텍처: 저해상 field(64) 예측 → 적용, 타입별 그래프.
  - **[자료]** `cvdlens_v2/model.py`, ONNX wrap 노트 `outputs/v2_phase2/RUNBOOK_phase2.md` Step 1
- **[재조립]** loss 설계 (one-sided L_c, TV, excess penalty 실패한 개입 포함):
  Phase 0 검증 과정 요약. Step 4(excess penalty)는 **음성 결과**로 명시.
  - **[자료]** `outputs/v2_phase0/phase0_final_report.md`, `cvdlens_v2/losses.py`,
    `cvdlens_v2/validate_loss.py`, ray/narrow-valley 스캔
    (`outputs/ray_scan/`, `outputs/narrow_valley_stdout.txt`)

## 4. 실험
- **[재조립]** 학습 설정: λ_tv=0.03, λ_excess=0, lr, field_size=64, 배치 등.
  - **[자료]** `outputs/v2_phase1/RUNBOOK.md`, `cvdlens_v2/train.py`
- **[신규]** 수렴 곡선 해설 (ratio_w P/D/T가 1.0에서 상승).
  - **[자료]** `outputs/v2_phase1/history.json` (+ `cvdlens_v2/plot_convergence.py`로 그림 생성 필요 → **[신규] 그림 산출**)
- **[재조립]** best checkpoint 선정 = `model_best` (step 9000).
  - **[자료]** `cvdlens_v2/select_best.py`, `outputs/v2_phase2/RUNBOOK_phase2.md`
- **[재조립]** held-out 검증 (8 이미지, eval set과 disjoint).
  - **[자료]** `cvdlens_v2/heldout_check.py`

## 5. 평가
- **[재조립]** CRR-NP 프레임 정의 (2축: 회복↑ / 손상↓).
  - **[자료]** `outputs/v2_phase3/step3_report.md` §Metrics, `cvdlens_v2/step3_metrics.py`
- **[재조립]** daltonize 비교 (Step 3): per-type / per-tier / Wilcoxon, verdict.
  - **[자료]** `outputs/v2_phase3/step3_report.md`, `step3_scatter.png`, `eval_results.json`,
    `eval_set.json`, `cvdlens_v2/step3_eval.py` / `step3_eval_set.py`
- **[재조립]** 케이스 스터디 (win / tie / lose 각 1).
  - **[자료]** `outputs/v2_phase3/step3_case_{bigwin,similar,lose}.png`
- **[재조립]** 이시하라 정성 데모 (판독: **보정 후에도 숫자 29는 판독 불가**,
  단 blue-yellow 축에 대비가 추가됨 — 정직하게 서술).
  - **[자료]** `outputs/v2_phase2/ui_ishihara_sim_beforeafter.png`

## 6. 시스템
- **[신규]** 웹 배포 아키텍처: 이중 경로.
  - 브라우저 경로: ONNX Runtime Web(WebGL/WASM), 타입별 정적 그래프.
    - **[자료]** `outputs/v2_phase2/RUNBOOK_phase2.md`, `cvd-lens/app/**`, `cvd-lens/lib/cvdSim.ts`
  - 서버 경로: FastAPI 추론 서버(Render).
    - **[자료]** `cvd-lens/inference/main.py`, `cvd-lens/inference/Dockerfile`
- **[재조립]** PyTorch→ONNX 패리티 (max|diff|≈2.0e-05).
  - **[자료]** `outputs/v2_phase2/parity_report.json`
- **[신규]** 해상도 파이프라인 (저해상 계수 → 고해상 적용) 서술.

## 7. 한계 및 향후 연구
- **[재조립]** 시뮬레이터 기준 평가의 한계 (Brettel/Machado 기준, 실사용자 지각 미검증).
  - **[자료]** `outputs/v2_phase3/step3_report.md` §Limitations (이번에 추가한 문단)
- **[신규]** T(트리타노피아) 마진, 단일 severity 조건.
- **[신규] tritan 파랑 회복 ↔ HSV 채도 저하의 본질적 결합 — 3중 음성 실험(핵심 한계).**
  저채도 파랑의 tritan 보정은 opponent 채널(R) 추가(→보라)로 달성되며, 이것이 곧 HSV 채도 저하다.
  세 경로 모두 (blueΔE≥5 & satΔ≥−0.08 & gate 통과 & p/d 회귀 없음)를 **동시 충족 못 함**을 실험으로
  확정 → 결합은 손실·학습 스케줄 수준에서 분리 불가.
  - **(i) 사후조정 불가:** L_sat 없는 fresh 체크포인트(8k~20k)는 단조 오목 프론티어를 이루며
    수용영역을 관통하지 않음(운영점 부재). *→ figure: Pareto 2장 `pareto_blueDE_vs_sat.png`·
    `pareto_blueDE_vs_skin.png`; 표: DECISION_TABLE 스캔표.*
  - **(ii) from-scratch 손실 제약 불가:** λ_sat=200 fresh 재학습은 satΔ는 개선(+0.004)하나
    **blueΔE가 2.73으로 붕괴**(프론티어 최저-blue점) + p/d 회귀 + t-gate 미달. *→ figure:
    오버레이 v3점; 표: DECISION_TABLE v3표.*
  - **(iii) 커리큘럼(2단계) 불안정:** step20000→L_sat 파인튜닝은 blue+sat 수용영역에 **진입은
    하나**(step2000 blueΔE 6.31·satΔ −0.011) **t-gate 1.26<1.27 고착 + p/d 회귀**로 4중 기준
    미충족, 궤적 비단조(조기종료 취약). *→ figure: `pareto_2stage_overlay.png`(궤적 화살표);
    표: DECISION_TABLE 궤적표.*
  - **[자료]** `reports/tritan_retrain_eval/DECISION_TABLE.md`(스캔·v3·2단계·최종결론),
    `ckpt_scan.json`, `stage2/stage2_trajectory.json`, `pareto_*.png`,
    `cvdlens_v2/losses.py::saturation_loss`(음성 개입, Exp 1-D excess penalty와 동형).
  - **[신규] 향후 과제 1문단:** 세 음성은 **문제가 손실 항이 아니라 보정 파라미터화에 있음**을
    시사한다. tritan 보정을 sRGB/opponent 공간의 자유 delta로 두면 파랑 이동이 필연적으로 채도를
    깎는다. 따라서 **지각균등 색공간(예: CAM02-UCS)에서 명도·채도를 고정하고 hue(또는 그에 준하는
    지각 색상각)만 회전**시키는 **hue-preserving 재파라미터화**가 필요하다 — 즉 개선은 손실 수준이
    아니라 **보정 파라미터화 수준의 재설계**가 요구됨을 본 실험들이 정량적으로 시사한다. (별도 연구
    사이클: 색공간 변환의 미분가능 구현 + ONNX 이식 비용 검토.)
  - **[신규] 채택된 해법(본 연구 결론):** 위 3중 음성을 근거로 **tritan을 학습모델에서 분리해
    채도보존 해석적 hue 회전**(blue→violet / yellow→yellow-green, S·V 고정)으로 전환·배포.
    실사 test-set(21장) 사전고정 지표에서 커버리지/구분(CRR)/선택성/무-물빠짐을 통과(§실험).
    → tritan 한계는 "손실로 못 품(음성)" + "파라미터화 전환으로 해결(양성)"의 **쌍**으로 서술.
    - **[자료]** `cvdlens_v2/tritan_hue_method.py`, `reports/tritan_gate_eval/`(manifest·scores·CRITERIA),
      `cvd-lens/inference/main.py`(`_tritan_hue_shift`).
- **[신규] 이미지-전역 관계적 보정(Choi 2019 흡수) — p/d·t 공통 향후과제.**
  현행 CVDLens 보정은 **per-pixel/국소**(혼동가중 w = "이 색이 추상적으로 혼동색인가")라, 이미지
  안의 **다른 영역과의 관계**(A를 옮기면 기존 B와 충돌하는가)는 안 본다. Choi의 **혼동선 DB +
  최소영역 보정 + 재보정 방지**를 관계적 항으로 흡수하면 양 축을 보완:
  - **p/d 보완 — #1 대면적 포화-빨강 과보정 완화.** 현행 w는 **모든 빨강에 w=1**(혼동 파트너 유무
    무관)이라 홀로 있는 대면적 빨강(토마토)을 과보정(protan p99 41.55). Choi식 **관계적 게이팅**
    (그 빨강이 이미지 내 **다른 영역과 실제로 혼동될 때만** 보정, 더 작은 영역 우선)을 도입하면
    과보정·얼룩(#1)이 구조적으로 줄어든다. → w를 "절대 혼동색"에서 "이미지 내 상대 혼동쌍"으로 확장.
    - **[자료]** `outputs/artifact_analysis/REPORT.md`(#1), `outputs/daily_test/daily_stats.json`.
  - **t 보완 — 고정 회전의 collision 회피 + 이미지-적응 목표색.** 현행 hue 회전은 **고정각**이라
    파랑을 옮긴 보라가 이미지에 **이미 있는 보라/자홍과 충돌**할 수 있다(Choi가 지적한 재보정 문제).
    Choi의 **재보정 방지 + 최적색 선정**(정상시야 최소변화 & CVD 최대구분)을 적용하면, 이미지의
    기존 색 분포를 보고 **충돌 없는 최적 회전량/목표색**을 정하는 이미지-적응 tritan으로 발전.
    - **[자료]** `cvdlens_v2/tritan_hue_method.py`(현행 고정각), Choi 2019 §II(혼동선 DB·재보정 방지).
  - **통합 난점(정직 서술):** Choi는 **region-growing + 이산 DB**(하드 경계·비미분)라 CVDLens의
    soft/고해상/학습 프레임에 그대로 못 붙임. 관계적 혼동 항을 **미분가능·per-pixel 근사**로
    재설계하거나(학습 손실에 상대-혼동 페널티), 추론단 후처리(영역분석)로 흡수하는 두 경로가 있음.
- **[신규] 피부 warm-shift = 선택성 원칙과의 트레이드오프(어느 갈래든 한계 서술).**
  피부톤은 tritan 혼동축(파랑-노랑) 상 노란끼에 위치 → w 게이팅이 피부를 confusion으로 잡는 것은
  **색채학적으로 올바르며**, 스킨톤 보호는 "혼동영역을 보정한다"는 선택성 원칙과 본질적으로
  상충. 재학습 skinΔE~2.6–3.1(전 체크포인트 2.5 초과, blue 강도와 무관한 상존 밴드). 해결은
  손실 게이팅이 아니라 **지각 민감 색역 보호항**(위 L_n 확장 항목과 연결) 방향.
  - **[자료]** `reports/tritan_retrain_eval/ckpt_scan.json`(skinΔE), `sweep_skin_portrait_step20000.png`,
    `outputs/v2_phase3/skin_analysis.json`
- **[신규]** 실사용자 심리물리 검증 부재 → 향후.
- **[신규]** 피부 인지 자연스러움 항: 피부 등 지각 민감 색역에 대한 민감도 가중을 L_n에
  반영하는 방향 (예: 얼굴/피부 마스크 기반 자연스러움 가중, 지각 민감도 가중 L_n 확장) —
  severity의 뭉툭한 전역 감쇠 대신 회복을 지키며 피부만 선택적으로 보존.
  - **[자료]** `outputs/v2_phase3/skin_analysis.json` (피부 w≈0.73 = 비피부 1.7배, severity
    트레이드오프 수치)
- **[신규]** 실시간 스트림(비디오/카메라) 성능.

---

## 신규 vs 재조립 요약
- **재조립 비중 높음**: §3 방법, §4 실험, §5 평가 (Phase 0/1/2/3 문서가 이미 서사 형태).
- **신규 서술 필요**: §1 서론, §2 관련 연구, §6 시스템 아키텍처 산문, §7 향후 연구,
  그리고 **수렴 곡선 그림 1장** (history.json → plot_convergence.py 실행).

## 확정 사항 / 미해결 질문
확정 (2026-08-04):
- 산출물: **국문 졸업논문** (작품 보고서 아님, 국문 서술).

미해결 (지도교수/학과 확인 후 알려주면 반영):
1. 지정 **논문 양식**(템플릿, 폰트, 인용 스타일)이 있는가? → 확인 전까지 표준 구성으로 진행.
2. **분량 요구**(페이지 수 / 장 구성 제약)는? → 확인 전까지 위 7장 구성을 기본안으로 둠.
