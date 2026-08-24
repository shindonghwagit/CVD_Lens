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
- **[신규]** 재색상화 기법 개괄 (rule-based daltonize, 최적화 기반, 학습 기반).
- **[신규]** CVD 시뮬레이션 모델: Brettel(error-shift), Machado(matrix).
  - **[자료]** 구현 근거: `cvdlens_v2/simulation.py` (machado_matrix_tensor, daltonize)
- **[신규]** HDRNet / bilateral grid 계열 (저해상 계수 예측 → 고해상 적용) — 아키텍처 착안점.

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
- **[신규] tritan 파랑 회복 ↔ 채도(HSV) 결합** — 두 갈래 준비(v3 결과 따라 택1):
  - **(a) v3(L_sat) 성공 시 → ablation figure.** Pareto 2장(`reports/tritan_retrain_eval/pareto_*.png`)
    을 **"채도 보존항 없는 학습의 실패 모드"** ablation으로 배치: L_sat 미적용 체크포인트(8k~20k)는
    수용영역(blueΔE≥5 & satΔ≥−0.08)에 진입 불가(운영점 부재), L_sat 추가 시 진입(스모크
    satΔ −0.165→+0.01@blueΔE 5.09). "결합은 손실 설계로 분리 가능"을 정량 기여로 서술.
    - **[자료]** `reports/tritan_retrain_eval/DECISION_TABLE.md`(스캔·L_sat·스모크), `ckpt_scan.json`,
      `losses.py::saturation_loss`, v3 `eval_after.json`(예정)
  - **(b) v3 미달 시 → 한계.** "tritan 저채도 파랑 회복은 R-추가(→보라)로 달성되며 HSV 채도
    저하와 단일 메커니즘으로 결합, 사후 조정(severity·체크포인트 선택)으로 분리 불가"를 정량
    한계로 서술 + 채도 보존을 향후 과제로.
    - **[자료]** `DECISION_TABLE.md` "운영점 부재" + Pareto(단조 오목 프론티어), `ckpt_scan.json`
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
