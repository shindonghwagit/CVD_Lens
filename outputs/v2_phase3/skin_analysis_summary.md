# Skin-tone shift diagnosis — summary

- Checkpoint: `outputs\phase3_kaggle\v2_phase1\model_best.pt`
- Selected (skin coverage, seed 20260811): 000000423506, 000000513567, 000000546475, 000000441491, 000000038678
- Skin mask: YCbCr 77<=Cb<=127, 133<=Cr<=173, 1x erosion
- Representative severity-sweep image: `000000423506` (type P)

## (a) Confusion weight w — skin vs non-skin (severity 1.0)

| type | mean w (skin) | mean w (non-skin) |
|---|---|---|
| P | 0.7320 | 0.4262 |
| D | 0.7345 | 0.4269 |

## (b) Skin |Δ| (mean sRGB change in skin) vs severity

| type | sev 1.0 | sev 0.7 | sev 0.5 |
|---|---|---|---|
| P | 0.0223 | 0.0207 | 0.0187 |
| D | 0.0234 | 0.0219 | 0.0202 |

## (c) Whole-image CRR (ratio_w) vs severity — recovery cost

| type | sev 1.0 | sev 0.7 | sev 0.5 |
|---|---|---|---|
| P | 1.2230 | 1.1931 | 1.1671 |
| D | 1.1959 | 1.1798 | 1.1637 |

> 값은 선정 5장 평균. per-image 원자료는 skin_analysis.json.

## 진단 후 선택지 (구현 금지 — 문서만)

_구현 금지. 아래 수치는 aggregates 기반. 결정은 사용자._

### A_default_0.7_only
- severity 0.7 기본값만으로 충분 → 종결.
  - evidence: aggregates.<type>.skin_delta['1.0'] vs ['0.7'] 감소폭, 그 대가는 crr_ratiow['1.0'] vs ['0.7'] 하락폭으로 판단.

### B_inference_skin_attenuation — **기각 (REJECTED, 2026-08-11)**
- (원안) 추론단 피부 감쇠 옵션: delta에 피부 마스크 감쇠(예: 1-α·skin_mask)를 곱해 피부 이동만 줄인다. 기본 OFF, 논문 평가(severity 1.0, 감쇠 없음)와 분리 명시.
- **기각 사유:**
  1. **아티팩트가 더 나쁨:** YCbCr 마스크는 배경/음식(나무·모래·피자 등 피부색)을 오탐한다(본 진단 몽타주에서 확인). 마스크 경계에서 국소 저보정 패치가 생기면, 균일한 경미 이동보다 시각적으로 더 거슬린다.
  2. **딜레마:** 기본 OFF면 실사용 실익이 없고, ON이면 배포 출력이 평가 조건(감쇠 없음)과 어긋나 배포-평가 괴리가 생긴다.
  3. **비용 대비 효과:** ONNX 재export·그래프 분기 비용 대비, severity 0.7이 이미 확보하는 완화(피부 -6~7%) 위 추가 이득이 미미하다.
- **결정:** severity 0.7 기본값(A)으로 완화를 흡수하고, 잔여 이슈는 논문 향후 연구(C)로 넘긴다.

### C_paper_future_work_only — **채택 (ADOPTED)**
- 논문 §7 향후 연구로 서술(피부 인지 자연스러움 항: 지각 민감도 가중 L_n 확장). 코드 변경 없음.
  - evidence: 본 skin_analysis 수치를 근거 문단으로 인용(ch5 §5.5, outline §7).

---

## 결정 (2026-08-11)

**A(기본 severity 0.7, 적용 완료) + C(논문 서술). B 기각(위 사유).** 스레드 종결.
