# CVDLens 추론 파이프라인 개선 + 진단 (종합 리포트)

2026-08-18. 몽타주 평가에서 확인된 3가지 문제에 대한 후처리 개선(guided filter)과 원인
진단. **ONNX 모델은 미변경 — 후처리만.** 모든 지표는 기존 스크립트 정의 그대로.

산출물: `reports/crr_diag/`(작업2), `reports/dilution/`(작업3),
`reports/guided_reeval/`(작업4, 몽타주 30 + REPORT.md + reeval.json).

---

## 작업 1 — guided filter 후처리 (구현)

업샘플된 delta를 **원본 이미지를 guide로 하는 color guided filter**(He et al. 2010/2013)로
후처리한 뒤 원본에 더한다. 원본 에지에 delta를 스냅(경계 선명화)하고 균일-guide 영역에서
delta를 평탄화(저주파 gradient 제거)한다.

- **구현:** `cvd-lens/inference/guided.py` — `cv2.boxFilter`(적분영상, O(1)) 기반, 3×3
  guide 공분산 폐형 역행렬. **opencv-contrib 의존성 없음.**
- **적용 지점:** `main.py::_correct_image` 업샘플 직후·더하기 전. FastAPI 추론 경로와
  로컬 평가(`cvdlens_v2/infer_local.py`)가 **동일 파일을 import**해 동일 필터 적용.
- **config:** `cvd-lens/inference/config.py` (env 오버라이드). 하드코딩 없음.
  - `CVDLENS_GUIDED_FILTER` (기본 on) — on/off 플래그
  - `CVDLENS_GUIDED_RADIUS_DIVISOR` (기본 20) → radius = max(H,W)//divisor
  - `CVDLENS_GUIDED_EPS` (기본 1e-3)

> ⚠️ 서버 코드 변경은 **재배포 시** 적용된다. 아직 push 안 함 — 배포는 사용자 결정.
> 아래 평가는 로컬 ONNX로 on/off를 통제해 산출.

---

## 작업 2 — CRR<1 원인 진단  → **magnitude_deficit (direction_error 아님)**

대상: food_tomatoes(d), traffic_street(p/d) @ s1.0. 판정 = delta 스케일 스윕(α배 delta의
채도-마스크 CRR).

| 케이스 | CRR α=0.5/1/1.5/2 | 추세 | 판정 |
|---|---|---|---|
| food_tomatoes_d | 1.012 / 1.058 / 1.101 / 1.127 | 단조 ↑ | **magnitude_deficit** |
| traffic_street_p | 1.004 / 1.010 / 1.016 / 1.021 | 단조 ↑ | **magnitude_deficit** |
| traffic_street_d | 1.003 / 1.009 / 1.013 / 1.018 | 단조 ↑ | **magnitude_deficit** |

- **더 보정할수록 CRR이 올라간다 → 보정 방향은 정확, 세기가 부족**하다. direction_error
  (부호 반대)가 아니다. 따라서 사용자 규칙(“direction_error면 수정 금지, 리포트만”)의
  수정 금지 대상이 아니며, 후처리로 완화 가능한 성질이다.
- **혼동축 성분 `<δ_lin, n_t>` ≈ 1e-1~1e-3 수준의 |·| = 사실상 0**(구조적: delta는 가시
  평면에 있음, 3장 단위시험 b). → “혼동축 부호”는 진단 정보가 못 됨을 수치로 확인, 대신
  가시 색축 `b_C` 부호 분포를 보조 출력.
- **배포에서 <1로 관측된 진짜 이유 = magnitude_deficit + JPEG q92 재압축.** raw CRR은
  1.00~1.04인데 API 응답 JPEG q92가 얇은 회복 마진을 **0.023~0.051 깎아** 1 아래로 떨어뜨린다
  (food_d 1.041→0.990, traffic_p/d 1.004→0.981). 배포 몽타주의 0.978~0.986과 일치.

근거: `reports/crr_diag/crr_diag.json`, `*_crrdiag.png`.

---

## 작업 3 — 작은 영역 delta 희석  → **가설 확정 (grid 공간 평균화)**

traffic_street 채도-빨강 마스크를 connected components(103개)로 분리, 영역 크기 vs 평균 |delta|.

- **corr(log area, 평균|δ|) = +0.32 (p), +0.32 (d)** → 큰 영역일수록 delta가 크고, 작은
  영역일수록 delta가 작다. **16×16 grid 셀 내 공간 평균화로 작은 빨강(신호등)이 희석되는
  가설 확정.**
- **guided filter는 작은 영역 magnitude를 살리지 못한다** — 작은영역 평균|δ| 0.0228→0.0174
  (**−23%**). guided는 평활자라 존재하지 않는 크기를 만들 수 없다. 즉 **작업3의 문제(작은
  영역 희석)는 guided로 해결되지 않는다** — grid 해상도 상향 또는 에지 인지 업샘플이 필요한
  별도 과제.

근거: `reports/dilution/dilution.json`, `*_dilution.png`.

---

## 작업 4 — before/after 재평가  → **전 케이스 CRR↑ · ΔE↓ · NP↓**

로컬 ONNX, guided on/off, 5이미지 × p/d/t × s0.6/s1.0 = 30케이스. 전체 표·몽타주는
`reports/guided_reeval/`.

**요약(guided OFF → ON, 방향):** 30/30 케이스에서 CRR 상승, ΔE_mean 하락, NP|Δ| 하락.
guided가 저주파 gradient·에지 헤일로(ΔE만 부풀리고 회복엔 기여 안 하던 성분)를 제거하고
delta를 진짜 에지에 집중시켜, **더 적은 변화(ΔE·NP↓)로 더 큰 혼동대비 회복(CRR↑)** 을 낸다.

대표값(s1.0): food_d CRR 1.041→1.205 (ΔE 5.86→4.83), stop_d 1.122→1.172, skin_d 1.014→1.059.

### 사용자 지정 특별 점검 (s1.0)

| 점검 | 결과 | 판정 |
|---|---|---|
| (a) stop 빨간 링 ΔE 균일도(CoV↓) | p 0.62→0.44, d 0.56→0.39, t 0.57→0.42 | ✅ **균일해짐**(문제3 해결) |
| (b) traffic 신호등(작은빨강) ΔE 상승? | p 2.57→1.93, d 2.09→1.51 (↓) | ❌ **상승 안 함**(문제2 미해결, 작업3과 일치) |
| (c) skin_portrait NP\|Δ\| 악화? | p .0062→.0056, d .0076→.0066 (↓) | ✅ **악화 없음**(오히려 개선) |
| (d) CRR<1 회복(raw / jpeg) | food_d 1.041→1.205 / 0.99→**1.14**; traffic_p 1.004→1.02 / 0.981→0.994; traffic_d 1.004→1.021 / 0.981→**0.998** | ◑ food_d 완전 회복, traffic p/d 개선하나 JPEG 후 ~0.99 잔존 |

근거: `reports/guided_reeval/REPORT.md`, `reeval.json`, 몽타주 30장.

---

## 결론 및 권고

1. **guided filter 후처리는 순개선** — 전 30케이스에서 CRR↑·ΔE↓·NP↓ 동시. 특히 문제3
   (균일 채도 영역 delta 저주파 불균일)을 해결(stop 링 CoV 0.62→0.44). **기본 ON으로
   배포 권고.**
   - **해석:** guided ON에서 ΔE_mean 감소는 보정량 축소가 아니라 halo/과보정 정리에
     해당함 — 동일 조건에서 CRR은 전 케이스 상승.
2. **문제2(작은 영역 희석)는 guided로 해결 안 됨** — grid 공간 평균화(corr +0.32)가 근본
   원인이라 후처리로 magnitude를 복원 못 한다(작은빨강 ΔE 오히려 ↓). 신호등 등 작은 채도
   객체 보정은 **grid 해상도 상향 / 에지 인지 업샘플**이라는 별도 작업이 필요 —
   ONNX 재학습·재수출 수반이라 별도 결정 사안.
3. **CRR<1은 direction_error 아님(magnitude_deficit)** — 수정 금지 대상 아님. guided가
   food류(큰 채도 영역)는 JPEG 후에도 >1로 회복시키나, traffic(작은 신호등)은 dilution
   한계로 JPEG 후 ~0.99 잔존. 이는 (2)의 grid 해상도 과제와 동일 뿌리.
4. **부수 관찰:** API 응답 JPEG q92가 회복 마진을 0.02~0.05 깎는다. 무손실/고품질(q98+)
   또는 PNG 응답이 CRR<1 경계 케이스에 도움 — 저비용 옵션으로 검토 가치.

### 다음 결정 필요 / 후속 결과 (2026-08-18 갱신)
- (A) guided filter 배포 반영 — **진행 중**(기본 ON, config `CVDLENS_GUIDED_FILTER`).
- (B) 작은 영역 희석(문제2) — **타일 추론 ablation 완료**(`reports/tiled_inference/`):
  N×N 타일링이 작은빨강 ΔE를 p +81%/d +74% 상승시켜 dilution 완화 → **grid 셀 점유율이
  근본 원인임을 입증**. 부작용 미미하나 ~N² 추론 비용. 배포 반영은 별도 결정(비용/효과).
  근본 해법(재학습으로 grid 해상도 상향)은 여전히 유효.
- (C) API 응답 JPEG 품질 — **해결**(`reports/jpeg_sweep/`): q92→**q94**로 상향하면 경계
  3케이스(traffic p/d, food d) 모두 CRR≥1.0 회복(q92 대비 응답 +~21%). config
  `RESPONSE_JPEG_QUALITY=94` 기본값 반영, guided와 같은 배포에 포함.
