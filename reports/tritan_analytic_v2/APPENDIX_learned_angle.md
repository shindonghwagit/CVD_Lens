# [부록] 학습-각도 HSV 회전 — 1회 스모크 (타임박스 종료)

- 2026-08-27. `cvdlens_v2/learned_tritan_hsv.py` (theta-net 구조·손실·하이퍼파라미터·스텝·seed 동일,
  회전만 YCbCr→미분가능 HSV). 800스텝 1회. 데이터 `hsv_scores.json`, 모델 `theta_net_hsv.pt`.
- 결론(analytic 채택)에 영향 없음. PASS해도 배포 analytic 유지.

## 채점 (원 기준, test-set 21장 전체, 완화·신규 지표 없음)
| 항목 | 실측 | 판정 |
|---|---|---|
| satΔ(blue_sea) ≥ −0.08 | **+0.0000** | PASS |
| blueΔE(blue_sea) ≥ 5 | 29.42 | PASS |
| red_ctrl dE_mask ≤ 0.1 | 0.073 | PASS |
| green_ctrl dE_mask ≤ 0.1 | **0.154** | **FAIL** |
| gray_ctrl dE_mask ≤ 0.1 | 0.009 | PASS |
| p/d 회귀 | 미통합, p/d 불변 | 없음 |

## 핵심 확인 대상 2개 (분리 보고)
1. **satΔ가 HSV 전환으로 해결됐는가 → YES(완전).** 21장 전체 max|satΔ| = 0.0000. HSV2RGB는 항상
   in-gamut(L_gamut=0 학습 내내), HSV 회전은 S 정의상 보존. YCbCr theta-net satΔ −0.13과 대조 →
   **satΔ FAIL의 원인이 색공간(YCbCr)이었음이 확정.**
2. **선택성이 여전히 FAIL인가 → 부분(green만).** red 0.073·gray 0.009 PASS, **green_ctrl 0.154 FAIL**.
   ※ 사전 예상("선택성은 색공간 무관·미해결")과 달리 **red는 0.9→0.07로 개선**됨(gamut clip 제거 효과).
   green만 리터럴 임계(0.1) 소폭 초과.

## 이분법 최종 판정: **FAIL** (green_ctrl 0.154 > 0.1)

## 마감
- 타임박스 종료. **추가 변형·재시도·λ 스윕·재학습 없음.**
- 학습-각도 HSV는 satΔ를 구조적으로 해결하나(신규성은 analytic과 동일한 HSV 회전 원리) green 선택성
  리터럴 미달로 원 기준 FAIL → **배포/논문 주 채택 = analytic HSV 회전(선행 커밋 7e389fe, PASS) 확정.**
- 학습-각도 방식은 "가능하나 analytic 대비 신규성 낮고 원기준 미달"로 논문 부록 처리.
