# API 응답 JPEG quality 스윕 (작업 2)

guided ON, sev 1.0. CRR은 기존 daily_test.crr 정의(sim severity 1.0). 각 quality로 원해상도 인코딩→디코딩 후 CRR 측정. raw=무압축 상한.

## quality vs CRR (경계 케이스 3개, ≥1.0 굵게 판정)

| fmt/quality | traffic_street_p | traffic_street_d | food_tomatoes_d |
|---|---|---|---|
| raw | **1.02** | **1.021** | **1.205** |
| PNG | **1.021** | **1.023** | **1.211** |
| JPEG q92 | 0.994 | 0.998 | **1.14** |
| JPEG q94 | **1.022** | **1.025** | **1.137** |
| JPEG q95 | **1.003** | **1.006** | **1.135** |
| JPEG q96 | **1.018** | **1.021** | **1.132** |
| JPEG q98 | **1.014** | **1.017** | **1.125** |

## quality vs 응답 파일크기 (bytes, 원해상도)

| fmt/quality | traffic_street_p | traffic_street_d | food_tomatoes_d |
|---|---|---|---|
| PNG | 2285 KB | 2300 KB | 3952 KB |
| q92 | 490 KB | 491 KB | 739 KB |
| q94 | 597 KB | 597 KB | 887 KB |
| q95 | 631 KB | 632 KB | 981 KB |
| q96 | 673 KB | 674 KB | 1106 KB |
| q98 | 789 KB | 789 KB | 1415 KB |

## 선정

**선정 JPEG quality = 94** — 경계 3케이스 모두 CRR ≥ 1.0 을 만족하는 최소값. q92 대비 응답 크기 평균 +21%. config `RESPONSE_JPEG_QUALITY` 기본값으로 반영.
