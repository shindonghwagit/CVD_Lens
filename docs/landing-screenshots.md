# 랜딩 기능 카드 썸네일

랜딩 페이지의 이미지·카메라 보정 카드는 실제 보정 파이프라인으로 만든 원본/보정 비교 이미지를 사용한다.

## 현재 자산

| 카드 | 파일 | 장면 | 보정 유형 |
| --- | --- | --- | --- |
| 02 / IMAGE | `cvd-lens/public/landing/preview_image.jpg` | 색연필 | 적색맹(P) |
| 03 / CAMERA | `cvd-lens/public/landing/preview_camera.jpg` | 화단 | 녹색맹(D) |

- 크기: 800×500(16:10)
- 구성: 왼쪽 원본, 오른쪽 보정본
- 표시 방식: `object-cover object-top`
- 원본은 `external_eval_v2/images`의 검증 데이터에서 선택했다.

## 교체 기준

1. 색 변화가 분명하지만 과도한 왜곡이나 클리핑이 없는 장면을 고른다.
2. 배포 코드와 같은 보정 경로를 사용한다.
3. 800×500 JPEG로 저장하고 두 파일명을 유지한다.
4. `npm run build`로 이미지 경로와 프로덕션 빌드를 확인한다.
