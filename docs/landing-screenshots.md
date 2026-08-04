# 랜딩 기능 카드 스크린샷 촬영 가이드

02/03 기능 카드 프리뷰는 **실제 /correction 화면 스크린샷**을 사용한다. 이 환경에서는
백엔드(Render) 기동 + 파일 업로드 + 웹캠 권한이 필요해 자동 촬영이 불가하므로, 아래
절차대로 직접 촬영해 파일만 넣으면 코드 변경 없이 자동 반영된다(없으면 줄무늬
플레이스홀더로 폴백).

## 공통
- 촬영 뷰포트: **데스크톱 1440×900** (DevTools 디바이스툴바 off, 100% 배율)
- 테마: 현행 light (`data-theme="light"`)
- 로컬 실행: `cd cvd-lens && npm run dev` → http://localhost:3000
- 백엔드 필요: `NEXT_PUBLIC_API_URL`이 살아있는 추론 서버를 가리켜야 실제 보정본이 뜸
  (로컬 `cvd-lens/inference` 기동 또는 https://cvd-lens.onrender.com)
- 저장 규격: **장변 1200px, JPEG q85, 300KB 이하**, object-cover object-top로 잘리므로
  상단(비교 슬라이더 영역)이 프레임에 들어오게 촬영

## preview_image.jpg (02 / IMAGE 카드)
1. URL: `http://localhost:3000/correction?tab=image`
2. COCO `000000001584.jpg`(빨강 버스) 업로드 → 타입 **녹색맹(D)**, severity 1.0
3. 원본·보정본 **나란히 비교(슬라이더)** 상태가 보이도록 대기(보정 완료 후)
4. 비교 영역 위주로 크롭 → `cvd-lens/public/landing/preview_image.jpg`

## preview_camera.jpg (03 / CAMERA 카드)
1. URL: `http://localhost:3000/correction` (카메라 탭)
2. 웹캠 권한 허용, 프리뷰/촬영 후 보정 결과 슬라이더 상태 노출
3. 크롭 → `cvd-lens/public/landing/preview_camera.jpg`

## 검증
- 파일 넣고 새로고침 → 카드에 스크린샷이 뜨면 성공(폴백 줄무늬가 사라짐)
- `npm run build` 재실행해 최적화 경로 확인
