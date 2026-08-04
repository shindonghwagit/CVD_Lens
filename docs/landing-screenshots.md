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
**사용자 제공 자산**: 모바일에서 보정한 한식(찌개) 원본/보정 비교 이미지(나란히
또는 wipe 상태). 아래 스펙으로 넣으면 코드 변경 없이 카드에 반영된다.
1. 원본/보정이 한 프레임에 담긴 비교 이미지를 준비
2. **카드 비율 16:10에 맞게 크롭**(카드가 `object-cover object-top`이라 상단이 살아남음 —
   비교 라벨/핵심 대비가 상단 절반에 오도록)
3. **장변 800px, JPEG q85, 300KB 이하**로 저장
4. 경로: `cvd-lens/public/landing/preview_image.jpg`

> 크롭/리사이즈가 필요하면 파일만 repo 어딘가(예: 프로젝트 루트)에 두고 알려주면
> 위 스펙으로 처리해 배치해 줄 수 있음.

## preview_camera.jpg (03 / CAMERA 카드)
1. URL: `http://localhost:3000/correction` (카메라 탭)
2. 웹캠 권한 허용, 프리뷰/촬영 후 보정 결과 슬라이더 상태 노출
3. 크롭 → `cvd-lens/public/landing/preview_camera.jpg`

## 검증
- 파일 넣고 새로고침 → 카드에 스크린샷이 뜨면 성공(폴백 줄무늬가 사라짐)
- `npm run build` 재실행해 최적화 경로 확인
