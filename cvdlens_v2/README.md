# cvdlens_v2 코드 구성

이 패키지는 배포 서버가 아니라 모델 학습과 연구 결과 재현에 사용한다. 배포 런타임은
`cvd-lens/inference`에 있다.

## 핵심 모듈

- `color.py`: 색 공간 변환
- `simulation.py`: P/D/T 시뮬레이터와 Daltonization 기준선
- `confusion.py`: 혼동 가중치
- `basis.py`: 색각 유형별 가시 부분공간
- `model.py`: P/D 학습 모델 구조
- `losses.py`: 학습 손실

이 파일들은 학습·평가 스크립트가 공통으로 사용하므로 유지한다.

## 학습과 내보내기

- `train.py`, `kaggle_train.py`: 학습 진입점
- `validate_loss.py`, `validate_multi.py`, `test_model.py`: 학습 검증
- `select_best.py`, `heldout_check.py`: 체크포인트 선택과 홀드아웃 검사
- `export_onnx.py`, `parity_check.py`: P/D ONNX 내보내기와 동등성 검사

## 정량 평가

- `step3_eval_set.py`, `step3_metrics.py`, `step3_eval.py`: Phase 3 평가
- `daily_test.py`, `post_retrain_eval.py`: 일상 이미지와 재학습 전후 평가
- `infer_local.py`: 배포 추론 경로의 로컬 미러
- `smoke_deploy.py`, `stress_deploy.py`: 배포 API 검사
- `tritan_hue_method.py`, `verify_analytic.py`: 현행 T 규칙 기반 방식 검증

## 연구 진단과 그림 생성

`artifact_probe*`, `audit_*`, `diag_*`, `diagnose_*`, `gate_probe.py`,
`ray_scan.py`, `sanity*`, `skin_analysis.py`, `tiled_infer.py` 등은 일반 실행에
필요하지 않지만 `outputs/`, `reports/`, `paper/`의 수치와 그림을 재현하기 위해 보존한다.

새 기능은 핵심 모듈에 넣고, 일회성 분석은 기존 진단 스크립트에 섞지 말고 목적이 드러나는
별도 스크립트로 추가한다. 결과가 채택되면 대응 보고서에 실행 명령과 산출물 경로를 기록한다.

## 실행 방식

저장소 루트에서 모듈로 실행한다.

```powershell
python -m cvdlens_v2.test_model
python -m cvdlens_v2.kaggle_train --help
python -m cvdlens_v2.verify_analytic
```

일부 과거 진단은 로컬 COCO 경로나 체크포인트를 요구한다. 해당 파일의 상단 상수와 연결된
보고서를 확인한 뒤 실행한다.
