# 디렉토리 정리 가이드

## 현재 기준 핵심 구조

- `experiments/`
  - 실험 관리 루트
- `experiments/docs/`
  - 학습/제출/분석용 문서
- `experiments/assets/`
  - 예시 입력, breakdown 자료, 시각화 자료
- `experiments/analysis/`
  - 데이터 분석 결과 JSON
- `experiments/outputs/`
  - 학습 결과, 제출 파일, 블렌드 결과
- `experiments/logs/`
  - 장시간 실행 로그
- `experiments/training/`
  - hybrid 계열 학습 코드와 원본 실험 실행 산출물
- `experiments/models/`
  - physnet, baseline, blend 관련 모델 코드
- `dataset_analysis/scripts/`
  - 데이터셋 분석 스크립트
- `dataset_analysis/reports/`
  - 데이터셋 분석 결과물과 리포트

## 이번에 정리한 내용

- macOS 메타파일 `._*`, `.DS_Store` 삭제
- Python 캐시 `__pycache__` 삭제
- `codex_test`를 `experiments`로 변경
- 상단을 `experiments / dataset_analysis` 기준으로 재편
- 예시 자료 디렉토리를 `experiments/assets/` 아래로 이동
  - `hybrid_input_breakdown`
  - `physnet_input_breakdown`
  - `hybrid_input_examples`
- 공부용 문서 모음을 `experiments/docs/study_md_ordered/`로 이동
- 루트의 `대회정보 정리.md`를 `experiments/docs/reference/`로 연결
- `data_reconstrcture`는 `experiments/training/`으로 이동
- `model_file`은 `experiments/models/`로 이동
- `data_analyze`는 `dataset_analysis/scripts/`로 이동
- `dataset_report`는 `dataset_analysis/reports/`로 이동

## 호환성

- 기존 문서 링크가 깨지지 않도록 예전 경로에는 심볼릭 링크를 남겨뒀다.
- 따라서 기존 경로로 접근해도 동작한다.
