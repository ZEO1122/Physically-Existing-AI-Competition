# Physically Existing AI Competition

DACON `구조물 안정성 물리 추론 AI 경진대회` 작업 폴더입니다.

공식 링크:
- 대회 설명: https://dacon.io/competitions/official/236686/overview/description
- 대회 규칙/동의사항: https://dacon.io/competitions/official/236686/overview/agreement

현재 상태:
- 대회는 현재 진행 중입니다.
- DACON 설명 페이지 기준, 2026년 3월 11일 현재 종료까지 `D-19`로 표시됩니다.
- 주요 일정:
  - 2026-03-03: 대회 시작
  - 2026-03-23: 팀 병합 마감
  - 2026-03-30: 대회 종료
  - 2026-04-02: 코드 및 PPT 제출 마감
  - 2026-04-10: 코드 검증
  - 2026-04-13: 최종 수상자 발표

## 대회 개요

이 대회는 `2가지 시점의 구조물 이미지`를 입력으로 사용해, 시뮬레이션 시작 후 `10초 동안 구조물이 안정 상태를 유지할지`, 혹은 `불안정 상태로 전환될지`를 확률로 예측하는 문제입니다.

핵심 포인트:
- 입력 데이터는 `front`, `top` 두 시점 이미지입니다.
- `train`에는 10초 분량의 `simulation.mp4`가 함께 제공됩니다.
- `stable`은 10초 동안 의미 있는 이동/변형이 없는 경우입니다.
- `unstable`은 10초 이내 누적 이동 거리 `1.5cm 이상` 또는 구조 붕괴가 발생한 경우입니다.
- `dev/test`는 광원과 카메라 좌표가 무작위로 변하는 실제 평가 환경과 동일한 설정입니다.
- 따라서 `train`의 고정 환경에 과적합하지 않는 강건한 모델이 중요합니다.

## 이 폴더의 목적

이 워크스페이스는 아래 내용을 한곳에서 관리하기 위해 정리했습니다.

- 데이터셋 분석
- `hybrid` / `physnet` 계열 모델 실험
- 제출 파일 관리
- 실험 로그 및 전략 문서 정리
- 포트폴리오용 코드/문서 정리

## 최상단 구조

- [experiments](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments)
  - 실험 코드, 실험 문서, 제출 우선순위, 결과물 관리
- [dataset_analysis](/home/ubuntu/Desktop/Physically Existing AI Competition/dataset_analysis)
  - 데이터 분석 스크립트와 분석 리포트용 구조

호환용 링크:
- [codex_test](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test)
- [data_reconstrcture](/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture)
- [model_file](/home/ubuntu/Desktop/Physically Existing AI Competition/model_file)
- [data_analyze](/home/ubuntu/Desktop/Physically Existing AI Competition/data_analyze)
- [dataset_report](/home/ubuntu/Desktop/Physically Existing AI Competition/dataset_report)

## 주요 경로

- 데이터 루트: [open](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/training/open)
- 실행 진입점: [run_competition_pipeline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/run_competition_pipeline.py)
- 데이터 분석 스크립트: [analyze_dataset.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/analyze_dataset.py)
- Hybrid 메인 학습 코드: [train_hybrid_structure_aware_gated_simaux.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/training/train_hybrid_structure_aware_gated_simaux.py)
- PhysNet 메인 학습 코드: [physnet_multiview_baseline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/models/baseline_model/physnet_multiview_baseline.py)
- 구조 안내 문서: [STRUCTURE_GUIDE.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/docs/STRUCTURE_GUIDE.md)

## 현재까지의 작업 방향

주요 실험 축:
- `hybrid`
  - RGB + mask + geometry + pseudo-3D + auxiliary
- `physnet`
  - front/top 멀티뷰 + motion pseudo-target auxiliary

현재까지의 실험상 가장 강한 제출 축:
- `physnet convnext_tiny 384 CV`

현재 제출 우선순위:
1. [gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv)
2. [daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_equal.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1_20260310_112018/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_equal.csv)
3. [daytime_physnet_convnext_tiny_448_holdout_tta096_v1__submission_holdout.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/daytime_physnet_convnext_tiny_448_holdout_tta096_v1_20260310_111733/daytime_physnet_convnext_tiny_448_holdout_tta096_v1__submission_holdout.csv)

관련 문서:
- 통합 보고서: [final_master_report.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/final_master_report.md)
- 전략 문서: [strategy.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/strategy.md)
- 실험 제안/비교: [recommendations.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/recommendations.md)
- 제출 점수 기록: [submission_score_note_20260310.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/submission_score_note_20260310.md)

## 빠른 실행

환경 활성화:
```bash
conda activate multiview
```

데이터 분석:
```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/analyze_dataset.py"
```

Hybrid 실행:
```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/run_competition_pipeline.py" train-hybrid
```

PhysNet 실행:
```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/run_competition_pipeline.py" train-physnet
```

## 문서 읽기 추천

1. [final_master_report.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/final_master_report.md)
2. [strategy.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/strategy.md)
3. [recommendations.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/recommendations.md)
4. [study_md_ordered](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/docs/study_md_ordered)

## 주의

- 대회 데이터, 제출 파일, 예측값, 로그, 체크포인트는 공개 업로드 대상이 아닙니다.
- 포트폴리오 목적이라도 공개 범위는 반드시 대회 규칙을 기준으로 다시 확인해야 합니다.
