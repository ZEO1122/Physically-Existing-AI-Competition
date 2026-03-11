# Physically Existing AI Competition

DACON `구조물 안정성 물리 추론 AI 경진대회`를 위해 구축한 실험용 프로젝트입니다.  
이 프로젝트의 목표는 `front / top` 두 시점의 정적 이미지만으로 구조물이 시뮬레이션 시작 후 10초 동안 안정 상태를 유지할지, 혹은 불안정 상태로 전환될지를 확률로 예측하는 것입니다.

한 줄 요약:
- `멀티뷰 이미지 + 구조 해석 + 미래 motion 보조학습`으로 구조물의 물리적 안정성을 예측하는 실험 프로젝트

공식 링크:
- 대회 설명: https://dacon.io/competitions/official/236686/overview/description
- 대회 규칙/동의사항: https://dacon.io/competitions/official/236686/overview/agreement

현재 상태:
- 2026년 3월 11일 기준 대회는 진행 중입니다.

## 프로젝트 개요

이 대회는 단순한 이미지 분류보다 `물리적 안정성 추론`에 더 가깝습니다.  
학습 데이터에는 `front.png`, `top.png`와 함께 `simulation.mp4`가 제공되지만, 평가 데이터에서는 정적 이미지 두 장만 사용할 수 있습니다. 따라서 핵심 과제는 학습 단계에서 영상의 물리 정보를 최대한 활용하고, 추론 단계에서는 이를 정적 이미지 표현으로 일반화하는 것입니다.

실험 과정에서 확인한 중요한 특성은 다음과 같습니다.
- `train`과 `dev/test` 사이에 밝기와 조명 분포 차이가 존재합니다.
- `simulation.mp4`에서 추출한 motion 정보는 안정성 라벨과 높은 상관을 가집니다.
- 따라서 RGB 과적합을 줄이고, 구조 정보와 미래 움직임 신호를 함께 활용하는 접근이 유효했습니다.

## 접근 방식

이 프로젝트는 크게 두 가지 모델 축으로 전개되었습니다.

### 1. Hybrid

`hybrid`는 RGB만 사용하지 않고, 구조물 자체의 형상 정보를 함께 해석하는 모델입니다.

핵심 아이디어:
- `front / top` 이미지를 직접 입력
- heuristic mask 추출
- mask에서 geometry feature 39차원 생성
- front/top을 결합한 pseudo-3D 구조 힌트 사용
- simulation 기반 auxiliary supervision 추가

이 접근은 조명과 배경보다 `구조적 형태`에 더 집중하도록 만들기 위한 설계입니다.

### 2. PhysNet

`physnet`은 멀티뷰 이미지 분류에 `미래 motion pseudo-target` 보조학습을 결합한 모델입니다.

핵심 아이디어:
- `front / top` 이미지를 공유 backbone으로 인코딩
- 두 뷰 feature를 fusion하여 안정성 분류
- `simulation.mp4`에서 여러 시점의 motion heatmap을 추출
- 분류 loss와 motion auxiliary loss를 함께 최적화

이 접근은 “현재 구조가 미래에 어떻게 움직일지”를 내부 표현에 학습시키는 방향입니다.

## 아키텍처 개요

### Hybrid 파이프라인

```text
front.png / top.png
    -> heuristic mask 추출
    -> geometry feature 39차원 생성
    -> pseudo-3D 구조 힌트 생성
    -> RGB backbone(EfficientNet 계열)
    -> mask / geometry / RGB feature fusion
    -> stability classification
```

### PhysNet 파이프라인

```text
front.png / top.png
    -> shared backbone(ResNet / ConvNeXt)
    -> multi-view feature fusion
    -> classification head

train only:
simulation.mp4
    -> motion pseudo-target 생성
    -> motion auxiliary head supervision
```

## 현재까지의 결과

실험 결과, 현재 가장 강한 제출 축은 `physnet convnext_tiny 384 CV`입니다.

대표 실험 비교:

| 모델 | 설정 | 검증 방식 | 핵심 결과 |
|---|---|---|---|
| Hybrid | `mask_bbox margin 0.10` | CV | `OOF logloss_cal = 0.01528` |
| PhysNet | `resnet50 384` | CV | `OOF logloss_cal = 0.07421` |
| PhysNet | `convnext_tiny 384` | CV | `OOF logloss_cal = 0.00077` |
| PhysNet | `convnext_tiny 448 + dense motion` | CV | `OOF logloss_cal = 0.00321` |
| PhysNet | `convnext_tiny 448 + weak multi-scale TTA` | holdout | `valid_logloss_cal = 0.20222` |

현재 제출 우선순위:
1. [gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv)
2. [daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_equal.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1_20260310_112018/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_equal.csv)
3. [daytime_physnet_convnext_tiny_448_holdout_tta096_v1__submission_holdout.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/outputs/physnet/daytime_physnet_convnext_tiny_448_holdout_tta096_v1_20260310_111733/daytime_physnet_convnext_tiny_448_holdout_tta096_v1__submission_holdout.csv)

관련 문서:
- 통합 보고서: [final_master_report.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/final_master_report.md)
- 전략 문서: [strategy.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/strategy.md)
- 실험 비교 및 후속 제안: [recommendations.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/recommendations.md)
- 제출 점수 기록: [submission_score_note_20260310.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/submission_score_note_20260310.md)

## 디렉토리 구조

프로젝트는 현재 두 개의 상위 축으로 정리되어 있습니다.

- [experiments](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments)
  - 실험 코드, 모델 코드, 전략 문서, 제출 파일, 결과물
- [dataset_analysis](/home/ubuntu/Desktop/Physically Existing AI Competition/dataset_analysis)
  - 데이터셋 분석 스크립트와 분석 결과

주요 위치:
- 데이터 루트: [open](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/training/open)
- 실험 실행 진입점: [run_competition_pipeline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/run_competition_pipeline.py)
- Hybrid 메인 코드: [train_hybrid_structure_aware_gated_simaux.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/training/train_hybrid_structure_aware_gated_simaux.py)
- PhysNet 메인 코드: [physnet_multiview_baseline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/models/baseline_model/physnet_multiview_baseline.py)
- 구조 안내: [STRUCTURE_GUIDE.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/docs/STRUCTURE_GUIDE.md)

호환성 때문에 기존 이름도 심볼릭 링크로 유지하고 있습니다.
- [codex_test](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test)
- [data_reconstrcture](/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture)
- [model_file](/home/ubuntu/Desktop/Physically Existing AI Competition/model_file)
- [data_analyze](/home/ubuntu/Desktop/Physically Existing AI Competition/data_analyze)
- [dataset_report](/home/ubuntu/Desktop/Physically Existing AI Competition/dataset_report)

## 사용 환경

로컬 실행 환경:
- OS: `Ubuntu 24.04.3 LTS`
- Kernel: `Linux 6.8.0-101-generic x86_64`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- GPU Memory: `16376 MiB`
- NVIDIA Driver: `590.48.01`

실험에 사용한 주요 버전:
- Python: `3.11.6`
- PyTorch: `2.10.0`
- TorchVision: `0.25.0`
- NumPy: `2.4.2`
- Pandas: `3.0.1`
- OpenCV: `4.13.0`
- scikit-learn: `1.8.0`
- Pillow: `12.1.1`
- CUDA runtime: `13.0`
- cuDNN: `9.19.0.56`

## 문서 안내

이 프로젝트를 처음 읽는다면 아래 순서를 추천합니다.

1. [final_master_report.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/final_master_report.md)
2. [strategy.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/strategy.md)
3. [recommendations.md](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/recommendations.md)
4. [study_md_ordered](/home/ubuntu/Desktop/Physically Existing AI Competition/experiments/docs/study_md_ordered)
