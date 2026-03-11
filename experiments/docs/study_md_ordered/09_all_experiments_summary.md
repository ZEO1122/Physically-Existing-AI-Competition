# Codex Test 전체 실험 요약

이 문서는 `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test` 안에서 지금까지 수행한 주요 분석, 코드 변경, 실험, 비교 결과를 한 파일로 정리한 통합 요약본이다.

기준 시각:

- 2026-03-09 KST

## 1. 작업 루트와 데이터 경로

작업 루트:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test`

실제 데이터 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/open`

구성:

- `train`: 1000
- `dev`: 100
- `test`: 1000

## 2. 환경 확인

학습은 현재 이 컴퓨터의 로컬 환경에서 실행했다.

- conda env: `multiview`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- 실행 로그 기준 장치: `cuda`

초기에는 샌드박스에서 GPU 접근이 막혀 있었지만,
권한 상승 후 동일 머신의 GPU를 사용해 학습을 진행했다.

## 3. 데이터 분석 핵심 결론

분석 파일:

- [analyze_dataset.py](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analyze_dataset.py)
- [dataset_report.json](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analysis/dataset_report.json)

핵심 관찰:

- `train`이 `dev/test`보다 전반적으로 더 밝다
- 조명/배경 도메인 차이가 크다
- `simulation.mp4`에서 얻는 motion 통계가 unstable 라벨과 강한 상관을 가진다

해석:

- RGB만 보는 모델은 도메인 차이에 취약할 수 있다
- 구조물 중심 crop, mask/geometry, motion auxiliary가 중요하다

## 4. 실험 전략

초기 전략:

- `hybrid`: 구조물 mask, geometry, pseudo-3D, simulation auxiliary 활용
- `physnet`: motion pseudo-target 활용

이후 방향:

- `hybrid`는 crop과 도메인 강건 증강 최적화
- `physnet`은 backbone 교체와 해상도 확장

## 5. 코드 변경 요약

### Hybrid 쪽

- `mask_bbox` crop 전처리 추가
- `crop_margin_ratio`, `crop_min_side_ratio` 튜닝 옵션 추가
- `strong_domain`, `crop_tuned` augmentation profile 추가

관련 파일:

- [train_hybrid_structure_aware_gated_simaux.py](/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/train_hybrid_structure_aware_gated_simaux.py)
- [run_competition_pipeline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py)

### PhysNet 쪽

- `resnet50`, `resnet101`, `convnext_tiny` backbone 지원 추가
- `384` 입력 기반 실험 지원

관련 파일:

- [physnet_multiview_baseline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/model_file/baseline_model/physnet_multiview_baseline.py)
- [run_competition_pipeline.py](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py)

## 6. Holdout 실험 요약

### 6-1. Hybrid 계열

| Model | Run | Valid CalLL | Valid Acc | 메모 |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 초기 기준 |
| Hybrid strong domain | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명/도메인 강건 증강 |
| Hybrid mask-bbox crop | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 구조물 중심 crop |
| Hybrid mask-bbox + strong domain | `hybrid_mask_bbox_strong_domain_holdout_v1` | `0.44980` | `0.83` | 조합 시 악화 |
| Hybrid mask-bbox + crop_tuned | `hybrid_mask_bbox_crop_tuned_holdout_v1` | `0.41520` | `0.83` | 조합은 단독보다 약함 |
| Hybrid mask-bbox margin `0.10` | `hybrid_mask_bbox_margin010_holdout_v1` | `0.35864` | `0.87` | hybrid holdout 최적 |
| Hybrid mask-bbox margin `0.26` | `hybrid_mask_bbox_margin026_holdout_v1` | `0.37307` | `0.87` | margin 과대 |

결론:

- `hybrid`에서는 `mask_bbox crop`이 가장 큰 개선을 줬다
- crop margin은 `0.10`이 가장 좋았다
- 강한 photometric 증강을 crop과 단순 조합하면 오히려 악화됐다

### 6-2. PhysNet 계열

| Model | Run | Valid CalLL | Valid Acc | 메모 |
|---|---|---:|---:|---|
| PhysNet baseline | `gpu_physnet_holdout_v1` | `0.45928` | `0.82` | resnet34 기반 |
| PhysNet ResNet50 384 | `gpu_physnet_r50_384_v1` | `0.40320` | `0.84` | 해상도 확장 성공 |
| PhysNet ResNet101 384 | `gpu_physnet_r101_384_v1` | `0.51625` | `0.70` | 실패 |
| PhysNet ConvNeXt-Tiny 384 | `gpu_physnet_convnext_tiny_384_v1` | `0.30335` | `0.86` | physnet holdout 최적 |

결론:

- 같은 ResNet 계열을 키우는 것보다
- `ConvNeXt-Tiny`로 backbone 계열을 바꾸는 것이 훨씬 효과적이었다

## 7. CV 실험 요약

| Model | Run | OOF CalLL | OOF Acc | Dev-only OOF Logloss |
|---|---|---:|---:|---:|
| Hybrid baseline CV | `gpu_hybrid_cv_v1` | `0.02151044` | `0.99364` | `0.11808185` |
| Hybrid mask-bbox margin `0.10` CV | `hybrid_mask_bbox_margin010_cv_v1` | `0.01527694` | `0.99727` | `0.09747774` |
| PhysNet ResNet50 384 CV | `gpu_physnet_r50_384_cv_v1` | `0.07421409` | `0.98000` | `0.26508199` |
| PhysNet ConvNeXt-Tiny 384 CV | `gpu_physnet_convnext_tiny_384_cv_v1` | `0.00076575` | `1.00000` | `0.00410392` |

결론:

- `hybrid`는 crop tuning 후 CV가 개선됐다
- 하지만 최종 CV 최고는 `physnet convnext_tiny 384`

## 8. 블렌딩 검증

비교 대상:

- `physnet convnext_tiny 384 cv`
- `hybrid mask_bbox margin 0.10 cv`

OOF 블렌드 결과:

| Hybrid | PhysNet | OOF Logloss | OOF Acc |
|---:|---:|---:|---:|
| `0.00` | `1.00` | `0.00076575` | `1.00000` |
| `0.01` | `0.99` | `0.00080889` | `1.00000` |
| `0.02` | `0.98` | `0.00085235` | `1.00000` |
| `0.05` | `0.95` | `0.00098338` | `1.00000` |
| `0.10` | `0.90` | `0.00120919` | `1.00000` |

결론:

- 블렌딩 이득이 없었다
- `physnet convnext_tiny 384` 단독이 최선이었다

관련 파일:

- [blend_oof_scan.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/blend_oof_scan.csv)
- [blend_summary.json](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/blend_summary.json)

## 9. 현재 최종 판단

현재 기준 최우선 제출 후보:

- `physnet convnext_tiny 384 cv`

추천 제출 파일:

- [submission_cv_equal.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_equal.csv)

보조 제출 파일:

- [submission_cv_weighted.csv](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_weighted.csv)

## 10. 함께 정리된 참고 문서

- [experiment_log.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/experiment_log.md)
- [recommendations.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/recommendations.md)
- [strategy.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/strategy.md)
- [augmentation_and_normalization_note.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/augmentation_and_normalization_note.md)
- [train_vs_valid_loss_note.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/train_vs_valid_loss_note.md)
- [blending_note.md](/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/blending_note.md)

## 11. 한 줄 요약

- `hybrid`는 crop tuning으로 크게 좋아졌고
- `physnet`은 `ConvNeXt-Tiny 384`로 가장 크게 개선됐으며
- 최종적으로는 `physnet convnext_tiny 384` 단독 제출이 현재 최선이다.
