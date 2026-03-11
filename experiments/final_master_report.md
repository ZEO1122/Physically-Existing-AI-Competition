# Physically Existing AI Competition 최종 통합 보고서

이 문서는 `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test`에서 진행한 모든 핵심 작업을 한 파일에 통합 정리한 최종 보고서다.

이 문서만 보면 아래 내용을 모두 파악할 수 있게 구성했다.

- 데이터 위치와 구조
- 실행 환경
- 데이터 분석 결과
- 사용한 모델 전략
- 전처리와 증강
- 코드 변경 사항
- holdout 실험 결과
- CV 실험 결과
- 블렌딩 검증 결과
- 현재 최종 제출 후보

기준 시각:

- 2026-03-09 KST

---

## 1. 작업 위치와 데이터 경로

작업 루트:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test`

실제 데이터 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/open`

사용자 초기에 언급한 `/opne` 경로는 실제 데이터 위치가 아니었고,
실데이터는 위 경로에서 확인했다.

데이터 구조:

- `train`: 1000개
- `dev`: 100개
- `test`: 1000개

샘플 구성:

- `train`: `front.png`, `top.png`, `simulation.mp4`
- `dev`: `front.png`, `top.png`
- `test`: `front.png`, `top.png`

---

## 2. 실행 환경

학습과 실험은 외부 서버가 아니라 현재 로컬 머신에서 실행했다.

실행 환경:

- conda env: `multiview`
- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- driver: `590.48.01`

초기에는 기본 샌드박스 세션에서 GPU 접근이 막혀 있었고,
권한 상승 후 같은 머신의 GPU를 사용해 실험을 진행했다.

실제 학습 로그에는 반복적으로 아래 상태가 확인되었다.

- `Device: cuda | AMP: True`

즉 모든 본 실험은 로컬 머신의 GPU에서 수행되었다.

---

## 3. 데이터 분석 결과

### 3-1. 기본 통계

- `train` 라벨 분포: `unstable 500`, `stable 500`
- `dev` 라벨 분포: `unstable 52`, `stable 48`
- 모든 이미지 크기: `384 x 384`
- train 영상: `300 frame`, `30 fps`

### 3-2. 도메인 차이

밝기 평균:

- train front: `0.8548`
- dev front: `0.7621`
- test front: `0.7609`
- train top: `0.8840`
- dev top: `0.7812`
- test top: `0.7818`

해석:

- `train`과 `dev/test`는 밝기와 조명 분포가 다르다
- raw RGB에 과적합하면 일반화가 흔들릴 가능성이 높다
- 구조, silhouette, geometry 중심 해석이 중요하다

### 3-3. simulation.mp4의 가치

motion 통계:

- stable 평균 `motion_mean`: `0.0191`
- unstable 평균 `motion_mean`: `0.1556`
- `motion_mean`과 unstable 라벨 상관: `0.7018`
- `motion_q95`와 unstable 라벨 상관: `0.6862`

해석:

- `simulation.mp4`는 매우 강한 auxiliary supervision 신호다
- motion pseudo-target을 쓰는 전략은 타당하다

---

## 4. 전체 전략

### 4-1. 초기 전략

1순위 모델:

- `hybrid`

핵심:

- `front/top` 두 뷰 입력
- 구조물 mask 사용
- geometry feature 사용
- pseudo-3D 성격의 구조 정보 사용
- simulation auxiliary 사용

2순위 모델:

- `physnet`

핵심:

- `front/top` 두 뷰 입력
- `simulation.mp4`에서 motion pseudo-target 생성
- 분류 + 미래 물리 변화 보조학습

### 4-2. 이후 전략 변화

실험을 진행하면서 다음 방향으로 최적화했다.

- `hybrid`: 구조물 중심 crop과 margin 튜닝
- `physnet`: backbone 교체와 고해상도 입력 확장
- 최종적으로는 `physnet convnext_tiny 384`가 가장 강했다

---

## 5. 전처리와 증강 정리

## 5-1. Hybrid

핵심 전처리:

- `mask_bbox crop`
- mask binary tensor 생성
- geometry feature 추출
- mask meta feature 생성
- ImageNet mean/std normalize

핵심 증강:

- horizontal flip
- affine
- brightness / contrast / saturation / hue
- gamma
- grayscale
- gaussian blur
- autocontrast
- JPEG-like degradation

핵심 효과:

- 구조물 중심 정보 강화
- 배경 비중 감소
- 조명 차이에 대한 강건성 확보

## 5-2. PhysNet

핵심 전처리:

- 이미지 resize
- motion pseudo-target 생성
- ImageNet mean/std normalize

핵심 증강:

- horizontal flip
- affine
- brightness / contrast / saturation / hue
- blur
- JPEG-like degradation
- Random Erasing

핵심 효과:

- motion auxiliary와 이미지의 정합 유지
- 두 뷰를 동시에 흔들어 멀티뷰 일관성 확보

---

## 6. 코드 변경 사항

### 6-1. Hybrid 관련

수정 내용:

- `mask_bbox` crop 추가
- `crop_margin_ratio` 추가
- `crop_min_side_ratio` 추가
- `strong_domain` 증강 프로필 추가
- `crop_tuned` 증강 프로필 추가

수정 파일:

- `data_reconstrcture/train_hybrid_structure_aware_gated_simaux.py`
- `codex_test/run_competition_pipeline.py`

### 6-2. PhysNet 관련

수정 내용:

- `resnet50` 지원
- `resnet101` 지원
- `convnext_tiny` 지원
- `384` 입력 실험 지원

수정 파일:

- `model_file/baseline_model/physnet_multiview_baseline.py`
- `codex_test/run_competition_pipeline.py`

---

## 7. Holdout 실험 결과

## 7-1. Hybrid 계열

| Model | Run | Valid CalLL | Valid Acc | 해석 |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 기준 실험 |
| Hybrid strong domain | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명 대응 개선 |
| Hybrid mask_bbox | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 큰 개선 |
| Hybrid mask_bbox + strong_domain | `hybrid_mask_bbox_strong_domain_holdout_v1` | `0.44980` | `0.83` | 악화 |
| Hybrid mask_bbox + crop_tuned | `hybrid_mask_bbox_crop_tuned_holdout_v1` | `0.41520` | `0.83` | 단독보다 약함 |
| Hybrid margin `0.10` | `hybrid_mask_bbox_margin010_holdout_v1` | `0.35864` | `0.87` | hybrid holdout 최고 |
| Hybrid margin `0.26` | `hybrid_mask_bbox_margin026_holdout_v1` | `0.37307` | `0.87` | margin 과대 |

핵심 결론:

- `hybrid`는 구조물 중심 crop이 가장 중요했다
- crop margin은 `0.10`이 최적이었다
- 강한 color augmentation을 crop과 그대로 결합하면 오히려 나빠졌다

## 7-2. PhysNet 계열

| Model | Run | Valid CalLL | Valid Acc | 해석 |
|---|---|---:|---:|---|
| PhysNet baseline | `gpu_physnet_holdout_v1` | `0.45928` | `0.82` | 초기 기준 |
| PhysNet ResNet50 384 | `gpu_physnet_r50_384_v1` | `0.40320` | `0.84` | 해상도 확장 성공 |
| PhysNet ResNet101 384 | `gpu_physnet_r101_384_v1` | `0.51625` | `0.70` | 실패 |
| PhysNet ConvNeXt-Tiny 384 | `gpu_physnet_convnext_tiny_384_v1` | `0.30335` | `0.86` | physnet holdout 최고 |

핵심 결론:

- `ResNet101`은 오히려 나빴다
- `ConvNeXt-Tiny`는 크게 개선됐다
- 같은 ResNet 확장보다 backbone 계열 교체가 더 효과적이었다

---

## 8. CV 실험 결과

| Model | Run | OOF CalLL | OOF Acc | Dev-only OOF Logloss |
|---|---|---:|---:|---:|
| Hybrid baseline CV | `gpu_hybrid_cv_v1` | `0.02151044` | `0.99364` | `0.11808185` |
| Hybrid margin `0.10` CV | `hybrid_mask_bbox_margin010_cv_v1` | `0.01527694` | `0.99727` | `0.09747774` |
| PhysNet ResNet50 384 CV | `gpu_physnet_r50_384_cv_v1` | `0.07421409` | `0.98000` | `0.26508199` |
| PhysNet ConvNeXt-Tiny 384 CV | `gpu_physnet_convnext_tiny_384_cv_v1` | `0.00076575` | `1.00000` | `0.00410392` |

핵심 결론:

- `hybrid`는 margin `0.10`으로 CV가 좋아졌다
- 하지만 최종 CV 최고는 `physnet convnext_tiny 384`
- 현재까지의 최고 CV 후보는 `physnet convnext_tiny 384`

---

## 9. 블렌딩 검증

비교 대상:

- `physnet convnext_tiny 384 CV`
- `hybrid mask_bbox margin 0.10 CV`

OOF 블렌드 스캔:

| Hybrid Weight | PhysNet Weight | OOF Logloss | OOF Acc |
|---:|---:|---:|---:|
| `0.00` | `1.00` | `0.00076575` | `1.00000` |
| `0.01` | `0.99` | `0.00080889` | `1.00000` |
| `0.02` | `0.98` | `0.00085235` | `1.00000` |
| `0.05` | `0.95` | `0.00098338` | `1.00000` |
| `0.10` | `0.90` | `0.00120919` | `1.00000` |

결론:

- 블렌딩 이득이 없었다
- `hybrid`를 아주 조금만 섞어도 OOF logloss가 상승했다
- 최종적으로는 `physnet convnext_tiny 384` 단독이 최선이다

---

## 10. 블렌딩 관련 개념 정리

### Soft Voting

- 모델 확률을 평균하는 방식
- 이번 블렌딩 실험이 여기에 해당한다

### Hard Voting

- 클래스 다수결 방식
- logloss 기반 대회에서는 보통 비효율적이다

### Stacking

- 여러 모델의 예측을 다시 메타 모델에 넣어 학습하는 방식
- 더 정교하지만 leakage와 과적합 위험이 있다

이번 대회 결론:

- `soft voting`으로도 단독 `physnet convnext_tiny 384`를 넘지 못했다
- `hard voting`은 애초에 적합하지 않다
- `stacking`은 지금 상황에서 우선순위가 낮다

---

## 11. train loss보다 valid loss가 더 낮았던 이유

주요 원인:

- train은 증강이 들어가서 더 어렵다
- dropout, label smoothing, auxiliary loss가 train loss를 올린다
- train loss는 epoch 전체 평균이고, valid는 epoch 끝 모델로 계산된다
- 특정 fold가 상대적으로 쉬울 수 있다

현재 해석:

- 이 현상 자체는 이상하지 않다
- 현재 설정에서는 충분히 자연스럽다

---

## 12. 현재 최종 결론

현재 최우선 모델:

- `physnet convnext_tiny 384`

현재 최우선 제출 방식:

- 블렌딩 없음
- 단독 제출

현재 판단 기준:

- holdout 최고: `physnet convnext_tiny 384`
- CV 최고: `physnet convnext_tiny 384`
- 블렌딩 이득 없음

따라서 현재 최종 제출 우선순위는 아래와 같다.

1. `physnet convnext_tiny 384` 단독
2. 필요 시 weighted 제출본도 후보로 검토
3. 현재 시점에서는 `hybrid` 혼합 비추천

---

## 13. 제출 파일 경로

현재 최우선 제출 파일:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_equal.csv`

같은 실험의 가중 제출 파일:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_weighted.csv`

블렌드 제출 파일:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/submission_blend_best.csv`

하지만 현재 권장 제출은:

- `submission_cv_equal.csv`

---

## 14. 최종 한 줄 요약

이번 대회에서는
`hybrid`는 crop tuning으로 많이 좋아졌고,
`physnet`은 `ConvNeXt-Tiny 384`로 가장 크게 개선됐으며,
최종적으로는 `physnet convnext_tiny 384` 단독 제출이 현재 가장 합리적이다.
