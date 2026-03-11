# Next Experiment Recommendations

기준 모델:

- hybrid holdout: `gpu_hybrid_holdout_v1`
- hybrid cv: `gpu_hybrid_cv_v1`
- physnet holdout best: `gpu_physnet_r50_384_v1`
- physnet cv best: `gpu_physnet_r50_384_cv_v1`

## 우선순위

### 1. Hybrid에 조명 강건 증강 강화

목표:

- `train`과 `dev/test` 사이 밝기/배경 도메인 차이를 더 강하게 완화

아이디어:

- brightness / contrast / saturation / hue 증강 범위 확대
- gamma 증강 추가
- grayscale 확률성 적용
- autocontrast / blur / jpeg degradation 확률 소폭 증가

예상 효과:

- RGB 과적합 감소
- `dev/test` 일반화 개선

리스크:

- 과한 photometric distortion은 구조 단서를 훼손할 수 있음

### 2. Hybrid에 구조물 중심 crop 정규화

목표:

- 배경 비중 축소, 구조물 형상 비중 확대

아이디어:

- mask bounding box를 기준으로 margin crop 후 resize

예상 효과:

- 도메인 차이 완화
- 형상 기반 특징 강화

### 3. PhysNet trunk 추가 실험

후보:

- `ResNet101`
- `ConvNeXt-T`

목표:

- motion auxiliary 구조는 유지하면서 trunk 표현력 확장

### 4. CV 블렌딩

후보:

- `hybrid_cv` 중심 소량 `physnet_cv` 블렌드
- 예: `0.95 / 0.05`, `0.90 / 0.10`

목표:

- 에러 패턴 상호보완 확인

## 현재 턴에서 실행할 실험

실험명:

- `hybrid_strong_domain_aug_holdout_v1`

비교 기준:

- `gpu_hybrid_holdout_v1`

변경 내용:

- stronger photometric augmentation
- gamma jitter 추가
- grayscale augmentation 추가

실험 결과:

- run: `hybrid_strong_domain_aug_holdout_v1`
- `valid_logloss_cal`: `0.4262861996711964`
- `valid_accuracy`: `0.82`

기존 hybrid holdout 대비:

- baseline `gpu_hybrid_holdout_v1`
  - `valid_logloss_cal`: `0.5187817784923585`
  - `valid_accuracy`: `0.79`
- 개선폭:
  - logloss `-0.09250`
  - accuracy `+0.03`

판단:

- 1번 실험은 holdout 기준으로 유효했다.
- 다음 우선순위는 `2. Hybrid에 구조물 중심 crop 정규화`가 맞다.

## 후속 실행 결과

실험명:

- `hybrid_mask_bbox_holdout_v1`

변경 내용:

- `front/top` 각각에서 mask bounding box 기준 square crop
- margin 포함 crop 후 `384` resize
- photometric 증강은 baseline과 동일하게 유지

실험 결과:

- `valid_logloss_cal`: `0.3591907615391547`
- `valid_accuracy`: `0.86`

비교:

- baseline `gpu_hybrid_holdout_v1`
  - `0.5187817784923585`, `0.79`
- strong-domain `hybrid_strong_domain_aug_holdout_v1`
  - `0.4262861996711964`, `0.82`
- mask-bbox crop `hybrid_mask_bbox_holdout_v1`
  - `0.3591907615391547`, `0.86`

판단:

- 2번 실험은 1번보다 더 크게 개선됐다.
- `strong_domain + mask_bbox` 조합 검증이 다음 우선순위였다.

## 조합 검증 결과

실험명:

- `hybrid_mask_bbox_strong_domain_holdout_v1`

실험 결과:

- `valid_logloss_cal`: `0.44979716444498696`
- `valid_accuracy`: `0.83`

비교:

- `hybrid_mask_bbox_holdout_v1`
  - `0.3591907615391547`, `0.86`
- `hybrid_mask_bbox_strong_domain_holdout_v1`
  - `0.44979716444498696`, `0.83`

판단:

- `mask_bbox`에 `strong_domain`을 그대로 더하면 오히려 성능이 떨어졌다.
- 현재 hybrid holdout 최적안은 `mask_bbox` 단독이다.
- 다음 우선순위는 `mask_bbox`를 유지한 채 photometric 증강 강도를 더 약하게 재설계하는 것이다.

## 약화 증강 재설계 결과

실험명:

- `hybrid_mask_bbox_crop_tuned_holdout_v1`

실험 결과:

- `valid_logloss_cal`: `0.41519983160023016`
- `valid_accuracy`: `0.83`

비교:

- `hybrid_mask_bbox_holdout_v1`
  - `0.3591907615391547`, `0.86`
- `hybrid_mask_bbox_strong_domain_holdout_v1`
  - `0.44979716444498696`, `0.83`
- `hybrid_mask_bbox_crop_tuned_holdout_v1`
  - `0.41519983160023016`, `0.83`

판단:

- `crop_tuned`는 `strong_domain + mask_bbox`보다는 나았지만, `mask_bbox` 단독을 넘지 못했다.
- 현재 hybrid holdout 최적안은 여전히 `mask_bbox` 단독이다.
- 다음으로는 photometric 증강보다 `mask_bbox` 기반 CV 검증이나, crop margin 자체를 튜닝하는 편이 더 타당하다.

## crop margin 튜닝 결과

실험 설정:

- `crop_profile=mask_bbox`
- `crop_min_side_ratio=0.42` 고정
- `crop_margin_ratio`만 변경해 holdout 비교

holdout 비교:

| Margin | Run | Valid CalLL | Valid Acc | 판단 |
|---:|---|---:|---:|---|
| `0.10` | `hybrid_mask_bbox_margin010_holdout_v1` | `0.35864` | `0.87` | 최고 |
| `0.18` | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 기존 기본값 |
| `0.26` | `hybrid_mask_bbox_margin026_holdout_v1` | `0.37307` | `0.87` | margin 과대 |

판단:

- holdout 기준 최적 crop margin은 `0.10`이다.
- 기존 기본값 `0.18`보다 개선폭은 작지만, calibrated logloss와 accuracy 모두 소폭 개선됐다.
- margin을 더 크게 주면 배경 비중이 다시 늘어나며 성능이 악화되는 흐름이 보인다.

## crop margin 0.10 CV 결과

실험명:

- `hybrid_mask_bbox_margin010_cv_v1`

실험 결과:

- `oof_logloss_cal`: `0.015276939762022709`
- `oof_accuracy`: `0.9972727272727273`
- `dev-only oof logloss`: `0.09747774026268885`

기존 hybrid CV와 비교:

- baseline `gpu_hybrid_cv_v1`
  - `oof_logloss_cal 0.021510442103023657`
  - `oof_accuracy 0.9936363636363637`
- tuned margin `hybrid_mask_bbox_margin010_cv_v1`
  - `oof_logloss_cal 0.015276939762022709`
  - `oof_accuracy 0.9972727272727273`

판단:

- crop margin `0.10`은 holdout뿐 아니라 CV에서도 기존 hybrid baseline보다 더 좋았다.
- 현재 hybrid 계열 최고 CV 설정은 `mask_bbox + crop_margin_ratio=0.10`이다.
- 다음 단계는 이 설정을 기준으로 제출 후보를 만들거나, `physnet`과 낮은 비율 블렌딩을 검증하는 것이다.

## PhysNet 백본 확장 검증

실험명:

- `gpu_physnet_r101_384_v1`

설정:

- backbone: `resnet101`
- img size: `384`
- motion size: `96`

실험 결과:

- `valid_logloss_cal`: `0.5162458024837846`
- `valid_accuracy`: `0.7`

비교:

- `physnet resnet50 384`
  - `0.40319519115848723`, `0.84`
- `physnet resnet101 384`
  - `0.5162458024837846`, `0.7`

판단:

- 추천 후보 중 먼저 검증한 `ResNet101`은 현재 설정에서는 실패다.
- backbone을 크게 키우는 것만으로는 개선되지 않았다.
- 현재 physnet 최적안은 여전히 `ResNet50 384`다.
- physnet에서 다음 백본 후보를 더 보려면 동일 ResNet 확장보다 `ConvNeXt-T`처럼 계열 자체를 바꾸는 쪽이 더 의미 있다.

## PhysNet ConvNeXt-Tiny 결과

실험명:

- `gpu_physnet_convnext_tiny_384_v1`

설정:

- backbone: `convnext_tiny`
- img size: `384`
- motion size: `96`
- batch size: `8`

실험 결과:

- `valid_logloss_cal`: `0.3033521122607969`
- `valid_accuracy`: `0.86`

비교:

- `physnet resnet50 384`
  - `0.40319519115848723`, `0.84`
- `physnet convnext_tiny 384`
  - `0.3033521122607969`, `0.86`

판단:

- `ConvNeXt-Tiny`는 `ResNet50 384`보다 명확히 더 좋았다.
- `ResNet101`이 실패한 반면, backbone 계열을 바꾸는 접근은 효과가 있었다.
- 현재 physnet 최적 holdout 후보는 `ConvNeXt-Tiny 384`다.
- 다음 단계는 이 설정으로 `CV`를 돌리거나, `hybrid mask_bbox margin 0.10 CV`와 블렌딩 가능성을 보는 것이다.

## PhysNet ConvNeXt-Tiny CV 결과

실험명:

- `gpu_physnet_convnext_tiny_384_cv_v1`

설정:

- backbone: `convnext_tiny`
- img size: `384`
- motion size: `96`
- batch size: `8`

실험 결과:

- `oof_logloss_cal`: `0.0007657534577136482`
- `oof_accuracy`: `1.0`
- `dev-only oof logloss`: `0.0041039192668788655`

비교:

- `physnet resnet50 384 cv`
  - `oof_logloss_cal 0.07421408550876968`
  - `oof_accuracy 0.98`
- `hybrid mask_bbox margin 0.10 cv`
  - `oof_logloss_cal 0.015276939762022709`
  - `oof_accuracy 0.9972727272727273`
- `physnet convnext_tiny 384 cv`
  - `oof_logloss_cal 0.0007657534577136482`
  - `oof_accuracy 1.0`

판단:

- 현재 CV 기준 최고 모델은 `physnet convnext_tiny 384`다.
- `hybrid margin 0.10 CV`보다도 훨씬 낮은 OOF logloss가 나왔다.
- 다만 수치가 매우 낮기 때문에, 실제 제출 일반화는 별도로 주의해서 봐야 한다.
- 다음 단계는 이 설정을 최우선 제출 후보로 두고, 필요하면 `hybrid`와 낮은 비율 블렌딩이 정말 추가 이득이 있는지만 검증하는 것이다.

## PhysNet ConvNeXt-Tiny와 Hybrid 블렌드 검증

비교 대상:

- `physnet convnext_tiny 384 cv`
- `hybrid mask_bbox margin 0.10 cv`

OOF blend scan 결과:

| Hybrid | PhysNet | OOF Logloss | OOF Acc | 판단 |
|---:|---:|---:|---:|---|
| `0.00` | `1.00` | `0.00076575` | `1.00000` | 최고 |
| `0.01` | `0.99` | `0.00080889` | `1.00000` | 악화 |
| `0.02` | `0.98` | `0.00085235` | `1.00000` | 악화 |
| `0.05` | `0.95` | `0.00098338` | `1.00000` | 악화 |
| `0.10` | `0.90` | `0.00120919` | `1.00000` | 악화 |

판단:

- 현재는 블렌딩 이득이 없다.
- `hybrid`를 아주 낮은 비율로 섞어도 OOF logloss가 오히려 증가한다.
- 따라서 최종 제출 우선순위는 `physnet convnext_tiny 384` 단독이다.
