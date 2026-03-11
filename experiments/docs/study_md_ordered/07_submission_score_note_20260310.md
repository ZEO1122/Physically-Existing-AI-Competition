# 제출 점수 기록 (2026-03-10)

## 제출 파일과 점수

1. `gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv`
   - Public 점수: `0.0405710304`

2. `gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_weighted.csv`
   - Public 점수: `0.0633754883`

3. `hybrid_mask_bbox_margin010_cv_v1_20260309_194332__submission_cv_equal.csv`
   - Public 점수: `0.0823073731`

## 빠른 해석

- 현재 public 기준 최고 점수는 `physnet convnext tiny 384 cv equal`입니다.
- `cv weighted`는 `cv equal`보다 점수가 더 나빴습니다. 즉 inverse-calibrated fold weighting은 현재 리더보드에서 도움이 되지 않았습니다.
- `hybrid`는 완전히 배제할 모델은 아니지만, 현재 단독 제출 성능은 최고 `physnet`보다 뒤집니다.

## 현재 최우선 제출 파일

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909__submission_cv_equal.csv`

## 다음 실험 우선순위

1. 같은 모델, 다른 seed 앙상블
   - `physnet convnext tiny 384`를 3~5개 seed로 다시 학습합니다.
   - 각 seed의 CV test prediction을 평균합니다.
   - 현재 최고 모델이 이미 강하기 때문에, seed ensemble이 logloss를 더 안정적으로 낮출 가능성이 가장 큽니다.

2. 같은 모델, 더 강한 TTA 비교
   - 현재 모델은 그대로 둡니다.
   - `original + hflip`에 더해 resized crop variant나 약한 scale TTA를 비교합니다.
   - OOF 또는 dev logloss가 실제로 좋아지는 조합만 남깁니다.

3. ConvNeXt-Small 백본 실험
   - `physnet` 백본을 `convnext_tiny`에서 `convnext_small`로 키우고 입력 크기는 `384`를 유지합니다.
   - 현재 가장 잘 되는 계열을 자연스럽게 확장하는 실험입니다.

4. 입력 해상도 상향
   - 메모리가 허용하면 `physnet convnext tiny`를 `448` 또는 `512`로 실험합니다.
   - 구조 안정성 판단에서는 더 미세한 형상 단서가 도움이 될 수 있습니다.

5. motion pseudo-target 생성 방식 개선
   - 현재 `physnet`은 motion auxiliary supervision에 의존합니다.
   - simulation sampling 시점을 더 촘촘하게 하거나 motion aggregation 방식을 바꿔서, 후반 붕괴 신호를 더 잘 반영하도록 실험합니다.

6. hybrid는 단독 제출보다 보조 축으로 재정비
   - 현재는 `hybrid`를 더 개선한 뒤 다시 블렌딩을 검토하는 쪽이 맞습니다.
   - 지금 상태에서는 `physnet`이 훨씬 강해서 블렌딩 이득이 없었습니다.

## 실전 추천

- 다음으로 가장 유망한 방향은 완전히 새로운 구조를 찾는 것이 아닙니다.
- `physnet convnext tiny 384`를 기준으로 seed ensemble을 먼저 보고, 그 다음에 더 큰 ConvNeXt나 더 높은 해상도를 실험하는 것이 좋습니다.
