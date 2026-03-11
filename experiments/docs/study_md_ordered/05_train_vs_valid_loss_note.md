# Train Loss보다 Valid Loss가 더 낮게 보이는 이유

## 요약

이번 대회 실험처럼 강한 증강, 정규화, auxiliary loss가 들어간 설정에서는 `train loss > valid loss`가 충분히 정상적으로 나타날 수 있다.

핵심은 `train`과 `valid`가 같은 난이도로 계산되지 않는다는 점이다.

## 주요 원인

### 1. 학습 데이터 쪽이 더 어렵다

학습 중에는 이미지에 증강이 적용된다.

- horizontal flip
- affine transform
- color jitter
- blur
- JPEG-like degradation
- crop 기반 재구성

즉 `train loss`는 일부러 더 어려워진 샘플에서 계산된다.

반면 `valid loss`는 보통 `resize + normalize` 중심의 더 깨끗한 입력에서 계산된다.
그래서 같은 모델이라도 `train loss`가 더 높게 나올 수 있다.

### 2. 정규화 기법이 학습 loss를 올린다

현재 실험에는 아래 요소들이 들어간다.

- dropout
- label smoothing
- weight decay
- auxiliary loss

이런 요소들은 학습 중 모델이 지나치게 확신하지 않도록 만든다.
그 결과 `train` 단계에서는 loss가 더 높게 측정될 수 있다.

반대로 validation에서는 dropout이 꺼지고, 입력도 더 안정적이어서 loss가 더 낮아질 수 있다.

### 3. 기록 시점이 다르다

일반적으로:

- `train loss`: 한 epoch 동안 여러 batch의 평균
- `valid loss`: epoch 마지막 시점의 모델로 계산

즉 `train loss`에는 epoch 초반의 덜 학습된 상태가 섞여 있다.
반면 `valid loss`는 epoch 끝에서 업데이트가 반영된 모델로 계산된다.

그래서 같은 epoch 번호라도 `valid loss`가 더 낮게 보일 수 있다.

### 4. validation fold가 상대적으로 쉬울 수 있다

CV나 holdout에서 특정 fold는 현재 모델이 잘 맞추는 샘플 비율이 높을 수 있다.
이 경우 validation 성능이 비정상적으로 잘 보일 수 있다.

이번 실험에서도 일부 fold는 calibrated logloss가 매우 낮게 나왔다.
이럴 때는 단순히 모델이 그 validation split에 잘 맞았을 가능성이 있다.

## 이번 대회 실험에서 특히 해당되는 이유

이번 `hybrid` 실험에서는 아래 요인이 동시에 작용한다.

- 강한 데이터 증강
- `mask_bbox` crop 전처리
- dropout
- label smoothing
- geometry / simulation auxiliary loss
- calibration 기준 logloss 확인

그래서 `train loss > valid loss`는 충분히 자연스럽다.

## 언제는 의심해야 하는가

아래 경우에는 단순 현상으로 넘기지 말고 점검하는 편이 맞다.

- `valid loss`가 지나치게 0에 가깝다
- fold 간 편차가 매우 크다
- holdout에서는 성능이 안 좋은데 CV만 비정상적으로 좋다
- 데이터 leakage 가능성이 있다

즉 중요한 것은 `train loss`와 `valid loss`의 상대 크기 자체보다,
`fold 일관성`, `holdout 일치성`, `재현성`이다.

## 현재 해석

현재 기준으로는 이 현상을 바로 문제로 볼 이유는 크지 않다.

이유:

- holdout에서도 crop tuning 방향이 개선됐다
- CV에서도 같은 방향으로 점수가 좋아졌다
- 증강과 regularization이 실제로 강하게 들어가 있다

따라서 현재의 `train loss > valid loss`는
`학습 데이터가 더 어렵게 구성된 결과`로 해석하는 것이 가장 타당하다.
