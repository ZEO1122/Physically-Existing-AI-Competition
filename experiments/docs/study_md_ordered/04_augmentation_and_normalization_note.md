# 데이터 증강과 정규화 정리

## 개요

이번 대회 실험에서는 `hybrid`와 `physnet` 두 계열 모델을 사용했다.
둘 다 `front.png`, `top.png` 두 뷰를 함께 사용하며, 가능한 한 두 뷰에 같은 변환을 적용해 멀티뷰 대응 관계가 깨지지 않도록 설계되어 있다.

핵심 차이는 다음과 같다.

- `hybrid`: 구조물 중심 crop, mask, geometry feature를 적극 활용
- `physnet`: motion pseudo-target과 동기화된 멀티뷰 증강을 활용

## 1. Hybrid의 데이터 증강과 정규화

기준 코드:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/train_hybrid_structure_aware_gated_simaux.py`

### 1-1. 구조물 중심 crop

- `crop_profile=mask_bbox`일 때 mask bounding box 기준 square crop을 적용
- margin을 포함해 crop한 뒤 최종 입력 크기로 resize

목적:

- 배경 비중 감소
- 구조물 실루엣과 형상에 집중
- train/dev/test 조명 및 배경 차이의 영향을 줄임

### 1-2. Resize

- RGB 이미지: `BICUBIC`
- mask 이미지: `NEAREST`
- 기본 입력 크기: `384 x 384`

목적:

- 이미지 정보는 부드럽게 리사이즈
- mask는 경계가 뭉개지지 않게 보존

### 1-3. Horizontal Flip

- 확률 `0.5`
- `front`, `top`, `front_mask`, `top_mask`에 동일 적용

목적:

- 좌우 방향 편향 완화
- 시점 편향 감소

### 1-4. Affine 변환

- 회전: `-6 ~ +6도`
- 이동: 이미지 크기의 최대 `5%`
- 스케일: `0.95 ~ 1.05`

목적:

- 촬영 위치 오차 대응
- 구조물 위치 및 크기 변화에 대한 강건성 확보

### 1-5. Photometric 변환

두 뷰에 같은 파라미터를 적용한다.

기본 `base` 프로필:

- brightness: 약 `±18%`
- contrast: 약 `±18%`
- saturation: 약 `±18%`
- hue: 약 `±0.03`

`strong_domain` 프로필:

- brightness: 약 `±32%`
- contrast: 약 `±28%`
- saturation: 약 `±30%`
- hue: 약 `±0.05`

`crop_tuned` 프로필:

- brightness: 약 `±24%`
- contrast: 약 `±22%`
- saturation: 약 `±20%`
- hue: 약 `±0.035`

목적:

- 조명 변화 대응
- train/dev/test 색감 차이 완화
- 밝기와 색온도 편차에 대한 일반화 향상

### 1-6. Gamma 변환

`strong_domain`:

- 확률 `0.55`
- gamma 범위 `0.75 ~ 1.45`

`crop_tuned`:

- 확률 `0.25`
- gamma 범위 `0.88 ~ 1.18`

목적:

- 카메라 감마 차이
- 노출 차이
- 명암 분포 차이 대응

### 1-7. Grayscale 변환

`strong_domain`:

- 확률 `0.12`

`crop_tuned`:

- 확률 `0.04`

목적:

- 색상 의존성 감소
- 구조와 형태 중심 학습 유도

### 1-8. Gaussian Blur

`base`:

- 확률 `0.12`
- radius `0.1 ~ 0.8`

`crop_tuned`:

- 확률 `0.14`
- radius `0.1 ~ 0.95`

`strong_domain`:

- 확률 `0.18`
- radius `0.1 ~ 1.2`

목적:

- 초점 흐림 대응
- 저해상도 입력에 대한 강건성 확보

### 1-9. Autocontrast

`base`: 확률 `0.10`

`crop_tuned`: 확률 `0.12`

`strong_domain`: 확률 `0.18`

목적:

- 명암 분포 변화 대응
- 조명 편차에 대한 민감도 완화

### 1-10. JPEG-like Degradation

`base`:

- 확률 `0.08`
- quality `48 ~ 82`

`crop_tuned`:

- 확률 `0.10`
- quality `42 ~ 80`

`strong_domain`:

- 확률 `0.14`
- quality `35 ~ 78`

목적:

- 압축 아티팩트 대응
- 데이터 품질 차이에 대한 일반화 향상

### 1-11. Normalize

- `TF.to_tensor`
- `ImageNet mean/std` 정규화

목적:

- pretrained backbone 입력 분포와 정렬
- 학습 안정화

### 1-12. Mask / Geometry 전처리

- mask를 `0/1` binary tensor로 변환
- mask에서 geometry feature 추출
- confidence와 mask 기반 meta feature 생성

목적:

- RGB만이 아니라 구조 정보도 함께 학습
- 실루엣, footprint, 높이 분포 등의 물리적 단서 반영

### 1-13. 추론 단계

- 평가 시에는 보통 `crop + resize + normalize`
- 추가 증강은 제거
- `tta_hflip` 사용 시 원본과 좌우반전 예측을 평균

목적:

- 예측 안정화
- 좌우 방향 편향 감소

## 2. PhysNet의 데이터 증강과 정규화

기준 코드:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/model_file/baseline_model/physnet_multiview_baseline.py`

### 2-1. Resize

- 입력 이미지 크기를 `img_size`로 조정
- 최근 주력 실험은 `384`

목적:

- 고정 크기 입력 구성
- backbone 입력 일관성 확보

### 2-2. Horizontal Flip

- 확률 `0.5`
- `front`, `top`에 동일 적용
- motion pseudo-target에도 동일 반전 적용

목적:

- 뷰 간 정합 유지
- 방향 편향 완화

### 2-3. Affine 변환

기본:

- 회전: `-8 ~ +8도`
- 이동: 이미지 크기의 최대 `5%`
- 스케일: `0.95 ~ 1.05`

강한 `strong` 프로필:

- 스케일 범위가 더 넓어짐: `0.92 ~ 1.08`

motion target에도 같은 affine을 적용한다.

목적:

- 이미지와 auxiliary target의 정합 유지
- 촬영 오차, 위치 변화 대응

### 2-4. Photometric 변환

기본:

- brightness: 약 `±15%`
- contrast: 약 `±15%`
- saturation: 약 `±15%`
- hue: 약 `±0.03`

`strong`:

- brightness: 약 `±25%`
- contrast: 약 `±25%`
- saturation: 약 `±25%`
- hue: 약 `±0.05`

목적:

- 조명 및 색감 변화에 대한 일반화 향상

### 2-5. Autocontrast

`strong`에서만:

- 확률 `0.10`

목적:

- 명암 변화 대응

### 2-6. Grayscale

`strong`에서만:

- 확률 `0.05`

목적:

- 색상 의존성 완화

### 2-7. Gaussian Blur

기본:

- 확률 `0.08`
- radius `0.1 ~ 0.7`

`strong`:

- 확률 `0.20`
- radius `0.2 ~ 1.2`

목적:

- 흐림, 초점 차이 대응

### 2-8. JPEG-like Degradation

`strong`에서만:

- 확률 `0.15`

목적:

- 압축 품질 변화 대응

### 2-9. Random Erasing

기본:

- 확률 `0.10`
- scale `0.02 ~ 0.10`

`strong`:

- 확률 `0.20`
- scale `0.02 ~ 0.12`

목적:

- 일부 영역 가림 대응
- 국소 텍스처 과적합 방지

### 2-10. Normalize

- `TF.to_tensor`
- `ImageNet mean/std` 정규화

목적:

- pretrained backbone 분포와 맞춤
- 학습 안정화

### 2-11. Motion pseudo-target 정렬

- `simulation.mp4` 기반 motion target 생성
- hflip / affine을 이미지와 같은 방식으로 적용

목적:

- auxiliary supervision 정합 유지
- 미래 물리 변화 정보를 함께 학습

### 2-12. 추론 단계

- `resize + normalize`
- `tta_hflip` 사용 시 원본/반전 평균

목적:

- 예측 안정성 향상

## 3. 입력 정규화 외의 regularization

아래 항목은 이미지 증강은 아니지만 일반화에 중요한 역할을 한다.

- dropout
- label smoothing
- weight decay

목적:

- 과적합 방지
- 과도한 확신 억제
- validation/generalization 성능 향상

## 4. 해석 요약

### Hybrid

핵심은 아래 두 가지다.

- 구조물 중심 crop
- mask 기반 geometry feature 활용

즉 단순 RGB 분류보다 구조와 형상에 더 집중하는 방향이다.

### PhysNet

핵심은 아래 두 가지다.

- 멀티뷰 동기화 증강
- motion pseudo-target과의 정합 유지

즉 정적 이미지 분류에 미래 물리 변화 힌트를 함께 학습시키는 방향이다.

## 5. 실전 관점 요약

- `hybrid`에서 가장 큰 효과를 준 것은 `mask_bbox crop`
- `strong_domain` 증강은 단독으로는 개선됐지만 crop과 단순 결합 시 오히려 악화
- `physnet`은 motion target과 증강 정합이 중요하므로, 이미지와 target에 같은 변환을 주는 점이 핵심
- 두 모델 모두 ImageNet normalize와 TTA hflip을 사용한다
