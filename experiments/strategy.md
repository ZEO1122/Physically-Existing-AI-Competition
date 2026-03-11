# 구조물 안정성 대회 전략

## 1. 문제 재정의

입력은 `front.png`, `top.png` 두 장뿐이지만, 학습용 `train`에는 `simulation.mp4`가 추가로 있다. 따라서 이 대회는 단순 이미지 분류보다:

- 정적 이미지에서 구조적 취약성을 읽는 능력
- `train`의 시뮬레이션 영상을 보조 감독으로 활용하는 능력
- `train`과 `dev/test` 사이의 조명/배경 도메인 차이를 버티는 능력

이 세 가지가 점수를 결정한다.

## 2. 실제 데이터에서 확인한 포인트

분석 대상 데이터 루트:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/open`

확인 결과:

- `train`: 1000개, 클래스 비율 `stable 500 / unstable 500`
- `dev`: 100개, 클래스 비율 `unstable 52 / stable 48`
- `test`: 1000개
- 모든 이미지 크기: `384 x 384`
- 모든 train 영상: `300 frame`, `30 fps`, 약 `10초`
- `train`이 `dev/test`보다 훨씬 밝다.
- 샘플링 분석에서 train 평균 밝기는 약 `0.87`, dev는 약 `0.77` 수준으로 확인됐다.
- train 영상의 frame-diff 기반 motion 통계는 라벨과 강하게 연결된다.
  - 300개 샘플 기준 `motion_mean`과 라벨 상관 약 `0.70`
  - unstable 쪽 motion_sum 평균이 stable보다 크게 높다.

결론:

- 색상 그 자체보다 silhouette, footprint, 중심축, occupancy 같은 구조 피처가 중요하다.
- `simulation.mp4`는 직접 추론에 쓰이지 않더라도 학습 시 강한 auxiliary supervision이다.

## 3. 권장 모델 우선순위

### A안. 메인 모델

`train_hybrid_structure_aware_gated_simaux.py`

이 모델을 1순위로 잡는 이유:

- front/top RGB를 둘 다 사용한다.
- 이미지에서 구조물 mask를 뽑아 geometry feature를 같이 쓴다.
- pseudo-3D visual hull 기반 구조 정보를 모델에 넣는다.
- simulation 기반 auxiliary target을 함께 학습한다.
- 조명 변화에 민감한 raw RGB만 쓰는 모델보다 dev/test 일반화 가능성이 높다.

### B안. 보조 모델

`physnet_multiview_baseline.py`

이 모델을 보조 앙상블 후보로 쓰는 이유:

- `simulation.mp4`에서 motion heatmap pseudo-target을 만들어 학습한다.
- 구조형 모델과 실패 패턴이 다를 가능성이 있다.
- holdout 결과가 비슷하면 soft-voting으로 logloss를 더 안정화할 수 있다.

## 4. 실전 운영 전략

### 4.1 검증 방식

- 초반 실험은 `holdout`으로 빠르게 반복한다.
- 후반에는 `cv`로 안정적인 OOF를 만든다.
- dev가 실제 평가 환경과 더 유사하므로, 최종 선택 기준은 train-only 성능보다 dev/holdout logloss에 둔다.

### 4.2 튜닝 우선순위

1. `hybrid` 모델에서 batch size와 epoch 수 안정화
2. `temperature scaling` 유지
3. `tta_hflip` 유지
4. `img_size=384` 유지
5. 과적합 시 `dropout`, `label_smoothing`, `patience` 조정

### 4.3 블렌딩 원칙

- 구조 인식 모델 0.6~0.8
- PhysNet 보조 모델 0.2~0.4
- 단순 평균보다 holdout logloss 기반 가중 평균이 낫다.

## 5. 추천 실행 플랜

1. `analyze_dataset.py`로 데이터 상태와 경로를 다시 검증한다.
2. `train-hybrid` 기본 설정으로 1회 학습한다.
3. 결과가 안정적이면 `train-physnet`도 실행한다.
4. 두 모델의 holdout logloss를 비교한다.
5. 우수 모델 단독 제출 또는 weighted blend 제출을 선택한다.

## 6. 주의할 점

- 사용자 설명의 `/opne`는 실제로 비어 있었고, 실데이터는 `data_reconstrcture/open`에 있다.
- 최종 코드 제출 대비를 위해 상대 경로 버전도 남겨 두는 것이 좋다.
- `torchvision` pretrained weight가 네트워크 없이 바로 안 받아질 수 있으므로, 스크립트 내 fallback 동작을 확인해야 한다.
