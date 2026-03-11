# Experiment Log

이 문서는 `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test`에서 수행한 분석과 실험 기록을 Markdown으로 누적 저장하는 로그다.

기준 시각:

- 2026-03-09 KST

## 1. 데이터 경로 확인

사용자 설명의 `/opne` 경로는 실제로 비어 있었고, 실데이터는 아래 경로에서 확인했다.

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/open`

구성 확인:

- `train`: 1000개
- `dev`: 100개
- `test`: 1000개
- train sample별 파일: `front.png`, `top.png`, `simulation.mp4`
- dev/test sample별 파일: `front.png`, `top.png`

## 2. 환경 확인

기본 샌드박스 환경에서는 GPU 접근이 되지 않았다.

- 샌드박스 내부:
  - `nvidia-smi` 실패
  - `torch.cuda.is_available() == False`
  - 에러: `cudaGetDeviceCount ... Error 304`

권한 상승 후 동일 머신에서 GPU 접근 가능함을 확인했다.

- GPU: `NVIDIA GeForce RTX 4070 Ti SUPER`
- Driver Version: `590.48.01`
- CUDA Version: `13.1`
- `multiview` conda 환경에서 `torch.cuda.is_available() == True`
- CUDA tensor 생성 정상

## 3. 데이터 분석 결과

분석 스크립트:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analyze_dataset.py`

분석 리포트:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analysis/dataset_report.json`

핵심 요약:

- `train` 라벨 분포: `unstable 500`, `stable 500`
- `dev` 라벨 분포: `unstable 52`, `stable 48`
- 모든 이미지 크기: `384 x 384`
- 모든 train 영상: `300 frame`, `30 fps`

밝기 분포 차이:

- train front 평균 밝기: `0.8548`
- dev front 평균 밝기: `0.7621`
- test front 평균 밝기: `0.7609`
- train top 평균 밝기: `0.8840`
- dev top 평균 밝기: `0.7812`
- test top 평균 밝기: `0.7818`

해석:

- `train`과 `dev/test` 사이에 명확한 조명/배경 도메인 차이가 있다.
- 따라서 raw RGB만 과신하면 일반화 성능이 흔들릴 가능성이 높다.
- silhouette, footprint, geometry, pseudo-3D 성분이 중요하다.

train video motion 통계:

- stable 평균 `motion_mean`: `0.0191`
- unstable 평균 `motion_mean`: `0.1556`
- stable 평균 `motion_sum`: `5.7041`
- unstable 평균 `motion_sum`: `46.5298`
- `motion_mean`과 unstable 라벨 상관: `0.7018`
- `motion_q95`와 unstable 라벨 상관: `0.6862`

해석:

- `simulation.mp4`는 강한 auxiliary supervision으로 활용할 가치가 충분하다.

## 4. 모델 전략

1순위 메인 모델:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/data_reconstrcture/train_hybrid_structure_aware_gated_simaux.py`

선정 이유:

- front/top RGB 동시 사용
- 구조물 mask 추출
- geometry feature 사용
- pseudo-3D visual hull 기반 구조 정보 사용
- simulation auxiliary target 사용

2순위 보조 모델:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/model_file/baseline_model/physnet_multiview_baseline.py`

선정 이유:

- `simulation.mp4` 기반 motion pseudo-target 사용
- hybrid와 다른 실패 패턴을 가지는 앙상블 후보

## 5. 실험 기록

### 5.1 Smoke Run

목적:

- `multiview` 환경에서 학습 스크립트 진입과 경로 확인

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name smoke_hybrid --epochs 1 --batch_size 4 --num_workers 0
```

결과:

- GPU 미사용 상태에서 CPU로 진입 확인
- 출력 디렉토리 생성 확인
- 산출물:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/smoke_hybrid_20260309_181141/run_config.json`

### 5.2 GPU Holdout Run

상태:

- 완료

실행 환경:

- conda env: `multiview`
- device: `cuda`
- AMP: `True`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name gpu_hybrid_holdout_v1 --epochs 14 --batch_size 12 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.52526 | 0.93323 | 0.93323 | 0.62551 | 0.68000 | 3.681 |
| 2 | 0.20706 | 0.79459 | 0.79459 | 0.60967 | 0.65000 | 2.812 |
| 3 | 0.17887 | 1.12152 | 1.12152 | 0.62435 | 0.66000 | 4.539 |
| 4 | 0.11285 | 1.00915 | 1.00915 | 0.57377 | 0.72000 | 3.681 |
| 5 | 0.10852 | 1.06700 | 1.06700 | 0.58071 | 0.73000 | 3.972 |
| 6 | 0.11581 | 0.83372 | 0.83372 | 0.52393 | 0.77000 | 2.978 |
| 7 | 0.09350 | 0.91558 | 0.91558 | 0.53776 | 0.75000 | 3.281 |
| 8 | 0.08509 | 0.86224 | 0.86224 | 0.51878 | 0.79000 | 3.065 |
| 9 | 0.08668 | 0.89754 | 0.89754 | 0.53815 | 0.75000 | 3.247 |
| 10 | 0.08602 | 0.86871 | 0.86871 | 0.52797 | 0.75000 | 3.122 |
| 11 | 0.08898 | 0.89934 | 0.89934 | 0.53215 | 0.75000 | 3.213 |
| 12 | 0.08336 | 1.12721 | 1.12721 | 0.58897 | 0.71000 | 4.251 |

최종 해석:

- 학습은 정상적으로 GPU에서 진행 중이다.
- calibration 전 raw score는 흔들리지만, temperature scaling 후 점수는 꾸준히 개선됐다.
- 현재까지 best calibrated valid logloss는 `0.51878`이다.
- 현재까지 best accuracy는 `0.79000`이다.
- early stopping이 `epoch 12`에서 발동했다.
- best epoch는 `epoch 8`이다.

최종 요약:

- `valid_logloss_cal`: `0.5187817784923585`
- `valid_accuracy`: `0.79`
- best checkpoint:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/best_gpu_hybrid_holdout_v1.pt`
- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/valid_predictions_gpu_hybrid_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/test_predictions_gpu_hybrid_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/epochs_gpu_hybrid_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454/holdout_summary.json`

### 5.3 GPU PhysNet Holdout Run

상태:

- 완료

실행 환경:

- conda env: `multiview`
- device: `cuda`
- AMP: `True`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode holdout --run_name gpu_physnet_holdout_v1 --epochs 16 --batch_size 24 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.66249 | 0.56271 | 0.56271 | 0.52510 | 0.75000 | 1.598 |
| 2 | 0.42295 | 1.02607 | 1.02607 | 0.60137 | 0.72000 | 3.930 |
| 3 | 0.26819 | 1.05755 | 1.05755 | 0.59039 | 0.73000 | 4.045 |
| 4 | 0.21862 | 1.09224 | 1.09224 | 0.60353 | 0.70000 | 4.285 |
| 5 | 0.16522 | 0.70105 | 0.70105 | 0.45928 | 0.82000 | 2.627 |
| 6 | 0.17006 | 0.75586 | 0.75586 | 0.49708 | 0.82000 | 2.760 |
| 7 | 0.16052 | 0.75841 | 0.75841 | 0.49234 | 0.79000 | 2.760 |
| 8 | 0.13567 | 0.77412 | 0.77412 | 0.48940 | 0.79000 | 2.841 |
| 9 | 0.13218 | 0.95213 | 0.95213 | 0.54602 | 0.74000 | 3.503 |
| 10 | 0.12102 | 0.89732 | 0.89732 | 0.52516 | 0.75000 | 3.281 |

최종 해석:

- early stopping이 `epoch 10`에서 발동했다.
- best epoch는 `epoch 5`다.
- best calibrated valid logloss는 `0.459281193032132`다.
- best accuracy는 `0.82`다.
- 현재까지 수행한 두 개의 holdout 실험 중에서는 PhysNet이 더 좋다.

최종 요약:

- `valid_logloss_cal`: `0.459281193032132`
- `valid_accuracy`: `0.82`
- best checkpoint:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/best_gpu_physnet_holdout_v1.pt`
- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/valid_predictions_gpu_physnet_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/test_predictions_gpu_physnet_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/epochs_gpu_physnet_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302/holdout_summary.json`

### 5.4 GPU PhysNet ResNet-50 384 Holdout Run

상태:

- 완료

변경점:

- backbone: `resnet34 -> resnet50`
- image size: `224 -> 384`
- motion target size: `56 -> 96`
- backbone lr: `5e-5 -> 3e-5`

코드 수정:

- `physnet_multiview_baseline.py`에 `--backbone_name {resnet34,resnet50}` 옵션 추가
- `img_size != motion_size*4` 상황에서도 motion auxiliary loss가 동작하도록 resolution mismatch 보정 추가

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode holdout --run_name gpu_physnet_r50_384_v1 --backbone_name resnet50 --img_size 384 --motion_size 96 --batch_size 8 --epochs 16 --num_workers 4 --backbone_lr 3e-5
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.68390 | 1.07459 | 1.07459 | 0.67235 | 0.58000 | 6.146 |
| 2 | 0.42512 | 0.60865 | 0.60865 | 0.60082 | 0.71000 | 1.344 |
| 3 | 0.24672 | 1.53260 | 1.53260 | 0.68265 | 0.54000 | 7.500 |
| 4 | 0.18269 | 0.40369 | 0.40369 | 0.40320 | 0.84000 | 1.068 |
| 5 | 0.16539 | 0.71105 | 0.71105 | 0.53344 | 0.79000 | 2.606 |
| 6 | 0.13374 | 0.88989 | 0.88989 | 0.56951 | 0.75000 | 3.403 |
| 7 | 0.13132 | 0.85704 | 0.85704 | 0.57038 | 0.71000 | 3.097 |
| 8 | 0.13968 | 0.51193 | 0.51193 | 0.45371 | 0.79000 | 1.724 |
| 9 | 0.14322 | 0.85501 | 0.85501 | 0.58474 | 0.70000 | 3.247 |

최종 해석:

- early stopping이 `epoch 9`에서 발동했다.
- best epoch는 `epoch 4`다.
- best calibrated valid logloss는 `0.40319519115848723`다.
- best accuracy는 `0.84`다.
- 현재까지 수행한 모든 holdout 실험 중 최고 성능이다.

최종 요약:

- `valid_logloss_cal`: `0.40319519115848723`
- `valid_accuracy`: `0.84`
- best checkpoint:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/best_gpu_physnet_r50_384_v1.pt`
- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/valid_predictions_gpu_physnet_r50_384_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/test_predictions_gpu_physnet_r50_384_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/epochs_gpu_physnet_r50_384_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409/holdout_summary.json`

## 7. 현재 기준 비교

| Model | Run | Best CalLL | Best Acc | Comment |
|---|---|---:|---:|---|
| Hybrid structure-aware + sim aux | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 구조 prior가 강점이지만 이번 holdout에서는 PhysNet보다 뒤짐 |
| PhysNet motion-aux baseline | `gpu_physnet_holdout_v1` | `0.45928` | `0.82` | 현재까지 최고 성능 |
| PhysNet ResNet-50 384 | `gpu_physnet_r50_384_v1` | `0.40320` | `0.84` | 현재 최상 성능, 제출 1순위 후보 |

### 7.1 PhysNet ResNet-50 384 CV

상태:

- 완료

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode cv --run_name gpu_physnet_r50_384_cv_v1 --backbone_name resnet50 --img_size 384 --motion_size 96 --batch_size 8 --epochs 16 --nfolds 5 --num_workers 4 --backbone_lr 3e-5
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716`

최종 요약:

- `oof_logloss_cal`: `0.07421408550876968`
- `oof_accuracy`: `0.98`
- dev-only metric inside OOF:
  - `logloss`: `0.2650819928648954`
  - `accuracy`: `0.92`
- train-only metric inside OOF:
  - `logloss`: `0.0551272947731571`
  - `accuracy`: `0.986`

fold별 best calibrated logloss:

| Fold | Best Epoch | CalLL | Acc |
|---:|---:|---:|---:|
| 0 | 14 | 0.02353 | 0.99545 |
| 1 | 8 | 0.10931 | 0.95909 |
| 2 | 5 | 0.09478 | 0.98636 |
| 3 | 12 | 0.11829 | 0.97273 |
| 4 | 10 | 0.02516 | 0.98636 |

산출물:

- CV summary:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716/cv_summary.json`
- fold metrics:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716/fold_metrics.csv`
- OOF predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716/oof_predictions.csv`
- submission equal:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716/submission_cv_equal.csv`
- submission weighted:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716/submission_cv_weighted.csv`

### 7.2 Hybrid CV

상태:

- 완료

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode cv --run_name gpu_hybrid_cv_v1 --epochs 14 --batch_size 12 --nfolds 5 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043`

최종 요약:

- `oof_logloss_cal`: `0.021510442103023657`
- `oof_accuracy`: `0.9936363636363637`
- dev-only metric inside OOF:
  - `logloss`: `0.11808185134315995`
  - `accuracy`: `0.95`
- train-only metric inside OOF:
  - `logloss`: `0.011853301179010027`
  - `accuracy`: `0.998`

fold별 best calibrated logloss:

| Fold | Best Epoch | CalLL | Acc |
|---:|---:|---:|---:|
| 0 | 11 | 0.00262 | 1.00000 |
| 1 | 4 | 0.02908 | 0.99545 |
| 2 | 6 | 0.00520 | 0.99545 |
| 3 | 8 | 0.05156 | 0.98636 |
| 4 | 2 | 0.01910 | 0.99091 |

산출물:

- CV summary:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043/cv_summary.json`
- fold metrics:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043/fold_metrics.csv`
- OOF predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043/oof_predictions.csv`
- submission equal:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043/submission_cv_equal.csv`
- submission weighted:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043/submission_cv_weighted.csv`

### 7.3 CV 비교

| Model | Run | OOF CalLL | OOF Acc | Dev-only OOF LogLoss | Comment |
|---|---|---:|---:|---:|---|
| PhysNet ResNet-50 384 CV | `gpu_physnet_r50_384_cv_v1` | `0.07421` | `0.98000` | `0.26508` | 강하지만 dev 구간 편차가 있음 |
| Hybrid CV | `gpu_hybrid_cv_v1` | `0.02151` | `0.99364` | `0.11808` | 현재 CV 기준 최고 성능 |

### 7.4 Hybrid Strong Domain Aug Holdout

상태:

- 완료

변경점:

- `strong_domain` 증강 프로필 추가
- brightness / contrast / saturation / hue 증강 범위 확대
- gamma jitter 추가
- grayscale 확률성 추가
- blur / autocontrast / jpeg degradation 확률 강화

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name hybrid_strong_domain_aug_holdout_v1 --aug_profile strong_domain --epochs 14 --batch_size 12 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.57142 | 1.26906 | 1.26906 | 0.68491 | 0.53000 | 7.500 |
| 2 | 0.22234 | 0.64516 | 0.64516 | 0.48179 | 0.81000 | 2.233 |
| 3 | 0.12909 | 0.78004 | 0.78004 | 0.55443 | 0.75000 | 2.732 |
| 4 | 0.22729 | 1.01377 | 1.01377 | 0.57266 | 0.74000 | 3.681 |
| 5 | 0.12620 | 0.82533 | 0.82533 | 0.51815 | 0.79000 | 3.681 |
| 6 | 0.09178 | 0.68275 | 0.68275 | 0.45107 | 0.81000 | 2.506 |
| 7 | 0.10441 | 0.99539 | 0.99539 | 0.56065 | 0.74000 | 3.605 |
| 8 | 0.09366 | 0.73497 | 0.73497 | 0.47385 | 0.80000 | 2.654 |
| 9 | 0.08898 | 0.78241 | 0.78241 | 0.49133 | 0.78000 | 2.812 |
| 10 | 0.08802 | 0.63183 | 0.63183 | 0.43560 | 0.82000 | 2.366 |
| 11 | 0.08550 | 0.61172 | 0.61172 | 0.42629 | 0.82000 | 2.323 |
| 12 | 0.08983 | 0.86048 | 0.86048 | 0.52141 | 0.75000 | 3.097 |
| 13 | 0.08682 | 0.67525 | 0.67525 | 0.45598 | 0.82000 | 2.480 |
| 14 | 0.08180 | 0.63720 | 0.63720 | 0.43572 | 0.83000 | 2.390 |

최종 해석:

- best calibrated valid logloss는 `0.4262861996711964`다.
- best accuracy는 `0.82`다.
- best epoch는 `epoch 11`이다.
- baseline hybrid holdout보다 domain shift 대응이 개선됐다.

기존 hybrid holdout 대비:

| Model | Run | Valid CalLL | Valid Acc | Comment |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 기본 증강 |
| Hybrid strong domain aug | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명/도메인 강건 증강 적용 |

개선폭:

- calibrated logloss: `-0.09250`
- accuracy: `+0.03`

산출물:

- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330/valid_predictions_hybrid_strong_domain_aug_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330/test_predictions_hybrid_strong_domain_aug_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330/epochs_hybrid_strong_domain_aug_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330/holdout_summary.json`

### 7.5 Hybrid Mask-BBox Crop Holdout

상태:

- 완료

변경점:

- `front/top` 각각에서 mask bounding box 기준 square crop 적용
- margin 포함 crop 후 `384 x 384` resize
- photometric augmentation은 baseline과 동일하게 유지

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name hybrid_mask_bbox_holdout_v1 --crop_profile mask_bbox --epochs 14 --batch_size 12 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.47051 | 0.72756 | 0.72756 | 0.52545 | 0.77000 | 2.506 |
| 2 | 0.19445 | 0.40542 | 0.40542 | 0.35919 | 0.86000 | 1.565 |
| 3 | 0.10957 | 0.72094 | 0.72094 | 0.47741 | 0.80000 | 2.579 |
| 4 | 0.08830 | 0.70479 | 0.70479 | 0.44975 | 0.83000 | 2.579 |
| 5 | 0.09065 | 0.63037 | 0.63037 | 0.42471 | 0.83000 | 2.366 |
| 6 | 0.08950 | 0.62345 | 0.62345 | 0.42826 | 0.83000 | 2.341 |

최종 해석:

- best calibrated valid logloss는 `0.3591907615391547`다.
- best accuracy는 `0.86`이다.
- early stopping은 `epoch 6`에서 발동했다.
- best epoch는 `epoch 2`다.
- 현재까지 수행한 hybrid holdout 계열 중 최고 성능이다.

기존 hybrid 계열과 비교:

| Model | Run | Valid CalLL | Valid Acc | Comment |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 기본 증강 |
| Hybrid strong domain aug | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명/도메인 강건 증강 |
| Hybrid mask-bbox crop | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 구조물 중심 crop 정규화 |

개선폭:

- baseline 대비 calibrated logloss: `-0.15959`
- baseline 대비 accuracy: `+0.07`
- strong-domain 대비 calibrated logloss: `-0.06710`
- strong-domain 대비 accuracy: `+0.04`

산출물:

- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149/valid_predictions_hybrid_mask_bbox_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149/test_predictions_hybrid_mask_bbox_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149/epochs_hybrid_mask_bbox_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149/holdout_summary.json`

### 7.6 Hybrid Mask-BBox + Strong Domain Holdout

상태:

- 완료

변경점:

- `mask_bbox` crop 유지
- `strong_domain` photometric augmentation 추가 적용

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name hybrid_mask_bbox_strong_domain_holdout_v1 --crop_profile mask_bbox --aug_profile strong_domain --epochs 14 --batch_size 12 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.50459 | 0.75430 | 0.75430 | 0.55376 | 0.75000 | 2.579 |
| 2 | 0.15117 | 0.84736 | 0.84736 | 0.53785 | 0.75000 | 3.009 |
| 3 | 0.09886 | 0.63388 | 0.63388 | 0.44980 | 0.83000 | 2.323 |
| 4 | 0.15845 | 0.92758 | 0.92758 | 0.55197 | 0.75000 | 3.307 |
| 5 | 0.12347 | 0.78854 | 0.78854 | 0.50829 | 0.78000 | 2.782 |
| 6 | 0.10030 | 0.71143 | 0.71143 | 0.46585 | 0.81000 | 2.579 |
| 7 | 0.09797 | 0.73774 | 0.73774 | 0.47401 | 0.80000 | 2.654 |

최종 해석:

- best calibrated valid logloss는 `0.44979716444498696`다.
- best accuracy는 `0.83`이다.
- early stopping은 `epoch 7`에서 발동했다.
- best epoch는 `epoch 3`다.
- `mask_bbox` 단독 실험보다 성능이 나빠졌다.

Hybrid holdout 비교 갱신:

| Model | Run | Valid CalLL | Valid Acc | Comment |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 기본 증강 |
| Hybrid strong domain aug | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명/도메인 강건 증강 |
| Hybrid mask-bbox crop | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 현재 hybrid holdout 최고 |
| Hybrid mask-bbox + strong domain | `hybrid_mask_bbox_strong_domain_holdout_v1` | `0.44980` | `0.83` | 조합 시 오히려 악화 |

개선/악화폭:

- mask-bbox 대비 calibrated logloss: `+0.09061`
- mask-bbox 대비 accuracy: `-0.03`

산출물:

- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541/valid_predictions_hybrid_mask_bbox_strong_domain_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541/test_predictions_hybrid_mask_bbox_strong_domain_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541/epochs_hybrid_mask_bbox_strong_domain_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541/holdout_summary.json`

### 7.7 Hybrid Mask-BBox + Crop-Tuned Aug Holdout

상태:

- 완료

변경점:

- `mask_bbox` crop 유지
- `strong_domain`보다 약한 중간 세기의 photometric augmentation 적용
- gamma / grayscale / jpeg degradation 강도를 완화

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode holdout --run_name hybrid_mask_bbox_crop_tuned_holdout_v1 --crop_profile mask_bbox --aug_profile crop_tuned --epochs 14 --batch_size 12 --num_workers 4
```

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028`

관찰된 epoch 로그:

| Epoch | TrainLoss | ValidBCE | RawLL | CalLL | Acc | Temp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.49243 | 1.29670 | 1.29670 | 0.66462 | 0.62000 | 6.927 |
| 2 | 0.14705 | 0.96689 | 0.96689 | 0.55274 | 0.75000 | 3.475 |
| 3 | 0.10191 | 0.90690 | 0.90690 | 0.57593 | 0.71000 | 3.247 |
| 4 | 0.11704 | 0.92039 | 0.92039 | 0.52724 | 0.75000 | 3.281 |
| 5 | 0.10900 | 0.54201 | 0.54201 | 0.41520 | 0.83000 | 2.027 |
| 6 | 0.09837 | 0.59548 | 0.59548 | 0.41964 | 0.83000 | 2.257 |
| 7 | 0.10133 | 0.66973 | 0.66973 | 0.44193 | 0.82000 | 2.480 |
| 8 | 0.08576 | 0.79351 | 0.79351 | 0.48510 | 0.79000 | 2.841 |
| 9 | 0.09079 | 0.82183 | 0.82183 | 0.49700 | 0.79000 | 2.924 |

최종 해석:

- best calibrated valid logloss는 `0.41519983160023016`다.
- best accuracy는 `0.83`이다.
- early stopping은 `epoch 9`에서 발동했다.
- best epoch는 `epoch 5`다.
- `mask_bbox` 단독보다 성능이 나쁘고, `mask_bbox + strong_domain`보다는 낫다.

Hybrid holdout 비교 최종판:

| Model | Run | Valid CalLL | Valid Acc | Comment |
|---|---|---:|---:|---|
| Hybrid baseline | `gpu_hybrid_holdout_v1` | `0.51878` | `0.79` | 기본 증강 |
| Hybrid strong domain aug | `hybrid_strong_domain_aug_holdout_v1` | `0.42629` | `0.82` | 조명/도메인 강건 증강 |
| Hybrid mask-bbox crop | `hybrid_mask_bbox_holdout_v1` | `0.35919` | `0.86` | 현재 hybrid holdout 최고 |
| Hybrid mask-bbox + crop_tuned | `hybrid_mask_bbox_crop_tuned_holdout_v1` | `0.41520` | `0.83` | 약화 증강 재설계, 2위 |
| Hybrid mask-bbox + strong domain | `hybrid_mask_bbox_strong_domain_holdout_v1` | `0.44980` | `0.83` | 조합 시 악화 |

산출물:

- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028/submission_holdout.csv`
- valid predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028/valid_predictions_hybrid_mask_bbox_crop_tuned_holdout_v1.csv`
- test predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028/test_predictions_hybrid_mask_bbox_crop_tuned_holdout_v1.csv`
- epoch history:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028/epochs_hybrid_mask_bbox_crop_tuned_holdout_v1.csv`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028/holdout_summary.json`

## 6. 산출물 목록

- 전략 문서: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/strategy.md`
- 실행 가이드: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/README.md`
- 분석 스크립트: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analyze_dataset.py`
- 실행 runner: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py`
- 분석 리포트: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/analysis/dataset_report.json`
- 실험 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/experiment_log.md`
- hybrid run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_holdout_v1_20260309_181454`
- physnet run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_holdout_v1_20260309_182302`
- physnet r50 384 run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_v1_20260309_183409`
- physnet r50 384 cv run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r50_384_cv_v1_20260309_183716`
- hybrid cv run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/gpu_hybrid_cv_v1_20260309_185043`
- hybrid strong domain aug run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_strong_domain_aug_holdout_v1_20260309_191330`
- hybrid mask bbox run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_holdout_v1_20260309_192149`
- hybrid mask bbox + strong domain run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_strong_domain_holdout_v1_20260309_192541`
- hybrid mask bbox + crop tuned run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_crop_tuned_holdout_v1_20260309_193028`

## 7. Hybrid crop margin 튜닝과 CV

### 7-1. Holdout margin sweep

목적:

- `mask_bbox` crop의 배경 포함 비율을 줄여 hybrid holdout/CV를 추가 개선

설정:

- `crop_profile=mask_bbox`
- `crop_min_side_ratio=0.42`
- `crop_margin_ratio`만 변경

비교 결과:

| Margin | Run | Valid CalLL | Valid Acc |
|---:|---|---:|---:|
| `0.10` | `hybrid_mask_bbox_margin010_holdout_v1` | `0.3586387311335207` | `0.87` |
| `0.18` | `hybrid_mask_bbox_holdout_v1` | `0.3591907615391547` | `0.86` |
| `0.26` | `hybrid_mask_bbox_margin026_holdout_v1` | `0.37306904530299706` | `0.87` |

해석:

- best holdout margin은 `0.10`이다.
- `0.18 -> 0.10` 조정으로 calibrated logloss가 소폭 개선됐고 accuracy도 `0.86 -> 0.87`로 상승했다.
- margin을 너무 크게 늘리면 crop 이점이 줄어들며 성능이 떨어졌다.

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_holdout_v1_20260309_193756`
- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin026_holdout_v1_20260309_194036`

### 7-2. CV 실행

실험명:

- `hybrid_mask_bbox_margin010_cv_v1`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-hybrid --mode cv --run_name hybrid_mask_bbox_margin010_cv_v1 --crop_profile mask_bbox --crop_margin_ratio 0.10 --crop_min_side_ratio 0.42 --epochs 14 --batch_size 12 --nfolds 5 --num_workers 4
```

최종 결과:

- `oof_logloss_cal`: `0.015276939762022709`
- `oof_accuracy`: `0.9972727272727273`
- `dev-only oof logloss`: `0.09747774026268885`
- `fold_mean_temperature`: `0.5787879634236087`

fold별 결과:

| Fold | Best Epoch | Valid CalLL | Valid Acc | Temp |
|---:|---:|---:|---:|---:|
| 0 | 14 | `0.00001272` | `1.00000` | `0.3333` |
| 1 | 5 | `0.02876540` | `0.99545` | `0.7780` |
| 2 | 4 | `0.02057262` | `0.99545` | `0.6934` |
| 3 | 6 | `0.02702466` | `0.99545` | `0.7559` |
| 4 | 3 | `0.00000930` | `1.00000` | `0.3333` |

기존 hybrid CV와 비교:

| Model | Run | OOF CalLL | OOF Acc | Dev-only OOF Logloss |
|---|---|---:|---:|---:|
| Hybrid baseline CV | `gpu_hybrid_cv_v1` | `0.021510442103023657` | `0.9936363636363637` | `0.11808185134315995` |
| Hybrid mask-bbox margin `0.10` CV | `hybrid_mask_bbox_margin010_cv_v1` | `0.015276939762022709` | `0.9972727272727273` | `0.09747774026268885` |

최종 해석:

- crop margin `0.10` 튜닝은 CV 기준에서도 기존 hybrid baseline보다 개선됐다.
- 현재 hybrid 계열 최고 CV 설정은 `mask_bbox + crop_margin_ratio=0.10`이다.
- holdout과 CV가 같은 방향으로 개선됐기 때문에, hybrid 최종 제출 후보는 이 설정을 우선 고려하는 것이 타당하다.

산출물:

- CV run dir:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/cv_summary.json`
- fold metrics:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/fold_metrics.csv`
- OOF predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/oof_predictions.csv`
- submission equal:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/submission_cv_equal.csv`
- submission weighted:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/submission_cv_weighted.csv`

## 8. PhysNet ResNet101 백본 검증

목적:

- 기존 `physnet resnet50 384` 대비 더 큰 동일 계열 backbone이 성능 개선을 주는지 확인

코드 변경:

- `physnet_multiview_baseline.py`에 `resnet101` backbone 지원 추가
- `run_competition_pipeline.py`의 `train-physnet` CLI에도 `resnet101` 선택지 추가

실험명:

- `gpu_physnet_r101_384_v1`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode holdout --run_name gpu_physnet_r101_384_v1 --backbone_name resnet101 --img_size 384 --motion_size 96 --batch_size 10 --epochs 16 --num_workers 4
```

최종 결과:

- `valid_logloss_cal`: `0.5162458024837846`
- `valid_accuracy`: `0.7`
- early stopping: `epoch 7`

중간 경향:

| Epoch | TrainLoss | ValidBCE | CalLL | Acc |
|---:|---:|---:|---:|---:|
| 1 | `0.67362` | `0.63839` | `0.60753` | `0.67` |
| 2 | `0.35248` | `0.58849` | `0.51625` | `0.70` |
| 3 | `0.27817` | `0.70196` | `0.54919` | `0.71` |
| 4 | `0.17224` | `1.22326` | `0.65895` | `0.53` |
| 5 | `0.16217` | `0.85546` | `0.56798` | `0.68` |
| 6 | `0.12103` | `0.75732` | `0.54743` | `0.71` |
| 7 | `0.11978` | `1.08273` | `0.61960` | `0.64` |

기존 physnet 결과와 비교:

| Model | Run | Valid CalLL | Valid Acc |
|---|---|---:|---:|
| PhysNet ResNet34 224 | `gpu_physnet_holdout_v1` | `0.459281193032132` | `0.82` |
| PhysNet ResNet50 384 | `gpu_physnet_r50_384_v1` | `0.40319519115848723` | `0.84` |
| PhysNet ResNet101 384 | `gpu_physnet_r101_384_v1` | `0.5162458024837846` | `0.70` |

최종 해석:

- `ResNet101`은 현재 설정에서는 `ResNet50 384`보다 명확히 나빴다.
- backbone을 키우는 것만으로는 성능이 오르지 않았고, 오히려 validation이 더 불안정했다.
- 현재 physnet 계열 최적 holdout 후보는 여전히 `ResNet50 384`다.

산출물:

- run dir:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r101_384_v1_20260309_200647`
- summary:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r101_384_v1_20260309_200647/holdout_summary.json`
- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_r101_384_v1_20260309_200647/submission_holdout.csv`

## 9. PhysNet ConvNeXt-Tiny 백본 검증

목적:

- `physnet` backbone을 ResNet 계열에서 벗어나 `ConvNeXt-Tiny`로 교체했을 때 holdout 성능 개선 여부 확인

코드 변경:

- `physnet_multiview_baseline.py`에 `convnext_tiny` backbone 지원 추가
- `run_competition_pipeline.py`의 `train-physnet` CLI에도 `convnext_tiny` 선택지 추가
- `ConvNeXt-Tiny`는 `base.features`를 backbone feature extractor로 사용

실험명:

- `gpu_physnet_convnext_tiny_384_v1`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode holdout --run_name gpu_physnet_convnext_tiny_384_v1 --backbone_name convnext_tiny --img_size 384 --motion_size 96 --batch_size 8 --epochs 16 --num_workers 4
```

최종 결과:

- `valid_logloss_cal`: `0.3033521122607969`
- `valid_accuracy`: `0.86`
- early stopping: `epoch 7`
- best calibrated epoch: `epoch 2`

중간 경향:

| Epoch | TrainLoss | ValidBCE | CalLL | Acc |
|---:|---:|---:|---:|---:|
| 1 | `0.57949` | `0.75115` | `0.63178` | `0.64` |
| 2 | `0.34203` | `0.30412` | `0.30335` | `0.86` |
| 3 | `0.19405` | `0.54219` | `0.45111` | `0.81` |
| 4 | `0.15757` | `0.50375` | `0.41353` | `0.83` |
| 5 | `0.14026` | `0.92173` | `0.57950` | `0.69` |
| 6 | `0.12959` | `0.48208` | `0.41123` | `0.86` |
| 7 | `0.11813` | `0.36134` | `0.33594` | `0.91` |

기존 physnet 결과와 비교:

| Model | Run | Valid CalLL | Valid Acc |
|---|---|---:|---:|
| PhysNet ResNet34 224 | `gpu_physnet_holdout_v1` | `0.459281193032132` | `0.82` |
| PhysNet ResNet50 384 | `gpu_physnet_r50_384_v1` | `0.40319519115848723` | `0.84` |
| PhysNet ResNet101 384 | `gpu_physnet_r101_384_v1` | `0.5162458024837846` | `0.70` |
| PhysNet ConvNeXt-Tiny 384 | `gpu_physnet_convnext_tiny_384_v1` | `0.3033521122607969` | `0.86` |

최종 해석:

- `ConvNeXt-Tiny`는 `ResNet50 384`보다 유의미하게 더 좋았다.
- `ResNet101` 확장보다 backbone 계열을 바꾸는 접근이 훨씬 효과적이었다.
- 현재 physnet 계열 최고 holdout 후보는 `ConvNeXt-Tiny 384`다.

산출물:

- run dir:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_v1_20260309_205641`
- summary:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_v1_20260309_205641/holdout_summary.json`
- submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_v1_20260309_205641/submission_holdout.csv`

## 10. PhysNet ConvNeXt-Tiny CV

실험명:

- `gpu_physnet_convnext_tiny_384_cv_v1`

실행 경로:

- `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909`

명령:

```bash
python "/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/run_competition_pipeline.py" train-physnet --mode cv --run_name gpu_physnet_convnext_tiny_384_cv_v1 --backbone_name convnext_tiny --img_size 384 --motion_size 96 --batch_size 8 --epochs 16 --nfolds 5 --num_workers 4
```

최종 결과:

- `oof_logloss_cal`: `0.0007657534577136482`
- `oof_accuracy`: `1.0`
- `dev-only oof logloss`: `0.0041039192668788655`
- `fold_mean_valid_logloss_raw`: `0.01809316570938434`
- `fold_mean_temperature`: `0.3333333333333333`

fold별 결과:

| Fold | Best Epoch | Valid CalLL | Valid Acc | Temp |
|---:|---:|---:|---:|---:|
| 0 | 11 | `0.00351664` | `1.00000` | `0.3333` |
| 1 | 10 | `0.00000734` | `1.00000` | `0.3333` |
| 2 | 11 | `0.00015034` | `1.00000` | `0.3333` |
| 3 | 11 | `0.00013834` | `1.00000` | `0.3333` |
| 4 | 6 | `0.00001611` | `1.00000` | `0.3333` |

기존 CV 결과와 비교:

| Model | Run | OOF CalLL | OOF Acc | Dev-only OOF Logloss |
|---|---|---:|---:|---:|
| PhysNet ResNet50 384 CV | `gpu_physnet_r50_384_cv_v1` | `0.07421408550876968` | `0.98` | `0.2650819928648954` |
| Hybrid mask-bbox margin `0.10` CV | `hybrid_mask_bbox_margin010_cv_v1` | `0.015276939762022709` | `0.9972727272727273` | `0.09747774026268885` |
| PhysNet ConvNeXt-Tiny 384 CV | `gpu_physnet_convnext_tiny_384_cv_v1` | `0.0007657534577136482` | `1.0` | `0.0041039192668788655` |

최종 해석:

- `physnet convnext_tiny 384`는 현재 전체 실험 중 최고 CV 성능을 기록했다.
- `physnet r50 384 CV`는 물론이고 `hybrid margin 0.10 CV`보다도 크게 낮은 OOF logloss를 보였다.
- 수치가 매우 낮으므로 CV 분할과 실제 제출 간 일반화 차이는 별도 주의가 필요하지만, 현재 기준 최우선 제출 후보는 이 설정이다.

산출물:

- CV run dir:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909`
- summary json:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/cv_summary.json`
- fold metrics:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/fold_metrics.csv`
- OOF predictions:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/oof_predictions.csv`
- submission equal:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_equal.csv`
- submission weighted:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/submission_cv_weighted.csv`

## 11. PhysNet ConvNeXt-Tiny와 Hybrid 블렌드 검증

목적:

- `physnet convnext_tiny 384 cv`가 이미 매우 강한 상황에서 `hybrid mask_bbox margin 0.10 cv`를 낮은 비율로 섞을 때 추가 이득이 있는지 확인

대상:

- physnet OOF:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/gpu_physnet_convnext_tiny_384_cv_v1_20260309_205909/oof_predictions.csv`
- hybrid OOF:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/hybrid/hybrid_mask_bbox_margin010_cv_v1_20260309_194332/oof_predictions.csv`

OOF blend scan:

| Hybrid Weight | PhysNet Weight | OOF Logloss | OOF Acc | Dev Logloss |
|---:|---:|---:|---:|---:|
| `0.00` | `1.00` | `0.0007657534577136519` | `1.0` | `0.004103919266878871` |
| `0.01` | `0.99` | `0.00080889` | `1.0` | `0.004315` |
| `0.02` | `0.98` | `0.00085235` | `1.0` | `0.004528` |
| `0.05` | `0.95` | `0.00098338` | `1.0` | `0.005180` |
| `0.10` | `0.90` | `0.00120919` | `1.0` | `0.006314` |
| `0.15` | `0.85` | `0.001444` | `1.0` | `0.007511` |
| `0.20` | `0.80` | `0.001690` | `1.0` | `0.008779` |
| `0.25` | `0.75` | `0.001946` | `1.0` | `0.010126` |
| `0.30` | `0.70` | `0.002215` | `1.0` | `0.011562` |

최종 해석:

- best blend는 사실상 blend가 아닌 `physnet 1.00 + hybrid 0.00`이다.
- `hybrid`를 극소량 섞는 경우조차 OOF logloss가 계속 증가했다.
- 현재 기준 최종 제출 우선순위는 `physnet convnext_tiny 384` 단독이다.

산출물:

- blend scan csv:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/blend_oof_scan.csv`
- blend summary:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/blend_summary.json`
- best blend submission:
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_vs_hybrid_margin010_20260309/submission_blend_best.csv`
