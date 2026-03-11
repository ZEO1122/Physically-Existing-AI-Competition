# Overnight Experiment Queue (2026-03-10)

## 목적

- 현재 최고 모델인 `physnet convnext tiny 384` 계열을 기준으로 성능 개선 가능성이 높은 우선순위 실험을 밤새 순차 실행합니다.
- 각 실험이 끝날 때마다 이 파일에 결과를 자동으로 추가합니다.

## 실행 순서

1. seed ensemble용 추가 CV: seed 52
2. seed ensemble용 추가 CV: seed 62
3. 같은 모델 + multi-scale TTA holdout
4. `convnext_small` 384 holdout
5. `convnext_tiny` 448 holdout
6. `convnext_tiny` 512 holdout
7. dense motion timepoints holdout

## 시작 시각

- `2026-03-10 08:28:39 KST`

## 결과 기록

## 1. PhysNet ConvNeXt-Tiny 384 CV Seed 52
- 시작 시각: `2026-03-10 08:28:39 KST`
- run_name: `overnight_physnet_convnext_tiny_384_cv_seed52_v1`
- mode: `cv`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_384_cv_seed52_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed52_v1_20260310_082841`
- OOF logloss_cal: `0.013107108553172724`
- OOF accuracy: `0.9972727272727273`
- 제출(equal): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed52_v1_20260310_082841/overnight_physnet_convnext_tiny_384_cv_seed52_v1__submission_cv_equal.csv`
- 제출(weighted): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed52_v1_20260310_082841/overnight_physnet_convnext_tiny_384_cv_seed52_v1__submission_cv_weighted.csv`

## 2. PhysNet ConvNeXt-Tiny 384 CV Seed 62
- 시작 시각: `2026-03-10 08:37:16 KST`
- run_name: `overnight_physnet_convnext_tiny_384_cv_seed62_v1`
- mode: `cv`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_384_cv_seed62_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed62_v1_20260310_083718`
- OOF logloss_cal: `0.004193758180473049`
- OOF accuracy: `0.9990909090909091`
- 제출(equal): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed62_v1_20260310_083718/overnight_physnet_convnext_tiny_384_cv_seed62_v1__submission_cv_equal.csv`
- 제출(weighted): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_cv_seed62_v1_20260310_083718/overnight_physnet_convnext_tiny_384_cv_seed62_v1__submission_cv_weighted.csv`

## 3. PhysNet ConvNeXt-Tiny 384 Holdout Multi-Scale TTA
- 시작 시각: `2026-03-10 08:47:34 KST`
- run_name: `overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1`
- mode: `holdout`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1_20260310_084737`
- valid_logloss_cal: `0.28502823358899465`
- valid_accuracy: `0.82`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1_20260310_084737/overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1__submission_holdout.csv`

## 4. PhysNet ConvNeXt-Small 384 Holdout
- 시작 시각: `2026-03-10 08:48:51 KST`
- run_name: `overnight_physnet_convnext_small_384_holdout_v1`
- mode: `holdout`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_small_384_holdout_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_small_384_holdout_v1_20260310_084853`
- valid_logloss_cal: `0.32789508597197164`
- valid_accuracy: `0.87`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_small_384_holdout_v1_20260310_084853/overnight_physnet_convnext_small_384_holdout_v1__submission_holdout.csv`

## 5. PhysNet ConvNeXt-Tiny 448 Holdout
- 시작 시각: `2026-03-10 08:50:36 KST`
- run_name: `overnight_physnet_convnext_tiny_448_holdout_v1`
- mode: `holdout`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_448_holdout_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_448_holdout_v1_20260310_085038`
- valid_logloss_cal: `0.210324540793154`
- valid_accuracy: `0.93`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_448_holdout_v1_20260310_085038/overnight_physnet_convnext_tiny_448_holdout_v1__submission_holdout.csv`

## 6. PhysNet ConvNeXt-Tiny 512 Holdout
- 시작 시각: `2026-03-10 08:52:49 KST`
- run_name: `overnight_physnet_convnext_tiny_512_holdout_v1`
- mode: `holdout`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_512_holdout_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_512_holdout_v1_20260310_085251`
- valid_logloss_cal: `0.2750082287132134`
- valid_accuracy: `0.89`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_512_holdout_v1_20260310_085251/overnight_physnet_convnext_tiny_512_holdout_v1__submission_holdout.csv`

## 7. PhysNet ConvNeXt-Tiny 384 Holdout Dense Motion Timepoints
- 시작 시각: `2026-03-10 08:56:56 KST`
- run_name: `overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1`
- mode: `holdout`
- 로그: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/logs/overnight_20260310/overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1.log`

- 상태: 완료
- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1_20260310_085659`
- valid_logloss_cal: `0.28287163493295453`
- valid_accuracy: `0.87`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1_20260310_085659/overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1__submission_holdout.csv`

## 종료 시각

- `2026-03-10 08:58:17 KST`

## 후처리 앙상블

- 생성 시각: `2026-03-10`
- 목적: `seed42`, `seed52`, `seed62`의 `physnet convnext tiny 384 CV` 제출 예측을 평균하여 seed ensemble 제출본 생성
- 결과 디렉터리: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_384_seed_ensemble_20260310`
- 기준 단독 OOF logloss
  - `seed42`: `0.0007657534577136482`
  - `seed52`: `0.013107108553172724`
  - `seed62`: `0.004193758180473049`
- 앙상블 OOF logloss
  - `3-seed equal`: `0.0029790247905391474`
  - `3-seed weighted`: `0.001181065628278623`
  - `2-seed equal (seed42 + seed62)`: `0.0016873497316952607`
- 결론: 현재 OOF 기준으로는 `seed42` 단독이 가장 좋고, 이번 seed ensemble은 개선을 만들지 못했습니다.
- 생성된 제출 파일
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_384_seed_ensemble_20260310/physnet_convnext_tiny_384_seed42_seed52_seed62__submission_cv_equal3.csv`
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_384_seed_ensemble_20260310/physnet_convnext_tiny_384_seed42_seed52_seed62__submission_cv_weighted3.csv`
  - `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_convnext_tiny_384_seed_ensemble_20260310/physnet_convnext_tiny_384_seed42_seed62__submission_cv_equal2.csv`

## 추가 실험: PhysNet ConvNeXt-Tiny 448 CV

- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_cv_v1_20260310_090724`
- OOF logloss_cal: `0.007937001964161264`
- OOF accuracy: `0.9972727272727273`
- dev-only logloss: `0.041732418326578716`
- 제출(equal): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_cv_v1_20260310_090724/daytime_physnet_convnext_tiny_448_cv_v1__submission_cv_equal.csv`
- 제출(weighted): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_cv_v1_20260310_090724/daytime_physnet_convnext_tiny_448_cv_v1__submission_cv_weighted.csv`
- 해석: `448 holdout`에서는 강했지만, `448 CV`는 기존 최고 `384 CV` (`0.0007657534577136482`)를 넘지 못했습니다.

## 추가 실험: PhysNet ConvNeXt-Tiny 448 Holdout + TTA(1.0, 0.96)

- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_holdout_tta096_v1_20260310_111733`
- valid_logloss_cal: `0.20222306316192534`
- valid_accuracy: `0.94`
- 제출: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_holdout_tta096_v1_20260310_111733/daytime_physnet_convnext_tiny_448_holdout_tta096_v1__submission_holdout.csv`
- 비교: 기존 `448 holdout` (`0.210324540793154`, `0.93`)보다 개선됨
- 해석: `448`에서는 약한 multi-scale TTA(`1.0 + 0.96`)가 유효했습니다.

## 추가 실험: PhysNet ConvNeXt-Tiny 448 + Dense Motion CV

- run dir: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1_20260310_112018`
- OOF logloss_cal: `0.003206456335133361`
- OOF accuracy: `0.9981818181818182`
- dev-only logloss: `0.01035768803554419`
- 제출(equal): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1_20260310_112018/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_equal.csv`
- 제출(weighted): `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/physnet/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1_20260310_112018/daytime_physnet_convnext_tiny_448_motion_dense_cv_v1__submission_cv_weighted.csv`
- 비교
  - `448 기본 CV`: `0.007937001964161264`
  - `448 + dense motion CV`: `0.003206456335133361`
  - 기존 최고 `384 CV`: `0.0007657534577136482`
- 해석: dense motion timepoints는 `448` 설정에서 분명히 도움됐지만, 아직 기존 최고 `384 CV`는 넘지 못했습니다.

## 블렌딩 검증: 384 최고 CV vs 448 Dense Motion CV

- 결과 디렉터리: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_384_best_vs_448_dense_20260310`
- baseline
  - `384 CV best`: `0.0007657534577136482`
  - `448 dense motion CV`: `0.003206456335133361`
- best blend
  - `weight_physnet384`: `1.0`
  - `weight_physnet448dense`: `0.0`
  - `best_oof_logloss`: `0.0007657534577136519`
- 결론: 낮은 비율 블렌딩도 이득이 없었고, 여전히 `384 CV best` 단독이 최선입니다.
- 파일
  - scan: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_384_best_vs_448_dense_20260310/blend_oof_scan.csv`
  - summary: `/home/ubuntu/Desktop/Physically Existing AI Competition/codex_test/outputs/blends/physnet_384_best_vs_448_dense_20260310/blend_summary.json`
