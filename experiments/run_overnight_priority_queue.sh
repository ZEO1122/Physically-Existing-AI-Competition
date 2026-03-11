#!/usr/bin/env bash
set -u

ROOT="/home/ubuntu/Desktop/Physically Existing AI Competition/experiments"
RUNNER="$ROOT/run_competition_pipeline.py"
OUT_ROOT="$ROOT/outputs/physnet"
LOG_DIR="$ROOT/logs/overnight_20260310"
REPORT="$ROOT/overnight_experiments_20260310.md"
CONDA_BIN="/home/ubuntu/anaconda3/bin/conda"

mkdir -p "$LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

append_report() {
  printf "%s\n" "$1" >> "$REPORT"
}

append_summary() {
  local run_name="$1"
  local mode="$2"
  local summary_name
  if [ "$mode" = "cv" ]; then
    summary_name="cv_summary.json"
  else
    summary_name="holdout_summary.json"
  fi

  python - "$run_name" "$summary_name" "$OUT_ROOT" "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

run_name = sys.argv[1]
summary_name = sys.argv[2]
out_root = Path(sys.argv[3])
report = Path(sys.argv[4])

candidates = sorted(
    [p for p in out_root.glob(f"{run_name}_*") if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
)
if not candidates:
    with report.open("a", encoding="utf-8") as f:
        f.write(f"- 결과 디렉터리를 찾지 못했습니다: `{run_name}`\n\n")
    raise SystemExit(0)

run_dir = candidates[-1]
summary_path = run_dir / summary_name
if not summary_path.exists():
    with report.open("a", encoding="utf-8") as f:
        f.write(f"- summary 파일이 없습니다: `{summary_path}`\n\n")
    raise SystemExit(0)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
with report.open("a", encoding="utf-8") as f:
    f.write(f"- run dir: `{run_dir}`\n")
    if summary.get("mode") == "cv":
        f.write(f"- OOF logloss_cal: `{summary.get('oof_logloss_cal')}`\n")
        f.write(f"- OOF accuracy: `{summary.get('oof_accuracy')}`\n")
        f.write(f"- 제출(equal): `{summary.get('submission_equal_alias', summary.get('submission_equal'))}`\n")
        f.write(f"- 제출(weighted): `{summary.get('submission_weighted_alias', summary.get('submission_weighted'))}`\n\n")
    else:
        f.write(f"- valid_logloss_cal: `{summary.get('valid_logloss_cal')}`\n")
        f.write(f"- valid_accuracy: `{summary.get('valid_accuracy')}`\n")
        f.write(f"- 제출: `{summary.get('submission_alias', summary.get('submission'))}`\n\n")
PY
}

run_exp() {
  local title="$1"
  local run_name="$2"
  local mode="$3"
  shift 3

  local log_file="$LOG_DIR/${run_name}.log"
  append_report "## ${title}"
  append_report "- 시작 시각: \`$(timestamp)\`"
  append_report "- run_name: \`${run_name}\`"
  append_report "- mode: \`${mode}\`"
  append_report "- 로그: \`${log_file}\`"
  append_report ""

  if "$CONDA_BIN" run -n multiview python "$RUNNER" train-physnet --run_name "$run_name" --mode "$mode" "$@" >"$log_file" 2>&1; then
    append_report "- 상태: 완료"
    append_summary "$run_name" "$mode"
  else
    local exit_code=$?
    append_report "- 상태: 실패"
    append_report "- exit code: \`${exit_code}\`"
    append_report "- 로그 파일을 확인하세요: \`${log_file}\`"
    append_report ""
  fi
}

cat > "$REPORT" <<EOF
# Overnight Experiment Queue (2026-03-10)

## 목적

- 현재 최고 모델인 \`physnet convnext tiny 384\` 계열을 기준으로 성능 개선 가능성이 높은 우선순위 실험을 밤새 순차 실행합니다.
- 각 실험이 끝날 때마다 이 파일에 결과를 자동으로 추가합니다.

## 실행 순서

1. seed ensemble용 추가 CV: seed 52
2. seed ensemble용 추가 CV: seed 62
3. 같은 모델 + multi-scale TTA holdout
4. \`convnext_small\` 384 holdout
5. \`convnext_tiny\` 448 holdout
6. \`convnext_tiny\` 512 holdout
7. dense motion timepoints holdout

## 시작 시각

- \`$(timestamp)\`

## 결과 기록

EOF

run_exp \
  "1. PhysNet ConvNeXt-Tiny 384 CV Seed 52" \
  "overnight_physnet_convnext_tiny_384_cv_seed52_v1" \
  "cv" \
  --backbone_name convnext_tiny \
  --img_size 384 \
  --motion_size 96 \
  --batch_size 8 \
  --epochs 16 \
  --nfolds 5 \
  --seed 52 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0

run_exp \
  "2. PhysNet ConvNeXt-Tiny 384 CV Seed 62" \
  "overnight_physnet_convnext_tiny_384_cv_seed62_v1" \
  "cv" \
  --backbone_name convnext_tiny \
  --img_size 384 \
  --motion_size 96 \
  --batch_size 8 \
  --epochs 16 \
  --nfolds 5 \
  --seed 62 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0

run_exp \
  "3. PhysNet ConvNeXt-Tiny 384 Holdout Multi-Scale TTA" \
  "overnight_physnet_convnext_tiny_384_holdout_tta_ms094_v1" \
  "holdout" \
  --backbone_name convnext_tiny \
  --img_size 384 \
  --motion_size 96 \
  --batch_size 8 \
  --epochs 16 \
  --seed 42 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0,0.94

run_exp \
  "4. PhysNet ConvNeXt-Small 384 Holdout" \
  "overnight_physnet_convnext_small_384_holdout_v1" \
  "holdout" \
  --backbone_name convnext_small \
  --img_size 384 \
  --motion_size 96 \
  --batch_size 6 \
  --epochs 16 \
  --seed 42 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 3e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0

run_exp \
  "5. PhysNet ConvNeXt-Tiny 448 Holdout" \
  "overnight_physnet_convnext_tiny_448_holdout_v1" \
  "holdout" \
  --backbone_name convnext_tiny \
  --img_size 448 \
  --motion_size 112 \
  --batch_size 6 \
  --epochs 16 \
  --seed 42 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0

run_exp \
  "6. PhysNet ConvNeXt-Tiny 512 Holdout" \
  "overnight_physnet_convnext_tiny_512_holdout_v1" \
  "holdout" \
  --backbone_name convnext_tiny \
  --img_size 512 \
  --motion_size 128 \
  --batch_size 4 \
  --epochs 16 \
  --seed 42 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 2.5,5.0,7.5,10.0 \
  --tta_scales 1.0

run_exp \
  "7. PhysNet ConvNeXt-Tiny 384 Holdout Dense Motion Timepoints" \
  "overnight_physnet_convnext_tiny_384_motion_dense_holdout_v1" \
  "holdout" \
  --backbone_name convnext_tiny \
  --img_size 384 \
  --motion_size 96 \
  --batch_size 8 \
  --epochs 16 \
  --seed 42 \
  --num_workers 4 \
  --learning_rate 5e-4 \
  --backbone_lr 5e-5 \
  --weight_decay 1e-2 \
  --dropout 0.20 \
  --label_smoothing 0.02 \
  --aug_profile strong \
  --patience 5 \
  --motion_loss_weight 0.15 \
  --motion_dice_weight 0.25 \
  --motion_timepoints 1.5,3.0,4.5,6.0,7.5,9.0,10.5 \
  --tta_scales 1.0

append_report "## 종료 시각"
append_report ""
append_report "- \`$(timestamp)\`"
