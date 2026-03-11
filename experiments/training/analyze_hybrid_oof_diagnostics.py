
"""
Diagnostics for hybrid_structure_aware runs.

기능
1) OOF를 train/dev로 분리한 성능 요약 CSV/JSON 저장
2) fold 3 dev 오답 샘플 추출
3) front / top / mask / overlay / 예측 정보를 묶은 시각 점검 패널 생성
4) contact sheet까지 저장

기본 사용 예시
python analyze_hybrid_oof_diagnostics.py \
  --oof_csv ./runs_xxx/oof_predictions.csv \
  --fold_valid_csv ./runs_xxx/valid_predictions_hybrid_structure_aware_fold3.csv \
  --data_root ./open \
  --trainer_py ./train_hybrid_structure_aware.py \
  --save_dir ./runs_xxx/oof_diagnostics
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clip_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def dacon_logloss(y_true: np.ndarray, unstable_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    unstable_prob = clip_probs(unstable_prob, eps=eps).reshape(-1)
    pred = np.stack([unstable_prob, 1.0 - unstable_prob], axis=1)
    pred = pred / pred.sum(axis=1, keepdims=True)
    true = np.stack([y_true, 1.0 - y_true], axis=1)
    return float(-np.mean(np.sum(true * np.log(pred), axis=1)))


def binary_accuracy(y_true: np.ndarray, unstable_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = (np.asarray(unstable_prob, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
    return float((pred == y_true).mean())


def per_sample_logloss(y_true: np.ndarray, unstable_prob: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = clip_probs(unstable_prob, eps=eps).reshape(-1)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def compute_metrics_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_name, sub in df.groupby("source"):
        y = sub["label_float"].values.astype(np.float64)
        p = sub["unstable_prob"].values.astype(np.float64)
        rows.append({
            "source": source_name,
            "n": int(len(sub)),
            "logloss": dacon_logloss(y, p),
            "accuracy": binary_accuracy(y, p),
            "raw_logloss": dacon_logloss(y, sub["unstable_prob_raw"].values.astype(np.float64)) if "unstable_prob_raw" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def compute_fold_source_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, source_name), sub in df.groupby(["fold", "source"]):
        y = sub["label_float"].values.astype(np.float64)
        p = sub["unstable_prob"].values.astype(np.float64)
        rows.append({
            "fold": int(fold),
            "source": source_name,
            "n": int(len(sub)),
            "logloss": dacon_logloss(y, p),
            "accuracy": binary_accuracy(y, p),
            "num_errors": int(((p >= 0.5).astype(np.float64) != y).sum()),
        })
    return pd.DataFrame(rows).sort_values(["fold", "source"]).reset_index(drop=True)


def load_trainer_module(trainer_py: Optional[str]):
    if trainer_py is None or (not os.path.exists(trainer_py)):
        return None
    spec = importlib.util.spec_from_file_location("trainer_mod", trainer_py)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def fallback_mask(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    sat = arr.max(axis=2) - arr.min(axis=2)
    mask = sat > np.percentile(sat, 94)
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def overlay_mask(base: Image.Image, mask: Image.Image, color=(255, 80, 80)) -> Image.Image:
    base = base.convert("RGB")
    mask_np = (np.asarray(mask, dtype=np.uint8) > 127)
    arr = np.asarray(base).copy()
    arr[mask_np] = (0.65 * arr[mask_np] + 0.35 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    return Image.fromarray(arr)


def make_text_panel(lines: List[str], width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    y = 12
    for line in lines:
        draw.text((12, y), line, fill=(20, 20, 20))
        y += 18
    return img


def make_error_panel(
    front: Image.Image,
    top: Image.Image,
    front_mask: Image.Image,
    top_mask: Image.Image,
    row: pd.Series,
    title: str,
) -> Image.Image:
    front_ov = overlay_mask(front, front_mask)
    top_ov = overlay_mask(top, top_mask)
    w, h = front.size
    text_lines = [
        f"id: {row['id']}",
        f"source: {row['source']}",
        f"fold: {row.get('fold', 'NA')}",
        f"label: {row['label']}",
        f"unstable_prob: {float(row['unstable_prob']):.6f}",
        f"stable_prob: {float(row['stable_prob']):.6f}",
        f"logit: {float(row['logit']):.6f}" if 'logit' in row else "",
        f"sample_logloss: {float(row['sample_logloss']):.6f}" if 'sample_logloss' in row else "",
        title,
    ]
    text_lines = [x for x in text_lines if x]
    text_panel = make_text_panel(text_lines, width=w, height=h)

    canvas = Image.new("RGB", (w * 3, h * 2), (245, 245, 245))
    canvas.paste(front.convert("RGB"), (0, 0))
    canvas.paste(top.convert("RGB"), (w, 0))
    canvas.paste(text_panel, (w * 2, 0))
    canvas.paste(front_ov, (0, h))
    canvas.paste(top_ov, (w, h))
    canvas.paste(front_mask.convert("RGB"), (w * 2, h))
    return canvas


def build_contact_sheet(images: List[Image.Image], cols: int = 2, bg=(255, 255, 255)) -> Optional[Image.Image]:
    if not images:
        return None
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (w * cols, h * rows), bg)
    for idx, im in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(im, (x, y))
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Analyze OOF + fold3 errors for hybrid structure-aware model")
    parser.add_argument("--oof_csv", type=str, required=True)
    parser.add_argument("--fold_valid_csv", type=str, default=None)
    parser.add_argument("--fold_idx", type=int, default=3)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--trainer_py", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.save_dir)
    oof_df = pd.read_csv(args.oof_csv)
    if "sample_logloss" not in oof_df.columns:
        oof_df["sample_logloss"] = per_sample_logloss(oof_df["label_float"].values, oof_df["unstable_prob"].values)

    split_metrics = compute_metrics_frame(oof_df)
    split_metrics.to_csv(os.path.join(args.save_dir, "oof_source_metrics.csv"), index=False)

    fold_source_metrics = compute_fold_source_metrics(oof_df)
    fold_source_metrics.to_csv(os.path.join(args.save_dir, "oof_fold_source_metrics.csv"), index=False)

    mis = oof_df[(oof_df["fold"] == int(args.fold_idx)) & (oof_df["source"] == "dev")]
    mis = mis[((mis["unstable_prob"] >= 0.5).astype(np.float64) != mis["label_float"].values)]
    mis = mis.sort_values("sample_logloss", ascending=False).reset_index(drop=True)
    mis.to_csv(os.path.join(args.save_dir, f"fold{args.fold_idx}_dev_errors.csv"), index=False)

    summary = {
        "fold_idx": int(args.fold_idx),
        "num_fold_dev_errors": int(len(mis)),
        "error_ids": mis["id"].tolist(),
        "oof_source_metrics": split_metrics.to_dict(orient="records"),
        "fold_source_metrics": fold_source_metrics.to_dict(orient="records"),
    }
    with open(os.path.join(args.save_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    trainer = load_trainer_module(args.trainer_py)
    panels = []
    if args.data_root is not None and len(mis) > 0:
        vis_dir = os.path.join(args.save_dir, f"fold{args.fold_idx}_dev_error_panels")
        ensure_dir(vis_dir)
        for _, row in mis.iterrows():
            front_path = row["front_path"]
            top_path = row["top_path"]
            if not (os.path.exists(front_path) and os.path.exists(top_path)):
                # data_root 기준으로 다시 구성해봅니다.
                sid = str(row["id"])
                front_path = str(Path(args.data_root) / row["source"] / sid / "front.png")
                top_path = str(Path(args.data_root) / row["source"] / sid / "top.png")
            if not (os.path.exists(front_path) and os.path.exists(top_path)):
                continue

            front = Image.open(front_path).convert("RGB")
            top = Image.open(top_path).convert("RGB")
            if trainer is not None and hasattr(trainer, "extract_object_mask_from_pil"):
                front_mask = trainer.extract_object_mask_from_pil(front, view="front")
                top_mask = trainer.extract_object_mask_from_pil(top, view="top")
            else:
                front_mask = fallback_mask(front)
                top_mask = fallback_mask(top)

            panel = make_error_panel(front, top, front_mask, top_mask, row, title="fold3 dev misclassification")
            out_path = os.path.join(vis_dir, f"{row['id']}_panel.png")
            panel.save(out_path)
            panels.append(panel)

        sheet = build_contact_sheet(panels, cols=2)
        if sheet is not None:
            sheet.save(os.path.join(args.save_dir, f"fold{args.fold_idx}_dev_error_contact_sheet.png"))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Done] saved diagnostics to: {args.save_dir}")


if __name__ == "__main__":
    main()
