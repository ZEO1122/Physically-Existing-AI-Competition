from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


CODEX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODEX_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data_reconstrcture" / "open"
OUTPUT_DIR = CODEX_ROOT / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split: str) -> pd.DataFrame:
    csv_name = "sample_submission.csv" if split == "test" else f"{split}.csv"
    df = pd.read_csv(DATA_ROOT / csv_name)
    df["id"] = df["id"].astype(str)
    return df


def image_summary(split: str, df: pd.DataFrame, n: int) -> dict:
    records = []
    for sample_id in df["id"].head(n):
        for view in ("front", "top"):
            path = DATA_ROOT / split / sample_id / f"{view}.png"
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            gray = img.mean(axis=2)
            records.append(
                {
                    "view": view,
                    "brightness_mean": float(img.mean()),
                    "brightness_std": float(img.std()),
                    "dark_ratio_090": float((gray < 0.90).mean()),
                    "dark_ratio_080": float((gray < 0.80).mean()),
                    "shape": list(img.shape[:2]),
                }
            )

    out = {}
    for view in ("front", "top"):
        part = [r for r in records if r["view"] == view]
        out[view] = {
            "samples": len(part),
            "brightness_mean": float(np.mean([r["brightness_mean"] for r in part])),
            "brightness_std": float(np.mean([r["brightness_std"] for r in part])),
            "dark_ratio_090": float(np.mean([r["dark_ratio_090"] for r in part])),
            "dark_ratio_080": float(np.mean([r["dark_ratio_080"] for r in part])),
            "image_shape": part[0]["shape"] if part else None,
        }
    return out


def video_motion_summary(df: pd.DataFrame, n: int) -> dict:
    rows = []
    for _, row in df.head(n).iterrows():
        sample_id = row["id"]
        video_path = DATA_ROOT / "train" / sample_id / "simulation.mp4"
        cap = cv2.VideoCapture(str(video_path))
        ok, prev = cap.read()
        if not ok:
            cap.release()
            continue
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diffs = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diffs.append(float(np.mean(cv2.absdiff(gray, prev))))
            prev = gray
        cap.release()
        arr = np.asarray(diffs, dtype=np.float32)
        rows.append(
            {
                "label": row["label"],
                "motion_mean": float(arr.mean()),
                "motion_std": float(arr.std()),
                "motion_max": float(arr.max()),
                "motion_q95": float(np.quantile(arr, 0.95)),
                "motion_sum": float(arr.sum()),
            }
        )

    stats = pd.DataFrame(rows)
    grouped = stats.groupby("label").mean(numeric_only=True).round(4)
    corr = stats.assign(y=(stats["label"] == "unstable").astype(int)).corr(numeric_only=True)["y"].round(4)
    return {
        "samples": len(stats),
        "group_mean": grouped.to_dict(),
        "correlation_with_unstable": corr.to_dict(),
    }


def main() -> None:
    train_df = load_split("train")
    dev_df = load_split("dev")
    test_df = load_split("test")

    report = {
        "data_root": str(DATA_ROOT),
        "split_sizes": {
            "train": int(len(train_df)),
            "dev": int(len(dev_df)),
            "test": int(len(test_df)),
        },
        "label_distribution": {
            "train": train_df["label"].value_counts().to_dict(),
            "dev": dev_df["label"].value_counts().to_dict(),
        },
        "image_summary": {
            "train": image_summary("train", train_df, n=200),
            "dev": image_summary("dev", dev_df, n=100),
            "test": image_summary("test", test_df, n=200),
        },
        "video_motion_summary": video_motion_summary(train_df, n=300),
    }

    out_path = OUTPUT_DIR / "dataset_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
