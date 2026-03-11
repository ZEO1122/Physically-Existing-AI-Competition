
"""
Structure-aware hybrid trainer for DACON voxel stability task.

핵심 아이디어
1) front/top 원본 RGB는 그대로 사용합니다.
2) checkerboard / 조명 / 그림자 영향을 줄이기 위해 각 뷰에서 구조물 mask를 추출합니다.
3) front/top mask로부터 flip-invariant 구조 피처와 visual-hull 기반 pseudo-3D 물리 피처를 만듭니다.
4) RGB backbone + mask encoder + geometry MLP를 함께 학습시켜 stable / unstable을 예측합니다.
5) RGB 특징에서 geometry feature를 복원하는 auxiliary regression을 추가해 구조 민감도를 높입니다.

이 스크립트는 baseline의 open 폴더 경로 규칙을 그대로 사용합니다.
- open/train/{id}/front.png, top.png
- open/dev/{id}/front.png, top.png
- open/test/{id}/front.png, top.png
"""

import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageFont
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchvision
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None

try:
    import cv2
except Exception:
    cv2 = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_FLOAT = {"stable": 0.0, "unstable": 1.0}
GEOM_FEATURE_DIM = 39
MASK_META_DIM = 6
SIM_TARGET_DIM = 4


# ============================================================
# Generic utilities
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


class NullScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def unscale_(self, optimizer):
        return None


@contextlib.contextmanager
def autocast_context(device_type: str, enabled: bool):
    """
    CUDA + AMP enabled일 때만 안전하게 autocast를 엽니다.
    내부 forward 예외를 삼키지 않도록 단순 구조로 유지합니다.
    """
    if enabled and device_type == "cuda":
        with torch.amp.autocast("cuda", enabled=True):
            yield
    else:
        yield


def build_grad_scaler(device_type: str, enabled: bool):
    if not enabled:
        return NullScaler()
    try:
        return torch.amp.GradScaler(device_type, enabled=enabled)
    except Exception:
        try:
            return torch.amp.GradScaler(enabled=enabled)
        except Exception:
            if device_type == "cuda":
                return torch.cuda.amp.GradScaler(enabled=enabled)
            return NullScaler()


def seed_worker_factory(base_seed: int):
    def seed_worker(worker_id: int) -> None:
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return seed_worker


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


# ============================================================
# Data root resolution and split loading
# ============================================================

def resolve_data_root(data_root: str) -> str:
    """
    baseline_model처럼 스크립트 실행 위치가 달라도 open 폴더를 찾기 쉽게 후보 경로를 순차 확인합니다.
    """
    candidates = []
    data_root = str(data_root)

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    raw = Path(data_root)
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((cwd / raw).resolve())
        candidates.append((script_dir / raw).resolve())

    # 사용자가 ./open 으로 줬는데 현재 폴더가 하위 디렉토리인 상황까지 흡수합니다.
    for base in [cwd, cwd.parent, cwd.parent.parent, cwd.parent.parent.parent, script_dir, script_dir.parent, script_dir.parent.parent]:
        candidates.append((base / "open").resolve())

    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        unique.append(c)

    for c in unique:
        if (c / "train.csv").exists() and (c / "dev.csv").exists() and (c / "sample_submission.csv").exists():
            return str(c)

    checked = "\n".join(str(x) for x in unique)
    raise FileNotFoundError(f"open 데이터 루트를 찾지 못했습니다. 확인한 경로:\n{checked}")


def load_split_df(root_dir: str, split: str) -> pd.DataFrame:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")

    csv_name = "sample_submission.csv" if split == "test" else f"{split}.csv"
    csv_path = os.path.join(root_dir, csv_name)
    df = pd.read_csv(csv_path, dtype={"id": str})
    df["id"] = df["id"].astype(str)
    df["source"] = split

    split_dir = os.path.join(root_dir, split)
    df["folder"] = df["id"].map(lambda x: os.path.join(split_dir, x))
    df["front_path"] = df["folder"].map(lambda x: os.path.join(x, "front.png"))
    df["top_path"] = df["folder"].map(lambda x: os.path.join(x, "top.png"))
    df["simulation_path"] = df["folder"].map(lambda x: os.path.join(x, "simulation.mp4"))

    if "label" in df.columns:
        df["label_float"] = df["label"].map(LABEL_TO_FLOAT).astype(np.float64)
    else:
        df["label_float"] = np.nan
    return df


def verify_paths(df: pd.DataFrame, max_show: int = 10) -> None:
    missing = []
    for _, row in df.iterrows():
        if not os.path.exists(row["front_path"]):
            missing.append(row["front_path"])
        if not os.path.exists(row["top_path"]):
            missing.append(row["top_path"])
    if missing:
        preview = "\n".join(missing[:max_show])
        raise FileNotFoundError(f"총 {len(missing)}개 경로가 없습니다. 예시:\n{preview}")


def choose_stratify_target(df: pd.DataFrame, n_splits: int) -> pd.Series:
    joint = df["source"].astype(str) + "__" + df["label"].astype(str)
    counts = joint.value_counts()
    if len(counts) > 0 and counts.min() >= n_splits:
        return joint
    return df["label"].astype(str)


# ============================================================
# Mask extraction and geometry features
# ============================================================

def _rgb_to_hsv_np(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    diff = mx - mn
    sat = np.where(mx > 1e-6, diff / (mx + 1e-6), 0.0)
    val = mx
    return sat.astype(np.float32), val.astype(np.float32)


def _normalize_map(x: np.ndarray, p_lo: float = 5.0, p_hi: float = 95.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def _keep_best_component(mask: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
    if ndi is None:
        return mask.astype(bool)

    lbl, num = ndi.label(mask)
    if num <= 1:
        return mask.astype(bool)

    best_score = -1.0
    best_lab = 0
    H, W = mask.shape
    objects = ndi.find_objects(lbl)
    for lab, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        ys, xs = sl
        sub = (lbl[ys, xs] == lab)
        area = float(sub.sum())
        if area < max(32.0, H * W * 0.00025):
            continue

        x0, x1 = xs.start, xs.stop
        y0, y1 = ys.start, ys.stop
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_weight = math.exp(-(((cx - center_x) / (0.35 * W + 1e-6)) ** 2 + ((cy - center_y) / (0.35 * H + 1e-6)) ** 2))
        fill = area / max((x1 - x0) * (y1 - y0), 1)
        score = area * (0.55 + 0.45 * center_weight) * (0.70 + 0.30 * fill)
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab == 0:
        return mask.astype(bool)
    return (lbl == best_lab)


def _collect_components(mask: np.ndarray, center_x: float, center_y: float) -> List[Dict[str, float]]:
    """
    connected component 후보들의 품질 점수를 계산합니다.
    이후 best component 선택과 confidence 추정에 모두 사용합니다.
    """
    mask = mask.astype(bool)
    H, W = mask.shape
    if ndi is None:
        area = float(mask.sum())
        if area <= 0:
            return []
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        cx = float(xs.mean())
        cy = float(ys.mean())
        fill = area / max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
        center_weight = math.exp(-(((cx - center_x) / (0.35 * W + 1e-6)) ** 2 + ((cy - center_y) / (0.35 * H + 1e-6)) ** 2))
        score = area * (0.55 + 0.45 * center_weight) * (0.70 + 0.30 * fill)
        return [{"lab": 1, "area": area, "fill": fill, "center_weight": center_weight, "score": score}]
    lbl, num = ndi.label(mask)
    if num <= 0:
        return []
    objects = ndi.find_objects(lbl)
    comps = []
    for lab, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        ys, xs = sl
        sub = (lbl[ys, xs] == lab)
        area = float(sub.sum())
        if area < max(24.0, H * W * 0.0002):
            continue
        x0, x1 = xs.start, xs.stop
        y0, y1 = ys.start, ys.stop
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_weight = math.exp(-(((cx - center_x) / (0.35 * W + 1e-6)) ** 2 + ((cy - center_y) / (0.35 * H + 1e-6)) ** 2))
        fill = area / max((x1 - x0) * (y1 - y0), 1)
        score = area * (0.55 + 0.45 * center_weight) * (0.70 + 0.30 * fill)
        comps.append({
            "lab": lab,
            "area": area,
            "fill": fill,
            "center_weight": float(center_weight),
            "score": float(score),
            "bbox_area": float(max((x1 - x0) * (y1 - y0), 1)),
        })
    comps.sort(key=lambda x: x["score"], reverse=True)
    return comps


def _reasonable_area_score(area_ratio: float, view: str) -> float:
    if view == "front":
        lo, hi = 0.01, 0.22
    else:
        lo, hi = 0.004, 0.12
    if area_ratio <= 0:
        return 0.0
    if lo <= area_ratio <= hi:
        return 1.0
    if area_ratio < lo:
        return float(np.clip(area_ratio / max(lo, 1e-6), 0.0, 1.0))
    return float(np.clip(hi / max(area_ratio, 1e-6), 0.0, 1.0))


def _mask_confidence_from_stats(
    view: str,
    score_map: np.ndarray,
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    sat_n: np.ndarray,
    col_n: np.ndarray,
    edge_n: np.ndarray,
) -> float:
    H, W = mask.shape
    area_ratio = float(mask.mean())
    area_score = _reasonable_area_score(area_ratio, view)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0

    cx = float(xs.mean())
    cy = float(ys.mean())
    center_score = math.exp(-(((cx - center_x) / (0.30 * W + 1e-6)) ** 2 + ((cy - center_y) / (0.30 * H + 1e-6)) ** 2))
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    fill = float(mask[y0:y1 + 1, x0:x1 + 1].mean())
    fill_score = float(np.clip((fill - 0.18) / 0.55, 0.0, 1.0))

    inside = mask > 0
    outside = ~inside
    score_inside = float(score_map[inside].mean()) if inside.any() else 0.0
    score_outside = float(score_map[outside].mean()) if outside.any() else 0.0
    contrast_score = float(np.clip((score_inside - score_outside + 0.05) / 0.35, 0.0, 1.0))

    edge_inside = float(edge_n[inside].mean()) if inside.any() else 0.0
    edge_support = float(np.clip(edge_inside / 0.35, 0.0, 1.0))
    color_inside = float((0.5 * sat_n[inside] + 0.5 * col_n[inside]).mean()) if inside.any() else 0.0
    color_score = float(np.clip(color_inside / 0.45, 0.0, 1.0))

    high_thr = np.percentile(score_map, 96.5 if view == "front" else 97.0)
    low_thr = np.percentile(score_map, 92.0 if view == "front" else 93.0)
    mask_hi = score_map >= high_thr
    mask_lo = score_map >= low_thr
    inter = float((mask_hi & mask_lo & inside).sum())
    union = float(((mask_hi | mask_lo) & inside).sum())
    stability_score = float(inter / max(union, 1.0))

    comps = _collect_components(mask, center_x=center_x, center_y=center_y)
    if len(comps) == 0:
        margin_score = 0.0
    elif len(comps) == 1:
        margin_score = 1.0
    else:
        margin = comps[0]["score"] / max(comps[1]["score"] + 1e-6, 1e-6)
        margin_score = float(np.clip((margin - 1.0) / 2.0, 0.0, 1.0))

    conf = (
        0.18 * area_score
        + 0.18 * float(center_score)
        + 0.15 * fill_score
        + 0.18 * contrast_score
        + 0.12 * edge_support
        + 0.09 * color_score
        + 0.05 * stability_score
        + 0.05 * margin_score
    )
    return float(np.clip(conf, 0.0, 1.0))


def extract_object_mask_and_confidence_from_pil(img: Image.Image, view: str) -> Tuple[Image.Image, float]:
    """
    heuristic foreground mask와 함께 mask 신뢰도(confidence)를 계산합니다.
    confidence는 이후 mask/geometry branch gating에 사용됩니다.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    sat, val = _rgb_to_hsv_np(arr)
    colorfulness = arr.max(axis=2) - arr.min(axis=2)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx * gx + gy * gy)

    sat_n = _normalize_map(sat, 5, 95)
    col_n = _normalize_map(colorfulness, 5, 95)
    edge_n = _normalize_map(edge, 5, 98)
    val_n = _normalize_map(val, 5, 95)

    H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    if view == "front":
        cx = W * 0.50
        cy = H * 0.58
        sx = W * 0.33
        sy = H * 0.28
        thr_percentile = 94.2
    else:
        cx = W * 0.50
        cy = H * 0.52
        sx = W * 0.22
        sy = H * 0.22
        thr_percentile = 95.2

    center_prior = np.exp(-(((xx - cx) / (sx + 1e-6)) ** 2 + ((yy - cy) / (sy + 1e-6)) ** 2)).astype(np.float32)
    score = 0.50 * sat_n + 0.32 * col_n + 0.20 * edge_n - 0.08 * val_n + 0.18 * center_prior

    mask = score >= np.percentile(score, thr_percentile)
    mask &= ~((val > np.percentile(val, 98.5)) & (sat < np.percentile(sat, 30.0)))

    if ndi is not None:
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
        mask = ndi.binary_fill_holes(mask)

    comps = _collect_components(mask.astype(bool), center_x=cx, center_y=cy)
    if ndi is not None and len(comps) > 0:
        lbl, _ = ndi.label(mask.astype(bool))
        best_lab = int(comps[0]["lab"])
        mask = (lbl == best_lab)
    else:
        mask = _keep_best_component(mask.astype(bool), center_x=cx, center_y=cy)

    if ndi is not None:
        mask = ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        mask = ndi.binary_fill_holes(mask)

    conf = _mask_confidence_from_stats(
        view=view,
        score_map=score,
        mask=mask.astype(np.uint8),
        center_x=float(cx),
        center_y=float(cy),
        sat_n=sat_n,
        col_n=col_n,
        edge_n=edge_n,
    )
    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L"), float(conf)


def extract_object_mask_from_pil(img: Image.Image, view: str) -> Image.Image:
    """
    기존 코드 호환용 wrapper.
    """
    mask, _ = extract_object_mask_and_confidence_from_pil(img, view=view)
    return mask


def _resize_mask_array(mask: np.ndarray, out_w: int, out_h: int, threshold: float = 0.20) -> np.ndarray:
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    arr = np.asarray(pil.resize((out_w, out_h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    return (arr >= threshold).astype(np.uint8)


def _crop_pair_to_mask_bbox(
    img: Image.Image,
    mask: Image.Image,
    margin_ratio: float = 0.18,
    min_side_ratio: float = 0.42,
) -> Tuple[Image.Image, Image.Image]:
    mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.uint8)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img, mask

    width, height = img.size
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())

    box_w = max(x1 - x0 + 1, 1)
    box_h = max(y1 - y0 + 1, 1)
    side = float(max(box_w, box_h))
    side = max(side * (1.0 + 2.0 * float(margin_ratio)), min(width, height) * float(min_side_ratio))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * side

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = int(round(cx + half))
    bottom = int(round(cy + half))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= (right - width)
        right = width
    if bottom > height:
        top -= (bottom - height)
        bottom = height

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, width)
    bottom = min(bottom, height)

    if right - left < 4 or bottom - top < 4:
        return img, mask
    return img.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))


def _bbox_stats(mask: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    area = float(mask.mean())
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = float((x1 - x0 + 1) / max(W, 1))
    bh = float((y1 - y0 + 1) / max(H, 1))
    fill = float(mask[y0:y1 + 1, x0:x1 + 1].mean())
    cx = float(xs.mean() / max(W - 1, 1))
    cy = float(ys.mean() / max(H - 1, 1))
    return area, bw, bh, fill, min(abs(cx - 0.5) * 2.0, 1.0), min(cy, 1.0)


def _shape_spread(mask: np.ndarray) -> Tuple[float, float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return 0.0, 0.0, 0.0
    H, W = mask.shape
    coords = np.stack([
        ys.astype(np.float32) / max(H - 1, 1),
        xs.astype(np.float32) / max(W - 1, 1),
    ], axis=1)
    cov = np.cov(coords.T)
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = float(np.clip(vals[0] * 12.0, 0.0, 1.0))
    minor = float(np.clip(vals[1] * 12.0, 0.0, 1.0))
    ecc = float(np.clip((vals[0] - vals[1]) / (vals[0] + 1e-6), 0.0, 1.0))
    return major, minor, ecc


def compute_geometry_features(
    front_mask: np.ndarray,
    top_mask: np.ndarray,
    grid_xy: int = 16,
    grid_z: int = 20,
) -> np.ndarray:
    """
    flip-invariant 구조 피처를 만듭니다.
    - 2D bbox / area / spread
    - front-top visual hull 기반 pseudo-3D 물리 피처
    """
    front_mask = (front_mask > 0).astype(np.uint8)
    top_mask = (top_mask > 0).astype(np.uint8)

    fa, fbw, fbh, ffill, fcx_abs, fcy = _bbox_stats(front_mask)
    ta, tbw, tbh, tfill, tcx_abs, tcy = _bbox_stats(top_mask)
    fmaj, fmin, fecc = _shape_spread(front_mask)
    tmaj, tmin, tecc = _shape_spread(top_mask)

    front_grid = _resize_mask_array(front_mask, grid_xy, grid_z, threshold=0.18)   # [z, x]
    top_grid_hw = _resize_mask_array(top_mask, grid_xy, grid_xy, threshold=0.18)   # [y, x]
    top_grid = top_grid_hw.T  # [x, y]

    occ = (front_grid[:, :, None] & top_grid[None, :, :]).astype(np.uint8)         # [z, x, y]
    total = float(occ.sum())

    if total <= 0:
        return np.zeros(GEOM_FEATURE_DIM, dtype=np.float32)

    layer_area = occ.reshape(grid_z, -1).mean(axis=1).astype(np.float32)
    width_profile = front_grid.mean(axis=1).astype(np.float32)  # 각 높이에서 x방향 점유율
    height_active = np.where(occ.any(axis=(1, 2)))[0]
    height_ratio = float((height_active.max() + 1) / max(grid_z, 1)) if len(height_active) else 0.0

    quarter = max(grid_z // 4, 1)
    bottom_area = float(layer_area[:quarter].mean())
    mid_area = float(layer_area[quarter: grid_z - quarter].mean()) if grid_z > 2 * quarter else float(layer_area.mean())
    top_area = float(layer_area[-quarter:].mean())
    base_width = float(width_profile[:quarter].mean())
    mid_width = float(width_profile[quarter: grid_z - quarter].mean()) if grid_z > 2 * quarter else float(width_profile.mean())
    top_width = float(width_profile[-quarter:].mean())

    support_count = int(occ[0].sum()) + int((occ[1:] & occ[:-1]).sum())
    support_ratio = float(support_count / max(total, 1.0))
    support_ratio = float(np.clip(support_ratio, 0.0, 1.0))
    unsupported_ratio = float(np.clip(1.0 - support_ratio, 0.0, 1.0))

    z_idx, x_idx, y_idx = np.where(occ > 0)
    com_z = float(z_idx.mean() / max(grid_z - 1, 1))
    com_x = float(x_idx.mean() / max(grid_xy - 1, 1))
    com_y = float(y_idx.mean() / max(grid_xy - 1, 1))
    com_x_abs = float(np.clip(abs(com_x - 0.5) * 2.0, 0.0, 1.0))
    com_y_abs = float(np.clip(abs(com_y - 0.5) * 2.0, 0.0, 1.0))

    base = occ[0]
    if base.any():
        bx, by = np.where(base > 0)
        base_cx = float(bx.mean() / max(grid_xy - 1, 1))
        base_cy = float(by.mean() / max(grid_xy - 1, 1))
        com_base_offset = float(np.clip(math.sqrt((com_x - base_cx) ** 2 + (com_y - base_cy) ** 2) * math.sqrt(2.0), 0.0, 1.0))
        base_footprint = float(base.mean())
    else:
        com_base_offset = 0.0
        base_footprint = 0.0

    layer_centroids = []
    for z in range(grid_z):
        xs, ys = np.where(occ[z] > 0)
        if len(xs) == 0:
            continue
        cx = float(xs.mean() / max(grid_xy - 1, 1))
        cy = float(ys.mean() / max(grid_xy - 1, 1))
        layer_centroids.append((cx, cy))
    if len(layer_centroids) >= 2:
        layer_centroids = np.asarray(layer_centroids, dtype=np.float32)
        drift = np.sqrt(((layer_centroids - layer_centroids[0:1]) ** 2).sum(axis=1))
        drift_mean = float(np.clip(drift.mean() * math.sqrt(2.0), 0.0, 1.0))
        drift_max = float(np.clip(drift.max() * math.sqrt(2.0), 0.0, 1.0))
    else:
        drift_mean = 0.0
        drift_max = 0.0

    voxel_ratio = float(occ.mean())
    layer_std = float(np.clip(layer_area.std() * 4.0, 0.0, 1.0))
    layer_max = float(np.clip(layer_area.max() * 4.0, 0.0, 1.0))
    top_heavy_ratio = float(np.clip(top_area / (bottom_area + 1e-6), 0.0, 2.0) / 2.0)
    slenderness = float(np.clip(height_ratio / (math.sqrt(base_footprint + 1e-6) + 1e-6), 0.0, 3.0) / 3.0)

    features = np.asarray([
        # front 2D
        fa, fbw, fbh, ffill, fcx_abs, fcy, fmaj, fmin, fecc,
        # top 2D
        ta, tbw, tbh, tfill, tcx_abs, tcy, tmaj, tmin, tecc,
        # pseudo 3D / physics
        voxel_ratio, height_ratio, base_footprint, bottom_area, mid_area, top_area,
        base_width, mid_width, top_width,
        support_ratio, unsupported_ratio, top_heavy_ratio, slenderness,
        com_z, com_x_abs, com_y_abs, com_base_offset, drift_mean, drift_max,
        layer_std, layer_max,
    ], dtype=np.float32)

    # 피처 범위를 안정적으로 유지합니다.
    features = np.nan_to_num(np.clip(features, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if len(features) != GEOM_FEATURE_DIM:
        raise ValueError(f"Geometry feature dim mismatch: expected {GEOM_FEATURE_DIM}, got {len(features)}")
    return features


def make_mask_debug_triptych(front: Image.Image, top: Image.Image, front_mask: Image.Image, top_mask: Image.Image) -> Image.Image:
    width = front.width
    height = front.height
    canvas = Image.new("RGB", (width * 4, height), (245, 245, 245))
    canvas.paste(front.convert("RGB"), (0, 0))
    canvas.paste(top.convert("RGB"), (width, 0))
    canvas.paste(front_mask.convert("RGB"), (width * 2, 0))
    canvas.paste(top_mask.convert("RGB"), (width * 3, 0))
    return canvas


def _mask_meta_from_conf_and_masks(front_conf: float, top_conf: float, front_mask_np: np.ndarray, top_mask_np: np.ndarray) -> np.ndarray:
    front_area = float(np.clip(front_mask_np.mean(), 0.0, 1.0))
    top_area = float(np.clip(top_mask_np.mean(), 0.0, 1.0))
    mean_conf = float(np.clip(0.5 * (front_conf + top_conf), 0.0, 1.0))
    gap_conf = float(np.clip(abs(front_conf - top_conf), 0.0, 1.0))
    return np.asarray([front_conf, top_conf, mean_conf, gap_conf, front_area, top_area], dtype=np.float32)


def compute_simulation_targets(video_path: str, frame_stride: int = 5, resize_hw: int = 128) -> Tuple[np.ndarray, float]:
    """
    train/sample/simulation.mp4에서 간단한 motion summary target을 뽑습니다.
    - final displacement proxy
    - peak displacement proxy
    - mean motion proxy
    - onset timing
    반환값은 [0,1] 범위의 4차원 target과 valid flag입니다.
    """
    if cv2 is None or (video_path is None) or (not os.path.exists(video_path)):
        return np.zeros(SIM_TARGET_DIM, dtype=np.float32), 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros(SIM_TARGET_DIM, dtype=np.float32), 0.0

    frames = []
    frame_idx = 0
    try:
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if frame_idx % max(frame_stride, 1) == 0:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                if resize_hw > 0:
                    gray = cv2.resize(gray, (resize_hw, resize_hw), interpolation=cv2.INTER_AREA)
                frames.append(gray.astype(np.float32) / 255.0)
            frame_idx += 1
    finally:
        cap.release()

    if len(frames) < 2:
        return np.zeros(SIM_TARGET_DIM, dtype=np.float32), 0.0

    first = frames[0]
    curve_first = np.asarray([float(np.abs(fr - first).mean()) for fr in frames[1:]], dtype=np.float32)
    if len(curve_first) == 0:
        return np.zeros(SIM_TARGET_DIM, dtype=np.float32), 0.0

    peak_motion = float(curve_first.max())
    final_motion = float(curve_first[-1])
    mean_motion = float(curve_first.mean())
    thr = max(0.0035, 0.20 * peak_motion)
    onset_idx = int(np.argmax(curve_first >= thr)) if np.any(curve_first >= thr) else len(curve_first) - 1
    onset_norm = float(np.clip(onset_idx / max(len(curve_first) - 1, 1), 0.0, 1.0))

    target = np.asarray([
        np.clip(final_motion * 35.0, 0.0, 1.0),
        np.clip(peak_motion * 35.0, 0.0, 1.0),
        np.clip(mean_motion * 55.0, 0.0, 1.0),
        onset_norm,
    ], dtype=np.float32)
    return target, 1.0


# ============================================================
# Pair transforms
# ============================================================

class StructureAwarePairTrainTransform:
    """
    원본 RGB와 mask를 함께 받아서 같은 기하 증강을 적용합니다.
    photometric 증강은 RGB에만 적용하고, 증강 후 geometry / mask meta를 다시 계산합니다.
    """
    def __init__(
        self,
        img_size: int,
        grid_xy: int,
        grid_z: int,
        aug_profile: str = "base",
        crop_profile: str = "none",
        crop_margin_ratio: float = 0.18,
        crop_min_side_ratio: float = 0.42,
    ):
        self.img_size = int(img_size)
        self.grid_xy = int(grid_xy)
        self.grid_z = int(grid_z)
        self.aug_profile = str(aug_profile)
        self.crop_profile = str(crop_profile)
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.crop_min_side_ratio = float(crop_min_side_ratio)

    def _resize(self, img: Image.Image, is_mask: bool) -> Image.Image:
        interp = InterpolationMode.NEAREST if is_mask else InterpolationMode.BICUBIC
        return TF.resize(img, [self.img_size, self.img_size], interpolation=interp)

    def _geometric_params(self) -> Tuple[float, Tuple[int, int], float]:
        angle = random.uniform(-6.0, 6.0)
        max_shift = int(self.img_size * 0.05)
        translate = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
        scale = random.uniform(0.95, 1.05)
        return angle, translate, scale

    def _photometric_params(self) -> Tuple[float, float, float, float]:
        if self.aug_profile == "strong_domain":
            brightness = 1.0 + random.uniform(-0.32, 0.32)
            contrast = 1.0 + random.uniform(-0.28, 0.28)
            saturation = 1.0 + random.uniform(-0.30, 0.30)
            hue = random.uniform(-0.05, 0.05)
        elif self.aug_profile == "crop_tuned":
            brightness = 1.0 + random.uniform(-0.24, 0.24)
            contrast = 1.0 + random.uniform(-0.22, 0.22)
            saturation = 1.0 + random.uniform(-0.20, 0.20)
            hue = random.uniform(-0.035, 0.035)
        else:
            brightness = 1.0 + random.uniform(-0.18, 0.18)
            contrast = 1.0 + random.uniform(-0.18, 0.18)
            saturation = 1.0 + random.uniform(-0.18, 0.18)
            hue = random.uniform(-0.03, 0.03)
        return brightness, contrast, saturation, hue

    @staticmethod
    def _apply_gamma(img: Image.Image, gamma: float) -> Image.Image:
        gamma = max(float(gamma), 1e-4)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _apply_same_affine(self, img: Image.Image, angle: float, translate: Tuple[int, int], scale: float, is_mask: bool) -> Image.Image:
        interp = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
        fill = 0
        return TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=interp, fill=fill)

    def __call__(self, front: Image.Image, top: Image.Image, front_mask: Image.Image, top_mask: Image.Image, front_conf: float, top_conf: float):
        if self.crop_profile == "mask_bbox":
            front, front_mask = _crop_pair_to_mask_bbox(
                front, front_mask, margin_ratio=self.crop_margin_ratio, min_side_ratio=self.crop_min_side_ratio
            )
            top, top_mask = _crop_pair_to_mask_bbox(
                top, top_mask, margin_ratio=self.crop_margin_ratio, min_side_ratio=self.crop_min_side_ratio
            )

        front = self._resize(front, is_mask=False)
        top = self._resize(top, is_mask=False)
        front_mask = self._resize(front_mask, is_mask=True)
        top_mask = self._resize(top_mask, is_mask=True)

        if random.random() < 0.5:
            front = TF.hflip(front)
            top = TF.hflip(top)
            front_mask = TF.hflip(front_mask)
            top_mask = TF.hflip(top_mask)

        angle, translate, scale = self._geometric_params()
        front = self._apply_same_affine(front, angle, translate, scale, is_mask=False)
        top = self._apply_same_affine(top, angle, translate, scale, is_mask=False)
        front_mask = self._apply_same_affine(front_mask, angle, translate, scale, is_mask=True)
        top_mask = self._apply_same_affine(top_mask, angle, translate, scale, is_mask=True)

        brightness, contrast, saturation, hue = self._photometric_params()
        front = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(front, brightness), contrast), saturation), hue)
        top = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(top, brightness), contrast), saturation), hue)

        if self.aug_profile == "strong_domain":
            if random.random() < 0.55:
                gamma = random.uniform(0.75, 1.45)
                front = self._apply_gamma(front, gamma)
                top = self._apply_gamma(top, gamma)
            if random.random() < 0.12:
                front = ImageOps.grayscale(front).convert("RGB")
                top = ImageOps.grayscale(top).convert("RGB")
        elif self.aug_profile == "crop_tuned":
            if random.random() < 0.25:
                gamma = random.uniform(0.88, 1.18)
                front = self._apply_gamma(front, gamma)
                top = self._apply_gamma(top, gamma)
            if random.random() < 0.04:
                front = ImageOps.grayscale(front).convert("RGB")
                top = ImageOps.grayscale(top).convert("RGB")

        if self.aug_profile == "strong_domain":
            blur_p, autocontrast_p, jpeg_p, blur_hi, q_lo, q_hi = 0.18, 0.18, 0.14, 1.2, 35, 78
        elif self.aug_profile == "crop_tuned":
            blur_p, autocontrast_p, jpeg_p, blur_hi, q_lo, q_hi = 0.14, 0.12, 0.10, 0.95, 42, 80
        else:
            blur_p, autocontrast_p, jpeg_p, blur_hi, q_lo, q_hi = 0.12, 0.10, 0.08, 0.8, 48, 82

        if random.random() < blur_p:
            front = front.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, blur_hi)))
            top = top.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, blur_hi)))
        if random.random() < autocontrast_p:
            front = ImageOps.autocontrast(front)
            top = ImageOps.autocontrast(top)
        if random.random() < jpeg_p:
            buf = io.BytesIO()
            front.save(buf, format="JPEG", quality=random.randint(q_lo, q_hi))
            buf.seek(0)
            front = Image.open(buf).convert("RGB")
            buf = io.BytesIO()
            top.save(buf, format="JPEG", quality=random.randint(q_lo, q_hi))
            buf.seek(0)
            top = Image.open(buf).convert("RGB")

        front_t = TF.normalize(TF.to_tensor(front), IMAGENET_MEAN, IMAGENET_STD)
        top_t = TF.normalize(TF.to_tensor(top), IMAGENET_MEAN, IMAGENET_STD)
        front_mask_np = (np.asarray(front_mask, dtype=np.uint8) > 127).astype(np.uint8)
        top_mask_np = (np.asarray(top_mask, dtype=np.uint8) > 127).astype(np.uint8)
        front_mask_t = torch.from_numpy(front_mask_np[None].astype(np.float32))
        top_mask_t = torch.from_numpy(top_mask_np[None].astype(np.float32))
        geom = torch.from_numpy(compute_geometry_features(front_mask_np, top_mask_np, grid_xy=self.grid_xy, grid_z=self.grid_z))
        mask_meta = torch.from_numpy(_mask_meta_from_conf_and_masks(front_conf, top_conf, front_mask_np, top_mask_np))
        return front_t, top_t, front_mask_t, top_mask_t, geom, mask_meta


class StructureAwarePairEvalTransform:
    def __init__(
        self,
        img_size: int,
        grid_xy: int,
        grid_z: int,
        crop_profile: str = "none",
        crop_margin_ratio: float = 0.18,
        crop_min_side_ratio: float = 0.42,
    ):
        self.img_size = int(img_size)
        self.grid_xy = int(grid_xy)
        self.grid_z = int(grid_z)
        self.crop_profile = str(crop_profile)
        self.crop_margin_ratio = float(crop_margin_ratio)
        self.crop_min_side_ratio = float(crop_min_side_ratio)

    def __call__(self, front: Image.Image, top: Image.Image, front_mask: Image.Image, top_mask: Image.Image, front_conf: float, top_conf: float):
        if self.crop_profile == "mask_bbox":
            front, front_mask = _crop_pair_to_mask_bbox(
                front, front_mask, margin_ratio=self.crop_margin_ratio, min_side_ratio=self.crop_min_side_ratio
            )
            top, top_mask = _crop_pair_to_mask_bbox(
                top, top_mask, margin_ratio=self.crop_margin_ratio, min_side_ratio=self.crop_min_side_ratio
            )

        front = TF.resize(front, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        top = TF.resize(top, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        front_mask = TF.resize(front_mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        top_mask = TF.resize(top_mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)

        front_t = TF.normalize(TF.to_tensor(front), IMAGENET_MEAN, IMAGENET_STD)
        top_t = TF.normalize(TF.to_tensor(top), IMAGENET_MEAN, IMAGENET_STD)
        front_mask_np = (np.asarray(front_mask, dtype=np.uint8) > 127).astype(np.uint8)
        top_mask_np = (np.asarray(top_mask, dtype=np.uint8) > 127).astype(np.uint8)
        front_mask_t = torch.from_numpy(front_mask_np[None].astype(np.float32))
        top_mask_t = torch.from_numpy(top_mask_np[None].astype(np.float32))
        geom = torch.from_numpy(compute_geometry_features(front_mask_np, top_mask_np, grid_xy=self.grid_xy, grid_z=self.grid_z))
        mask_meta = torch.from_numpy(_mask_meta_from_conf_and_masks(front_conf, top_conf, front_mask_np, top_mask_np))
        return front_t, top_t, front_mask_t, top_mask_t, geom, mask_meta


# ============================================================
# Dataset
# ============================================================

class StructureAwareDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, is_test: bool = False, cache_masks: bool = True, sim_frame_stride: int = 5):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.cache_masks = cache_masks
        self.mask_cache: Dict[str, Tuple[Image.Image, Image.Image, float, float]] = {}
        self.sim_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.sim_frame_stride = int(sim_frame_stride)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _get_masks(self, sample_id: str, front: Image.Image, top: Image.Image) -> Tuple[Image.Image, Image.Image, float, float]:
        if self.cache_masks and sample_id in self.mask_cache:
            fm, tm, fc, tc = self.mask_cache[sample_id]
            return fm.copy(), tm.copy(), float(fc), float(tc)

        front_mask, front_conf = extract_object_mask_and_confidence_from_pil(front, view="front")
        top_mask, top_conf = extract_object_mask_and_confidence_from_pil(top, view="top")

        if self.cache_masks:
            self.mask_cache[sample_id] = (front_mask.copy(), top_mask.copy(), float(front_conf), float(top_conf))
        return front_mask, top_mask, float(front_conf), float(top_conf)

    def _get_sim_target(self, row: pd.Series) -> Tuple[np.ndarray, float]:
        sample_id = str(row["id"])
        if sample_id in self.sim_cache:
            return self.sim_cache[sample_id]
        video_path = row.get("simulation_path", None)
        target, valid = compute_simulation_targets(video_path, frame_stride=self.sim_frame_stride, resize_hw=128)
        self.sim_cache[sample_id] = (target, float(valid))
        return target, float(valid)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        front = self._load_rgb(row["front_path"])
        top = self._load_rgb(row["top_path"])
        front_mask, top_mask, front_conf, top_conf = self._get_masks(sample_id, front, top)

        if self.transform is not None:
            front, top, front_mask, top_mask, geom, mask_meta = self.transform(front, top, front_mask, top_mask, front_conf, top_conf)
        else:
            raise RuntimeError("transform is required for StructureAwareDataset")

        if self.is_test:
            return front, top, front_mask, top_mask, geom, mask_meta, sample_id

        sim_target_np, sim_valid = self._get_sim_target(row)
        sim_target = torch.from_numpy(sim_target_np.astype(np.float32))
        sim_valid_t = torch.tensor([float(sim_valid)], dtype=torch.float32)
        label = torch.tensor([float(row["label_float"])], dtype=torch.float32)
        return front, top, front_mask, top_mask, geom, mask_meta, sim_target, sim_valid_t, label, sample_id


# ============================================================
# Model
# ============================================================

def build_backbone(backbone_name: str = "efficientnet_v2_s", pretrained: bool = True):
    if backbone_name != "efficientnet_v2_s":
        raise ValueError("이 스크립트는 efficientnet_v2_s 기준으로 고정되어 있습니다.")
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    try:
        base = models.efficientnet_v2_s(weights=weights)
    except Exception as e:
        print(f"[Warning] pretrained load 실패 -> random init 사용: {e}")
        base = models.efficientnet_v2_s(weights=None)
    feat_dim = base.classifier[1].in_features
    return {"features": base.features, "pool": base.avgpool, "feat_dim": feat_dim}


class MaskEncoder(nn.Module):
    """
    front/top mask 2장을 합쳐서 간단한 shape-aware 특징을 뽑습니다.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.flatten(x, 1)


class GeometryMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructureAwareFusionNet(nn.Module):
    """
    RGB backbone + mask encoder + geometry MLP를 함께 융합합니다.
    추가로 mask confidence 기반 gating과 simulation auxiliary head를 둡니다.
    """
    def __init__(self, geom_dim: int, pretrained: bool = True, dropout: float = 0.30):
        super().__init__()
        spec = build_backbone("efficientnet_v2_s", pretrained=pretrained)
        feat_dim = int(spec["feat_dim"])
        hidden = 384

        self.features = spec["features"]
        self.pool = spec["pool"]

        self.front_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.top_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mask_encoder = MaskEncoder(out_dim=128)
        self.geom_encoder = GeometryMLP(in_dim=geom_dim, out_dim=128, dropout=0.20)

        rgb_fuse_dim = hidden * 4
        self.rgb_pair_proj = nn.Sequential(
            nn.Linear(rgb_fuse_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mask_meta_encoder = nn.Sequential(
            nn.Linear(MASK_META_DIM, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(MASK_META_DIM, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        self.aux_geom_head = nn.Linear(256, geom_dim)
        self.sim_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, SIM_TARGET_DIM),
        )

        self.classifier = nn.Sequential(
            nn.Linear(rgb_fuse_dim + 128 + 128 + 32, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode_rgb(self, front: torch.Tensor, top: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([front, top], dim=0)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        front_feat, top_feat = x.chunk(2, dim=0)
        return front_feat, top_feat

    def forward(
        self,
        front: torch.Tensor,
        top: torch.Tensor,
        front_mask: torch.Tensor,
        top_mask: torch.Tensor,
        geom: torch.Tensor,
        mask_meta: torch.Tensor,
    ):
        front_feat, top_feat = self.encode_rgb(front, top)
        front_feat = self.front_proj(front_feat)
        top_feat = self.top_proj(top_feat)

        rgb_fused = torch.cat([front_feat, top_feat, torch.abs(front_feat - top_feat), front_feat * top_feat], dim=1)
        rgb_pair = self.rgb_pair_proj(rgb_fused)

        mask_fused = torch.cat([front_mask, top_mask], dim=1)
        mask_feat = self.mask_encoder(mask_fused)
        geom_feat = self.geom_encoder(geom)

        gate_raw = torch.sigmoid(self.gate_head(mask_meta))
        mean_conf = mask_meta[:, 2:3]
        mask_gate = 0.15 + 0.85 * gate_raw[:, 0:1] * mean_conf
        geom_gate = 0.15 + 0.85 * gate_raw[:, 1:2] * mean_conf

        mask_feat = mask_feat * mask_gate
        geom_feat = geom_feat * geom_gate
        meta_feat = self.mask_meta_encoder(mask_meta)

        logits = self.classifier(torch.cat([rgb_fused, mask_feat, geom_feat, meta_feat], dim=1))
        aux_geom = self.aux_geom_head(rgb_pair)
        aux_sim = self.sim_head(rgb_pair)
        return {
            "logits": logits,
            "aux_geom": aux_geom,
            "aux_sim": aux_sim,
            "mask_gate": mask_gate,
            "geom_gate": geom_gate,
        }


# ============================================================
# Training config and loaders
# ============================================================

@dataclass
class Config:
    run_name: str = "hybrid_structure_aware_gated_simaux"
    mode: str = "holdout"
    data_root: str = "./open"
    save_dir: str = "./runs_hybrid_structure_aware_gated_simaux"

    img_size: int = 384
    batch_size: int = 12
    epochs: int = 14
    nfolds: int = 5
    seed: int = 42
    num_workers: int = 4

    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.30
    label_smoothing: float = 0.03
    aug_profile: str = "base"
    crop_profile: str = "none"
    crop_margin_ratio: float = 0.18
    crop_min_side_ratio: float = 0.42

    patience: int = 4
    warmup_pct: float = 0.10

    use_amp: bool = True
    pin_memory: bool = True
    tta_hflip: bool = True
    temperature_scaling: bool = True
    class_balance: bool = False
    check_paths: bool = True

    grid_xy: int = 16
    grid_z: int = 20
    geom_aux_weight: float = 0.10
    sim_aux_weight: float = 0.08
    sim_frame_stride: int = 5
    export_debug_samples: int = 0

    def short(self) -> Dict:
        return asdict(self)


def build_train_sampler(df: pd.DataFrame, cfg: Config) -> Optional[WeightedRandomSampler]:
    if not cfg.class_balance or "label" not in df.columns:
        return None

    counts = df["label"].value_counts().to_dict()
    class_weight = {k: len(df) / max(v, 1) for k, v in counts.items()}
    weights = df["label"].map(class_weight).astype(np.float64).values
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(df), replacement=True)


def build_loader(df: pd.DataFrame, transform, cfg: Config, device: torch.device, is_test: bool, shuffle: bool, sampler=None) -> DataLoader:
    ds = StructureAwareDataset(df, transform=transform, is_test=is_test, cache_masks=True, sim_frame_stride=cfg.sim_frame_stride)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker_factory(cfg.seed),
        generator=generator if not is_test else None,
        drop_last=False,
    )


# ============================================================
# Loss and calibration
# ============================================================

def smooth_targets(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def classification_loss(logits: torch.Tensor, y: torch.Tensor, smoothing: float) -> torch.Tensor:
    y_sm = smooth_targets(y, smoothing=smoothing)
    return F.binary_cross_entropy_with_logits(logits, y_sm)


def fit_temperature_grid(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    raw_p = sigmoid_np(logits)
    raw_ll = dacon_logloss(labels, raw_p)

    coarse = np.exp(np.linspace(np.log(0.5), np.log(5.0), 81))
    losses = []
    for t in coarse:
        p = sigmoid_np(logits / t)
        losses.append(dacon_logloss(labels, p))
    best_idx = int(np.argmin(losses))
    best_t = float(coarse[best_idx])
    best_ll = float(losses[best_idx])

    left = max(best_t / 1.5, 0.25)
    right = min(best_t * 1.5, 10.0)
    for t in np.linspace(left, right, 81):
        p = sigmoid_np(logits / t)
        ll = dacon_logloss(labels, p)
        if ll < best_ll:
            best_t = float(t)
            best_ll = float(ll)
    return best_t, raw_ll, best_ll


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid_np(np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6))


# ============================================================
# Train / validate / infer
# ============================================================

class EpochOutput:
    def __init__(self):
        self.train_loss = math.nan
        self.valid_bce = math.nan
        self.valid_logloss = math.nan
        self.valid_logloss_cal = math.nan
        self.valid_acc = math.nan
        self.temperature = 1.0
        self.valid_logits = None
        self.valid_labels = None
        self.valid_ids = None
        self.valid_probs_raw = None
        self.valid_probs_cal = None


def build_optimizer_and_scheduler(model: StructureAwareFusionNet, cfg: Config, steps_per_epoch: int):
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": cfg.backbone_lr},
            {"params": model.front_proj.parameters(), "lr": cfg.learning_rate},
            {"params": model.top_proj.parameters(), "lr": cfg.learning_rate},
            {"params": model.mask_encoder.parameters(), "lr": cfg.learning_rate},
            {"params": model.geom_encoder.parameters(), "lr": cfg.learning_rate},
            {"params": model.mask_meta_encoder.parameters(), "lr": cfg.learning_rate},
            {"params": model.gate_head.parameters(), "lr": cfg.learning_rate},
            {"params": model.rgb_pair_proj.parameters(), "lr": cfg.learning_rate},
            {"params": model.aux_geom_head.parameters(), "lr": cfg.learning_rate},
            {"params": model.sim_head.parameters(), "lr": cfg.learning_rate},
            {"params": model.classifier.parameters(), "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            cfg.backbone_lr,
            cfg.learning_rate, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate,
            cfg.learning_rate, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate,
            cfg.learning_rate, cfg.learning_rate,
        ],
        epochs=max(cfg.epochs, 1),
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=float(cfg.warmup_pct),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg: Config, device: torch.device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for front, top, front_mask, top_mask, geom, mask_meta, sim_target, sim_valid, y, _ in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        front_mask = front_mask.to(device, non_blocking=True)
        top_mask = top_mask.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)
        mask_meta = mask_meta.to(device, non_blocking=True)
        sim_target = sim_target.to(device, non_blocking=True)
        sim_valid = sim_valid.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            out = model(front, top, front_mask, top_mask, geom, mask_meta)
            cls_loss = classification_loss(out["logits"], y, smoothing=cfg.label_smoothing)
            aux_loss = F.smooth_l1_loss(out["aux_geom"], geom)
            if sim_valid.sum() > 0:
                sim_mask = (sim_valid.reshape(-1) > 0.5)
                sim_loss = F.smooth_l1_loss(out["aux_sim"][sim_mask], sim_target[sim_mask])
            else:
                sim_loss = torch.zeros([], device=device, dtype=out["logits"].dtype)
            loss = cls_loss + cfg.geom_aux_weight * aux_loss + cfg.sim_aux_weight * sim_loss

        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        batch_size = front.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, cfg: Config, device: torch.device, use_amp: bool) -> EpochOutput:
    model.eval()
    out = EpochOutput()

    total_bce = 0.0
    total_count = 0
    logits_all = []
    labels_all = []
    ids_all = []

    for front, top, front_mask, top_mask, geom, mask_meta, sim_target, sim_valid, y, sample_ids in loader:
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        front_mask = front_mask.to(device, non_blocking=True)
        top_mask = top_mask.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)
        mask_meta = mask_meta.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_context(device.type, use_amp):
            pred = model(front, top, front_mask, top_mask, geom, mask_meta)["logits"]
            bce = F.binary_cross_entropy_with_logits(pred, y)

        total_bce += float(bce.item()) * front.size(0)
        total_count += front.size(0)
        logits_all.extend(pred.float().cpu().numpy().reshape(-1).tolist())
        labels_all.extend(y.float().cpu().numpy().reshape(-1).tolist())
        ids_all.extend(list(sample_ids))

    out.valid_bce = total_bce / max(total_count, 1)
    out.valid_logits = np.asarray(logits_all, dtype=np.float64)
    out.valid_labels = np.asarray(labels_all, dtype=np.float64)
    out.valid_ids = np.asarray(ids_all)
    out.valid_probs_raw = sigmoid_np(out.valid_logits)
    out.valid_logloss = dacon_logloss(out.valid_labels, out.valid_probs_raw)
    out.valid_acc = binary_accuracy(out.valid_labels, out.valid_probs_raw)

    if cfg.temperature_scaling:
        temp, _, cal_ll = fit_temperature_grid(out.valid_logits, out.valid_labels)
        out.temperature = temp
        out.valid_probs_cal = apply_temperature_to_logits(out.valid_logits, temp)
        out.valid_logloss_cal = cal_ll
    else:
        out.temperature = 1.0
        out.valid_probs_cal = out.valid_probs_raw.copy()
        out.valid_logloss_cal = out.valid_logloss
    return out


@torch.no_grad()
def infer_logits(model, loader, device: torch.device, use_amp: bool, tta_hflip: bool) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    logits_all = []
    ids_all = []

    for front, top, front_mask, top_mask, geom, mask_meta, sample_ids in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        front_mask = front_mask.to(device, non_blocking=True)
        top_mask = top_mask.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)
        mask_meta = mask_meta.to(device, non_blocking=True)

        with autocast_context(device.type, use_amp):
            logits = model(front, top, front_mask, top_mask, geom, mask_meta)["logits"]
            if tta_hflip:
                logits_flip = model(
                    torch.flip(front, dims=[3]),
                    torch.flip(top, dims=[3]),
                    torch.flip(front_mask, dims=[3]),
                    torch.flip(top_mask, dims=[3]),
                    geom,
                    mask_meta,
                )["logits"]
                logits = (logits + logits_flip) / 2.0

        logits_all.extend(logits.float().cpu().numpy().reshape(-1).tolist())
        ids_all.extend(list(sample_ids))
    return np.asarray(logits_all, dtype=np.float64), ids_all


def compute_source_metrics(df: pd.DataFrame, prob_col: str = "unstable_prob") -> Dict[str, Dict[str, float]]:
    metrics = {}
    for source_name, sub in df.groupby("source"):
        if "label_float" not in sub.columns or sub["label_float"].isna().all():
            continue
        y = sub["label_float"].values.astype(np.float64)
        p = sub[prob_col].values.astype(np.float64)
        metrics[source_name] = {
            "logloss": dacon_logloss(y, p),
            "accuracy": binary_accuracy(y, p),
            "n": int(len(sub)),
        }
    return metrics


# ============================================================
# Core train / holdout / CV
# ============================================================

def export_debug_samples(df: pd.DataFrame, run_dir: str, split_name: str, max_samples: int = 10) -> None:
    if max_samples <= 0:
        return
    debug_dir = os.path.join(run_dir, "debug_masks", split_name)
    ensure_dir(debug_dir)

    for _, row in df.head(max_samples).iterrows():
        front = Image.open(row["front_path"]).convert("RGB")
        top = Image.open(row["top_path"]).convert("RGB")
        fm, fc = extract_object_mask_and_confidence_from_pil(front, view="front")
        tm, tc = extract_object_mask_and_confidence_from_pil(top, view="top")
        panel = make_mask_debug_triptych(front, top, fm, tm)
        panel.save(os.path.join(debug_dir, f"{row['id']}_mask_triptych.png"))


def train_with_validation(
    cfg: Config,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
    fold_idx: Optional[int] = None,
) -> Dict:
    seed_everything(cfg.seed + (0 if fold_idx is None else fold_idx * 1000))

    train_tf = StructureAwarePairTrainTransform(
        cfg.img_size,
        cfg.grid_xy,
        cfg.grid_z,
        aug_profile=cfg.aug_profile,
        crop_profile=cfg.crop_profile,
        crop_margin_ratio=cfg.crop_margin_ratio,
        crop_min_side_ratio=cfg.crop_min_side_ratio,
    )
    eval_tf = StructureAwarePairEvalTransform(
        cfg.img_size,
        cfg.grid_xy,
        cfg.grid_z,
        crop_profile=cfg.crop_profile,
        crop_margin_ratio=cfg.crop_margin_ratio,
        crop_min_side_ratio=cfg.crop_min_side_ratio,
    )

    sampler = build_train_sampler(train_df, cfg)
    train_loader = build_loader(train_df, train_tf, cfg, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    valid_loader = build_loader(valid_df, eval_tf, cfg, device=device, is_test=False, shuffle=False, sampler=None)
    test_loader = build_loader(test_df, eval_tf, cfg, device=device, is_test=True, shuffle=False, sampler=None)

    model = StructureAwareFusionNet(geom_dim=GEOM_FEATURE_DIM, pretrained=True, dropout=cfg.dropout).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))
    scaler = build_grad_scaler(device.type, enabled=use_amp)

    best_score = float("inf")
    best_epoch = -1
    best_state = None
    best_epoch_output = None
    epochs_no_improve = 0
    epoch_records = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, device, use_amp)
        valid_out = validate_one_epoch(model, valid_loader, cfg, device, use_amp)
        valid_score = valid_out.valid_logloss_cal

        epoch_records.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_bce": float(valid_out.valid_bce),
            "valid_logloss_raw": float(valid_out.valid_logloss),
            "valid_logloss_cal": float(valid_out.valid_logloss_cal),
            "valid_acc": float(valid_out.valid_acc),
            "temperature": float(valid_out.temperature),
        })

        tag = "holdout" if fold_idx is None else f"fold {fold_idx}"
        print(
            f"[{cfg.run_name} | {tag}] Epoch {epoch:02d}/{cfg.epochs} | "
            f"TrainLoss {train_loss:.5f} | ValidBCE {valid_out.valid_bce:.5f} | "
            f"RawLL {valid_out.valid_logloss:.5f} | CalLL {valid_out.valid_logloss_cal:.5f} | "
            f"Acc {valid_out.valid_acc:.5f} | T {valid_out.temperature:.3f}"
        )

        if valid_score < best_score:
            best_score = valid_score
            best_epoch = epoch
            best_epoch_output = valid_out
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "temperature": float(valid_out.temperature),
                "best_valid_logloss_raw": float(valid_out.valid_logloss),
                "best_valid_logloss_cal": float(valid_out.valid_logloss_cal),
                "cfg": cfg.short(),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[{cfg.run_name}] early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    train_seconds = time.time() - t0

    ckpt_path = os.path.join(run_dir, f"best_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.pt")
    torch.save(best_state, ckpt_path)
    pd.DataFrame(epoch_records).to_csv(
        os.path.join(run_dir, f"epochs_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        index=False,
    )

    best_model = StructureAwareFusionNet(geom_dim=GEOM_FEATURE_DIM, pretrained=False, dropout=cfg.dropout).to(device)
    best_model.load_state_dict(best_state["model"])

    test_logits, test_ids = infer_logits(best_model, test_loader, device=device, use_amp=use_amp, tta_hflip=cfg.tta_hflip)
    test_probs_raw = sigmoid_np(test_logits)
    test_probs_cal = apply_temperature_to_logits(test_logits, best_state["temperature"])

    valid_pred_df = valid_df[["id", "source", "label", "label_float", "front_path", "top_path"]].reset_index(drop=True).copy()
    valid_pred_df["logit"] = best_epoch_output.valid_logits
    valid_pred_df["unstable_prob_raw"] = best_epoch_output.valid_probs_raw
    valid_pred_df["unstable_prob"] = best_epoch_output.valid_probs_cal
    valid_pred_df["stable_prob"] = 1.0 - valid_pred_df["unstable_prob"]
    valid_pred_df["temperature"] = float(best_state["temperature"])
    valid_pred_df["sample_logloss"] = per_sample_logloss(valid_pred_df["label_float"].values, valid_pred_df["unstable_prob"].values)
    valid_pred_csv = os.path.join(run_dir, f"valid_predictions_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv")
    valid_pred_df.to_csv(valid_pred_csv, index=False)

    test_pred_df = pd.DataFrame({
        "id": test_ids,
        "logit": test_logits,
        "unstable_prob_raw": test_probs_raw,
        "unstable_prob": test_probs_cal,
        "stable_prob": 1.0 - test_probs_cal,
        "temperature": float(best_state["temperature"]),
    })
    test_pred_csv = os.path.join(run_dir, f"test_predictions_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv")
    test_pred_df.to_csv(test_pred_csv, index=False)

    metrics = {
        "run_name": cfg.run_name,
        "mode": cfg.mode,
        "fold": None if fold_idx is None else int(fold_idx),
        "backbone": "efficientnet_v2_s",
        "img_size": cfg.img_size,
        "seed": cfg.seed,
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_seconds),
        "temperature": float(best_state["temperature"]),
        "valid_logloss_raw": float(best_state["best_valid_logloss_raw"]),
        "valid_logloss_cal": float(best_state["best_valid_logloss_cal"]),
        "valid_accuracy": float(binary_accuracy(valid_df["label_float"].values, valid_pred_df["unstable_prob"].values)),
        "source_metrics": compute_source_metrics(valid_pred_df, prob_col="unstable_prob"),
        "checkpoint": ckpt_path,
        "pred_valid_csv": valid_pred_csv,
        "pred_test_csv": test_pred_csv,
    }
    save_json(os.path.join(run_dir, f"result_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.json"), metrics)

    return {
        "metrics": metrics,
        "valid_df": valid_pred_df,
        "test_df": test_pred_df,
        "state": best_state,
    }


def run_holdout(cfg: Config, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict:
    result = train_with_validation(cfg, train_df, dev_df, test_df, run_dir, device, use_amp, fold_idx=None)
    submission = result["test_df"][["id", "unstable_prob", "stable_prob"]].sort_values("id").reset_index(drop=True)
    submission_filename = "submission_holdout.csv"
    submission_path = os.path.join(run_dir, submission_filename)
    submission_alias_path = os.path.join(run_dir, f"{cfg.run_name}__{submission_filename}")
    submission.to_csv(submission_path, index=False)
    submission.to_csv(submission_alias_path, index=False)
    summary = {
        "run_name": cfg.run_name,
        "mode": "holdout",
        "submission": submission_path,
        "submission_alias": submission_alias_path,
        "valid_logloss_cal": float(result["metrics"]["valid_logloss_cal"]),
        "dev_logloss_cal": float(result["metrics"]["source_metrics"].get("dev", {}).get("logloss", result["metrics"]["valid_logloss_cal"])),
        "valid_accuracy": float(result["metrics"]["valid_accuracy"]),
    }
    save_json(os.path.join(run_dir, "holdout_summary.json"), summary)
    return {"summary": summary, "result": result, "submission_df": submission}


def run_cv(cfg: Config, full_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict:
    strat_target = choose_stratify_target(full_df, cfg.nfolds)
    skf = StratifiedKFold(n_splits=cfg.nfolds, shuffle=True, random_state=cfg.seed)

    oof_frames = []
    fold_metrics = []
    test_pred_frames = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(full_df, strat_target)):
        tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
        va_df = full_df.iloc[va_idx].reset_index(drop=True)
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.seed = cfg.seed + fold_idx

        result = train_with_validation(fold_cfg, tr_df, va_df, test_df, run_dir, device, use_amp, fold_idx=fold_idx)

        valid_df = result["valid_df"].copy()
        valid_df["fold"] = fold_idx
        oof_frames.append(valid_df)
        fold_metrics.append(result["metrics"])

        test_fold = result["test_df"].copy()
        test_fold["fold"] = fold_idx
        test_fold["weight_equal"] = 1.0
        test_fold["weight_inv_cal_ll"] = 1.0 / max(result["metrics"]["valid_logloss_cal"], 1e-8)
        test_pred_frames.append(test_fold)

    fold_metrics_df = pd.DataFrame([{**{k: v for k, v in m.items() if k != "source_metrics"}, "source_metrics": json.dumps(m["source_metrics"], ensure_ascii=False)} for m in fold_metrics])
    fold_metrics_csv = os.path.join(run_dir, "fold_metrics.csv")
    fold_metrics_df.to_csv(fold_metrics_csv, index=False)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(["source", "id"]).reset_index(drop=True)
    oof_csv = os.path.join(run_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv, index=False)
    oof_df[oof_df["source"] == "train"].to_csv(os.path.join(run_dir, "oof_predictions_train.csv"), index=False)
    oof_df[oof_df["source"] == "dev"].to_csv(os.path.join(run_dir, "oof_predictions_dev.csv"), index=False)

    oof_logloss_cal = dacon_logloss(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    oof_accuracy = binary_accuracy(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    source_metrics = compute_source_metrics(oof_df, prob_col="unstable_prob")

    test_stack = pd.concat(test_pred_frames, axis=0, ignore_index=True)

    def blend_from_weights(weight_col: str) -> pd.DataFrame:
        parts = []
        for sid, sub in test_stack.groupby("id"):
            w = sub[weight_col].values.astype(np.float64)
            p = sub["unstable_prob"].values.astype(np.float64)
            w = w / w.sum()
            parts.append({
                "id": sid,
                "unstable_prob": float(np.sum(w * p)),
                "stable_prob": float(np.sum(w * (1.0 - p))),
            })
        return pd.DataFrame(parts).sort_values("id").reset_index(drop=True)

    sub_equal = blend_from_weights("weight_equal")
    sub_weighted = blend_from_weights("weight_inv_cal_ll")
    sub_equal_filename = "submission_cv_equal.csv"
    sub_weighted_filename = "submission_cv_weighted.csv"
    sub_equal_path = os.path.join(run_dir, sub_equal_filename)
    sub_weighted_path = os.path.join(run_dir, sub_weighted_filename)
    sub_equal_alias_path = os.path.join(run_dir, f"{cfg.run_name}__{sub_equal_filename}")
    sub_weighted_alias_path = os.path.join(run_dir, f"{cfg.run_name}__{sub_weighted_filename}")
    sub_equal.to_csv(sub_equal_path, index=False)
    sub_weighted.to_csv(sub_weighted_path, index=False)
    sub_equal.to_csv(sub_equal_alias_path, index=False)
    sub_weighted.to_csv(sub_weighted_alias_path, index=False)

    summary = {
        "run_name": cfg.run_name,
        "mode": "cv",
        "oof_logloss_cal": float(oof_logloss_cal),
        "oof_accuracy": float(oof_accuracy),
        "source_metrics": source_metrics,
        "fold_mean_valid_logloss_cal": float(fold_metrics_df["valid_logloss_cal"].mean()),
        "fold_mean_temperature": float(fold_metrics_df["temperature"].mean()),
        "oof_csv": oof_csv,
        "fold_metrics_csv": fold_metrics_csv,
        "submission_equal": sub_equal_path,
        "submission_weighted": sub_weighted_path,
        "submission_equal_alias": sub_equal_alias_path,
        "submission_weighted_alias": sub_weighted_alias_path,
    }
    save_json(os.path.join(run_dir, "cv_summary.json"), summary)
    return {
        "summary": summary,
        "oof_df": oof_df,
        "test_equal_df": sub_equal,
        "test_weighted_df": sub_weighted,
        "fold_metrics_df": fold_metrics_df,
    }


# ============================================================
# Main runner
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid structure-aware trainer with confidence gating + simulation aux")
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_hybrid_structure_aware_gated_simaux")
    parser.add_argument("--run_name", type=str, default="hybrid_structure_aware_gated_simaux")

    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--aug_profile", choices=["base", "strong_domain", "crop_tuned"], default="base")
    parser.add_argument("--crop_profile", choices=["none", "mask_bbox"], default="none")
    parser.add_argument("--crop_margin_ratio", type=float, default=0.18)
    parser.add_argument("--crop_min_side_ratio", type=float, default=0.42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup_pct", type=float, default=0.10)

    parser.add_argument("--grid_xy", type=int, default=16)
    parser.add_argument("--grid_z", type=int, default=20)
    parser.add_argument("--geom_aux_weight", type=float, default=0.10)
    parser.add_argument("--sim_aux_weight", type=float, default=0.08)
    parser.add_argument("--sim_frame_stride", type=int, default=5)
    parser.add_argument("--export_debug_samples", type=int, default=0)

    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--tta_hflip", action="store_true", default=True)
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")
    parser.add_argument("--temperature_scaling", action="store_true", default=True)
    parser.add_argument("--no_temperature_scaling", action="store_false", dest="temperature_scaling")
    parser.add_argument("--class_balance", action="store_true", default=False)
    parser.add_argument("--check_paths", action="store_true", default=True)
    parser.add_argument("--no_check_paths", action="store_false", dest="check_paths")
    return parser


def make_config_from_args(ns: argparse.Namespace) -> Config:
    return Config(
        run_name=ns.run_name,
        mode=ns.mode,
        data_root=ns.data_root,
        save_dir=ns.save_dir,
        img_size=ns.img_size,
        batch_size=ns.batch_size,
        epochs=ns.epochs,
        nfolds=ns.nfolds,
        seed=ns.seed,
        num_workers=ns.num_workers,
        learning_rate=ns.learning_rate,
        backbone_lr=ns.backbone_lr,
        weight_decay=ns.weight_decay,
        dropout=ns.dropout,
        label_smoothing=ns.label_smoothing,
        aug_profile=ns.aug_profile,
        crop_profile=ns.crop_profile,
        crop_margin_ratio=ns.crop_margin_ratio,
        crop_min_side_ratio=ns.crop_min_side_ratio,
        patience=ns.patience,
        warmup_pct=ns.warmup_pct,
        use_amp=ns.use_amp,
        pin_memory=ns.pin_memory,
        tta_hflip=ns.tta_hflip,
        temperature_scaling=ns.temperature_scaling,
        class_balance=ns.class_balance,
        check_paths=ns.check_paths,
        grid_xy=ns.grid_xy,
        grid_z=ns.grid_z,
        geom_aux_weight=ns.geom_aux_weight,
        sim_aux_weight=ns.sim_aux_weight,
        sim_frame_stride=ns.sim_frame_stride,
        export_debug_samples=ns.export_debug_samples,
    )


def run_baseline(cfg: Config) -> str:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp and device.type == "cuda")

    resolved_root = resolve_data_root(cfg.data_root)
    print(f"[Info] resolved data_root: {resolved_root}")
    cfg.data_root = resolved_root

    train_df = load_split_df(cfg.data_root, "train")
    dev_df = load_split_df(cfg.data_root, "dev")
    test_df = load_split_df(cfg.data_root, "test")
    full_df = pd.concat([train_df, dev_df], ignore_index=True)

    if cfg.check_paths:
        verify_paths(train_df)
        verify_paths(dev_df)
        verify_paths(test_df)

    run_dir = os.path.join(cfg.save_dir, f"{cfg.run_name}_{now_str()}")
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "run_config.json"), cfg.short())

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Mode: {cfg.mode}")
    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)} | Test samples: {len(test_df)}")
    print(f"Run dir: {run_dir}")

    export_debug_samples(train_df, run_dir, "train", max_samples=cfg.export_debug_samples)
    export_debug_samples(dev_df, run_dir, "dev", max_samples=cfg.export_debug_samples)
    export_debug_samples(test_df, run_dir, "test", max_samples=cfg.export_debug_samples)

    if cfg.mode == "holdout":
        result = run_holdout(cfg, train_df, dev_df, test_df, run_dir, device, use_amp)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        result = run_cv(cfg, full_df, test_df, run_dir, device, use_amp)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    return run_dir


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = make_config_from_args(args)
    run_dir = run_baseline(cfg)
    print(f"\nDone. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
