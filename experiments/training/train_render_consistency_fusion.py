
"""
Render-consistency fusion trainer for DACON voxel stability task.

핵심 아이디어
1) front / top 원본 RGB에서 구조물 mask를 추출합니다.
2) front silhouette + top footprint를 교집합하여 visual-hull 기반 pseudo-3D occupancy를 만듭니다.
3) occupancy로부터 다시 canonical front / top rendered view를 생성합니다.
4) 원본 2-view 특징과 rendered 2-view 특징을 함께 학습하고,
   두 branch의 예측이 서로 크게 어긋나지 않도록 consistency loss를 줍니다.

주의
- 여기서 만드는 3D는 완전한 metric reconstruction이 아니라 weak visual hull 입니다.
- 따라서 rendered view는 photorealistic 정답이 아니라 구조 prior를 주는 보조 view로 쓰는 것이 핵심입니다.
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
from PIL import Image, ImageDraw, ImageFilter, ImageOps
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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_FLOAT = {"stable": 0.0, "unstable": 1.0}


# ============================================================
# Utilities
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
    if not enabled:
        yield
        return
    try:
        with torch.amp.autocast(device_type=device_type, enabled=True):
            yield
    except Exception:
        if device_type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
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
# Mask extraction
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


def extract_object_mask_from_pil(img: Image.Image, view: str) -> Image.Image:
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

    mask = _keep_best_component(mask.astype(bool), center_x=cx, center_y=cy)

    if ndi is not None:
        mask = ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        mask = ndi.binary_fill_holes(mask)

    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")


def _resize_mask_array(mask: np.ndarray, out_w: int, out_h: int, threshold: float = 0.20) -> np.ndarray:
    pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    arr = np.asarray(pil.resize((out_w, out_h), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    return (arr >= threshold).astype(np.uint8)


# ============================================================
# Pseudo-3D and rendering
# ============================================================

def build_visual_hull(front_mask: np.ndarray, top_mask: np.ndarray, grid_xy: int = 16, grid_z: int = 20):
    front_grid = _resize_mask_array(front_mask, grid_xy, grid_z, threshold=0.18)   # [z, x]
    top_grid_hw = _resize_mask_array(top_mask, grid_xy, grid_xy, threshold=0.18)   # [y, x]
    top_grid = top_grid_hw.T                                                        # [x, y]
    occ = (front_grid[:, :, None] & top_grid[None, :, :]).astype(np.uint8)         # [z, x, y]
    return occ, front_grid, top_grid_hw


def _checkerboard_canvas(size: int, board_tiles: int = 8, with_sky: bool = False) -> Image.Image:
    img = Image.new("RGB", (size, size), (240, 243, 252))
    draw = ImageDraw.Draw(img)

    if with_sky:
        horizon = int(size * 0.38)

        # 하늘 그라데이션
        for y in range(horizon):
            t = y / max(horizon - 1, 1)
            r = int((1 - t) * 214 + t * 232)
            g = int((1 - t) * 214 + t * 232)
            b = int((1 - t) * 233 + t * 245)
            draw.line([(0, y), (size, y)], fill=(r, g, b))

        tile = max((size - horizon) // board_tiles, 1)
        c1 = (232, 238, 252)
        c2 = (181, 202, 236)

        for gy in range(board_tiles + 2):
            y0 = horizon + gy * tile
            y1 = min(size, horizon + (gy + 1) * tile)

            # 캔버스 바깥이면 건너뜀
            if y0 >= size:
                continue
            if y1 <= y0:
                continue

            for gx in range(board_tiles + 2):
                x0 = gx * tile
                x1 = min(size, (gx + 1) * tile)

                if x0 >= size:
                    continue
                if x1 <= x0:
                    continue

                color = c1 if (gx + gy) % 2 == 0 else c2
                draw.rectangle([x0, y0, x1, y1], fill=color)

    else:
        tile = max(size // board_tiles, 1)
        c1 = (239, 245, 255)
        c2 = (184, 205, 238)

        for gy in range(board_tiles + 2):
            y0 = gy * tile
            y1 = min(size, (gy + 1) * tile)

            if y0 >= size:
                continue
            if y1 <= y0:
                continue

            for gx in range(board_tiles + 2):
                x0 = gx * tile
                x1 = min(size, (gx + 1) * tile)

                if x0 >= size:
                    continue
                if x1 <= x0:
                    continue

                color = c1 if (gx + gy) % 2 == 0 else c2
                draw.rectangle([x0, y0, x1, y1], fill=color)

    return img


def _darken(rgb: np.ndarray, factor: float) -> Tuple[int, int, int]:
    arr = np.clip(rgb.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
    return int(arr[0]), int(arr[1]), int(arr[2])


def render_top_from_occ(
    occ: np.ndarray,
    top_img: Image.Image,
    size: int,
) -> Image.Image:
    grid_xy = occ.shape[1]
    top_colors = np.asarray(top_img.resize((grid_xy, grid_xy), resample=Image.BILINEAR), dtype=np.uint8)
    height_map = np.zeros((grid_xy, grid_xy), dtype=np.int32)  # [x, y]
    for z in range(occ.shape[0]):
        height_map = np.where(occ[z] > 0, z + 1, height_map)

    canvas = _checkerboard_canvas(size=size, board_tiles=8, with_sky=False)
    draw = ImageDraw.Draw(canvas)
    cell = max(int(size * 0.46 / grid_xy), 4)
    x_start = (size - grid_xy * cell) // 2
    y_start = (size - grid_xy * cell) // 2
    max_h = max(int(height_map.max()), 1)

    # 간단한 그림자
    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    shadow_offset = int(cell * 0.45)

    for x in range(grid_xy):
        for y in range(grid_xy):
            h = int(height_map[x, y])
            if h <= 0:
                continue
            x0 = x_start + x * cell
            y0 = y_start + y * cell
            sdraw.rectangle([x0 + shadow_offset, y0 + shadow_offset, x0 + cell + shadow_offset, y0 + cell + shadow_offset], fill=(80, 80, 80, 32))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    for x in range(grid_xy):
        for y in range(grid_xy):
            h = int(height_map[x, y])
            if h <= 0:
                continue
            x0 = x_start + x * cell
            y0 = y_start + y * cell
            x1 = x0 + cell
            y1 = y0 + cell
            base_color = top_colors[y, x].astype(np.float32)
            shade = 0.62 + 0.38 * (h / max_h)
            fill = _darken(base_color, shade)
            edge = _darken(base_color, max(0.45, shade - 0.18))
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=edge, width=max(1, cell // 10))
    return canvas


def render_front_from_occ(
    occ: np.ndarray,
    front_img: Image.Image,
    size: int,
) -> Image.Image:
    grid_z, grid_xy, _ = occ.shape
    front_colors = np.asarray(front_img.resize((grid_xy, grid_z), resample=Image.BILINEAR), dtype=np.uint8)
    depth_map = occ.sum(axis=2)  # [z, x]

    canvas = _checkerboard_canvas(size=size, board_tiles=8, with_sky=True)
    draw = ImageDraw.Draw(canvas)
    cell_w = max(int(size * 0.44 / grid_xy), 4)
    cell_h = max(int(size * 0.56 / grid_z), 4)
    x_start = (size - grid_xy * cell_w) // 2
    floor_y = int(size * 0.80)
    y_start = floor_y - grid_z * cell_h
    max_d = max(int(depth_map.max()), 1)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    for z in range(grid_z):
        for x in range(grid_xy):
            d = int(depth_map[z, x])
            if d <= 0:
                continue
            x0 = x_start + x * cell_w
            y0 = y_start + z * cell_h
            # 바닥 쪽으로 눌린 단순 그림자
            sx0 = x0 + int(0.45 * cell_w)
            sy0 = floor_y + int(0.10 * cell_h)
            sx1 = sx0 + int(cell_w * (0.7 + 0.3 * d / max_d))
            sy1 = sy0 + int(cell_h * (0.3 + 0.3 * d / max_d))
            sdraw.rectangle([sx0, sy0, sx1, sy1], fill=(70, 70, 70, 28))

    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # 앞에서 보이게 아래층부터 위층 순서로 draw
    for z in range(grid_z):
        for x in range(grid_xy):
            d = int(depth_map[z, x])
            if d <= 0:
                continue
            x0 = x_start + x * cell_w
            y0 = y_start + z * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            base_color = front_colors[z, x].astype(np.float32)
            shade = 0.68 + 0.32 * (d / max_d)
            fill = _darken(base_color, shade)
            edge = _darken(base_color, max(0.45, shade - 0.18))
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=edge, width=max(1, cell_w // 10))
    return canvas


def build_rendered_views(front_img: Image.Image, top_img: Image.Image, front_mask: Image.Image, top_mask: Image.Image, grid_xy: int, grid_z: int, render_size: int):
    front_mask_np = (np.asarray(front_mask, dtype=np.uint8) > 127).astype(np.uint8)
    top_mask_np = (np.asarray(top_mask, dtype=np.uint8) > 127).astype(np.uint8)
    occ, _, _ = build_visual_hull(front_mask_np, top_mask_np, grid_xy=grid_xy, grid_z=grid_z)
    render_front = render_front_from_occ(occ, front_img, size=render_size)
    render_top = render_top_from_occ(occ, top_img, size=render_size)
    return render_front, render_top


def make_render_debug_panel(front: Image.Image, top: Image.Image, render_front: Image.Image, render_top: Image.Image) -> Image.Image:
    width = front.width
    height = front.height
    panel = Image.new("RGB", (width * 4, height), (245, 245, 245))
    panel.paste(front.convert("RGB"), (0, 0))
    panel.paste(top.convert("RGB"), (width, 0))
    panel.paste(render_front.convert("RGB"), (width * 2, 0))
    panel.paste(render_top.convert("RGB"), (width * 3, 0))
    return panel


# ============================================================
# Transforms
# ============================================================

class RenderFusionTrainTransform:
    """
    원본 front/top 와 rendered front/top 에 같은 기하 증강을 적용합니다.
    photometric 증강은 원본 RGB에 더 강하게, rendered 쪽에는 약하게 적용합니다.
    """
    def __init__(self, img_size: int):
        self.img_size = int(img_size)

    def _resize(self, img: Image.Image) -> Image.Image:
        return TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)

    def _geometric_params(self) -> Tuple[float, Tuple[int, int], float]:
        angle = random.uniform(-6.0, 6.0)
        max_shift = int(self.img_size * 0.05)
        translate = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
        scale = random.uniform(0.95, 1.05)
        return angle, translate, scale

    def __call__(self, front: Image.Image, top: Image.Image, render_front: Image.Image, render_top: Image.Image):
        front = self._resize(front)
        top = self._resize(top)
        render_front = self._resize(render_front)
        render_top = self._resize(render_top)

        if random.random() < 0.5:
            front = TF.hflip(front)
            top = TF.hflip(top)
            render_front = TF.hflip(render_front)
            render_top = TF.hflip(render_top)

        angle, translate, scale = self._geometric_params()
        front = TF.affine(front, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
        top = TF.affine(top, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
        render_front = TF.affine(render_front, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
        render_top = TF.affine(render_top, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)

        # 원본 쪽에는 조명 변화 대비용 photometric augment를 조금 더 줍니다.
        brightness = 1.0 + random.uniform(-0.16, 0.16)
        contrast = 1.0 + random.uniform(-0.16, 0.16)
        saturation = 1.0 + random.uniform(-0.16, 0.16)
        hue = random.uniform(-0.03, 0.03)
        front = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(front, brightness), contrast), saturation), hue)
        top = TF.adjust_hue(TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(top, brightness), contrast), saturation), hue)

        if random.random() < 0.12:
            front = front.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
            top = top.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        if random.random() < 0.10:
            front = ImageOps.autocontrast(front)
            top = ImageOps.autocontrast(top)

        # synthetic rendered view는 너무 많이 흔들지 않고 약하게만 만집니다.
        if random.random() < 0.06:
            render_front = render_front.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.05, 0.45)))
            render_top = render_top.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.05, 0.45)))

        front_t = TF.normalize(TF.to_tensor(front), IMAGENET_MEAN, IMAGENET_STD)
        top_t = TF.normalize(TF.to_tensor(top), IMAGENET_MEAN, IMAGENET_STD)
        render_front_t = TF.normalize(TF.to_tensor(render_front), IMAGENET_MEAN, IMAGENET_STD)
        render_top_t = TF.normalize(TF.to_tensor(render_top), IMAGENET_MEAN, IMAGENET_STD)
        return front_t, top_t, render_front_t, render_top_t


class RenderFusionEvalTransform:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)

    def __call__(self, front: Image.Image, top: Image.Image, render_front: Image.Image, render_top: Image.Image):
        front = TF.resize(front, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        top = TF.resize(top, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        render_front = TF.resize(render_front, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        render_top = TF.resize(render_top, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)

        front_t = TF.normalize(TF.to_tensor(front), IMAGENET_MEAN, IMAGENET_STD)
        top_t = TF.normalize(TF.to_tensor(top), IMAGENET_MEAN, IMAGENET_STD)
        render_front_t = TF.normalize(TF.to_tensor(render_front), IMAGENET_MEAN, IMAGENET_STD)
        render_top_t = TF.normalize(TF.to_tensor(render_top), IMAGENET_MEAN, IMAGENET_STD)
        return front_t, top_t, render_front_t, render_top_t


# ============================================================
# Dataset
# ============================================================

class RenderFusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, is_test: bool = False, grid_xy: int = 16, grid_z: int = 20, render_size: int = 384, cache_render: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.grid_xy = int(grid_xy)
        self.grid_z = int(grid_z)
        self.render_size = int(render_size)
        self.cache_render = cache_render
        self.render_cache: Dict[str, Tuple[Image.Image, Image.Image]] = {}
        self.mask_cache: Dict[str, Tuple[Image.Image, Image.Image]] = {}

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _get_masks(self, sample_id: str, front: Image.Image, top: Image.Image):
        if sample_id in self.mask_cache:
            fm, tm = self.mask_cache[sample_id]
            return fm.copy(), tm.copy()
        fm = extract_object_mask_from_pil(front, view="front")
        tm = extract_object_mask_from_pil(top, view="top")
        self.mask_cache[sample_id] = (fm.copy(), tm.copy())
        return fm, tm

    def _get_renders(self, sample_id: str, front: Image.Image, top: Image.Image, front_mask: Image.Image, top_mask: Image.Image):
        if self.cache_render and sample_id in self.render_cache:
            rf, rt = self.render_cache[sample_id]
            return rf.copy(), rt.copy()

        rf, rt = build_rendered_views(front, top, front_mask, top_mask, grid_xy=self.grid_xy, grid_z=self.grid_z, render_size=self.render_size)
        if self.cache_render:
            self.render_cache[sample_id] = (rf.copy(), rt.copy())
        return rf, rt

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sample_id = str(row["id"])

        front = self._load_rgb(row["front_path"])
        top = self._load_rgb(row["top_path"])
        fm, tm = self._get_masks(sample_id, front, top)
        rf, rt = self._get_renders(sample_id, front, top, fm, tm)

        if self.transform is None:
            raise RuntimeError("transform is required")
        front_t, top_t, render_front_t, render_top_t = self.transform(front, top, rf, rt)

        if self.is_test:
            return front_t, top_t, render_front_t, render_top_t, sample_id

        label = torch.tensor([float(row["label_float"])], dtype=torch.float32)
        return front_t, top_t, render_front_t, render_top_t, label, sample_id


# ============================================================
# Model
# ============================================================

def build_backbone(pretrained: bool = True):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    try:
        base = models.efficientnet_v2_s(weights=weights)
    except Exception as e:
        print(f"[Warning] pretrained load 실패 -> random init 사용: {e}")
        base = models.efficientnet_v2_s(weights=None)
    feat_dim = base.classifier[1].in_features
    return {"features": base.features, "pool": base.avgpool, "feat_dim": feat_dim}


class PairProjector(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 384, dropout: float = 0.30):
        super().__init__()
        self.view_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_proj = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.view_proj(a)
        b = self.view_proj(b)
        fused = torch.cat([a, b, torch.abs(a - b), a * b], dim=1)
        pair = self.pair_proj(fused)
        return pair, fused


class RenderConsistencyNet(nn.Module):
    """
    원본 2-view 와 rendered 2-view 를 각각 pair embedding으로 만들고,
    마지막에 두 pair를 다시 fuse합니다.
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.30):
        super().__init__()
        spec = build_backbone(pretrained=pretrained)
        feat_dim = int(spec["feat_dim"])
        self.features = spec["features"]
        self.pool = spec["pool"]

        self.orig_pair = PairProjector(feat_dim=feat_dim, hidden=384, dropout=dropout)
        self.render_pair = PairProjector(feat_dim=feat_dim, hidden=384, dropout=dropout)

        self.orig_head = nn.Linear(384, 1)
        self.render_head = nn.Linear(384, 1)
        self.final_head = nn.Sequential(
            nn.Linear(384 * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode_all(self, front: torch.Tensor, top: torch.Tensor, render_front: torch.Tensor, render_top: torch.Tensor):
        x = torch.cat([front, top, render_front, render_top], dim=0)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        front_feat, top_feat, render_front_feat, render_top_feat = x.chunk(4, dim=0)
        return front_feat, top_feat, render_front_feat, render_top_feat

    def forward(self, front: torch.Tensor, top: torch.Tensor, render_front: torch.Tensor, render_top: torch.Tensor):
        ff, tf, rff, rtf = self.encode_all(front, top, render_front, render_top)

        orig_pair, _ = self.orig_pair(ff, tf)
        render_pair, _ = self.render_pair(rff, rtf)

        orig_logit = self.orig_head(orig_pair)
        render_logit = self.render_head(render_pair)
        final_fused = torch.cat([orig_pair, render_pair, torch.abs(orig_pair - render_pair), orig_pair * render_pair], dim=1)
        final_logit = self.final_head(final_fused)

        return {
            "final_logit": final_logit,
            "orig_logit": orig_logit,
            "render_logit": render_logit,
        }


# ============================================================
# Config and loaders
# ============================================================

@dataclass
class Config:
    run_name: str = "render_consistency_fusion"
    mode: str = "holdout"
    data_root: str = "./open"
    save_dir: str = "./runs_render_consistency"

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
    render_size: int = 384
    render_branch_weight: float = 0.35
    consistency_weight: float = 0.05
    export_render_samples: int = 0

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
    ds = RenderFusionDataset(
        df=df,
        transform=transform,
        is_test=is_test,
        grid_xy=cfg.grid_xy,
        grid_z=cfg.grid_z,
        render_size=cfg.render_size,
        cache_render=True,
    )
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


def build_optimizer_and_scheduler(model: RenderConsistencyNet, cfg: Config, steps_per_epoch: int):
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": cfg.backbone_lr},
            {"params": model.orig_pair.parameters(), "lr": cfg.learning_rate},
            {"params": model.render_pair.parameters(), "lr": cfg.learning_rate},
            {"params": model.orig_head.parameters(), "lr": cfg.learning_rate},
            {"params": model.render_head.parameters(), "lr": cfg.learning_rate},
            {"params": model.final_head.parameters(), "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.backbone_lr, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate],
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

    for front, top, render_front, render_top, y, _ in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        render_front = render_front.to(device, non_blocking=True)
        render_top = render_top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            out = model(front, top, render_front, render_top)
            final_loss = classification_loss(out["final_logit"], y, smoothing=cfg.label_smoothing)
            orig_loss = classification_loss(out["orig_logit"], y, smoothing=cfg.label_smoothing)
            render_loss = classification_loss(out["render_logit"], y, smoothing=cfg.label_smoothing)

            # 원본 branch와 rendered branch가 완전히 다른 판단을 하지 않도록 느슨한 consistency를 줍니다.
            consistency = F.mse_loss(torch.sigmoid(out["orig_logit"]), torch.sigmoid(out["render_logit"]))
            loss = final_loss + cfg.render_branch_weight * orig_loss + cfg.render_branch_weight * render_loss + cfg.consistency_weight * consistency

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

    for front, top, render_front, render_top, y, sample_ids in loader:
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        render_front = render_front.to(device, non_blocking=True)
        render_top = render_top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast_context(device.type, use_amp):
            pred = model(front, top, render_front, render_top)["final_logit"]
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

    for front, top, render_front, render_top, sample_ids in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        render_front = render_front.to(device, non_blocking=True)
        render_top = render_top.to(device, non_blocking=True)

        with autocast_context(device.type, use_amp):
            logits = model(front, top, render_front, render_top)["final_logit"]
            if tta_hflip:
                logits_flip = model(
                    torch.flip(front, dims=[3]),
                    torch.flip(top, dims=[3]),
                    torch.flip(render_front, dims=[3]),
                    torch.flip(render_top, dims=[3]),
                )["final_logit"]
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

def export_render_samples(df: pd.DataFrame, run_dir: str, split_name: str, grid_xy: int, grid_z: int, render_size: int, max_samples: int = 10) -> None:
    if max_samples <= 0:
        return
    debug_dir = os.path.join(run_dir, "debug_renders", split_name)
    ensure_dir(debug_dir)

    for _, row in df.head(max_samples).iterrows():
        front = Image.open(row["front_path"]).convert("RGB")
        top = Image.open(row["top_path"]).convert("RGB")
        fm = extract_object_mask_from_pil(front, view="front")
        tm = extract_object_mask_from_pil(top, view="top")
        rf, rt = build_rendered_views(front, top, fm, tm, grid_xy=grid_xy, grid_z=grid_z, render_size=render_size)
        panel = make_render_debug_panel(front.resize((render_size, render_size)), top.resize((render_size, render_size)), rf, rt)
        panel.save(os.path.join(debug_dir, f"{row['id']}_render_panel.png"))


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

    train_tf = RenderFusionTrainTransform(cfg.img_size)
    eval_tf = RenderFusionEvalTransform(cfg.img_size)

    sampler = build_train_sampler(train_df, cfg)
    train_loader = build_loader(train_df, train_tf, cfg, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    valid_loader = build_loader(valid_df, eval_tf, cfg, device=device, is_test=False, shuffle=False, sampler=None)
    test_loader = build_loader(test_df, eval_tf, cfg, device=device, is_test=True, shuffle=False, sampler=None)

    model = RenderConsistencyNet(pretrained=True, dropout=cfg.dropout).to(device)
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

    best_model = RenderConsistencyNet(pretrained=False, dropout=cfg.dropout).to(device)
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
    submission_path = os.path.join(run_dir, "submission_holdout.csv")
    submission.to_csv(submission_path, index=False)
    summary = {
        "run_name": cfg.run_name,
        "mode": "holdout",
        "submission": submission_path,
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
    fold_metrics_df.to_csv(os.path.join(run_dir, "fold_metrics.csv"), index=False)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True).sort_values(["source", "id"]).reset_index(drop=True)
    oof_df.to_csv(os.path.join(run_dir, "oof_predictions.csv"), index=False)

    oof_logloss_cal = dacon_logloss(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    oof_accuracy = binary_accuracy(oof_df["label_float"].values, oof_df["unstable_prob"].values)

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
    sub_equal.to_csv(os.path.join(run_dir, "submission_cv_equal.csv"), index=False)
    sub_weighted.to_csv(os.path.join(run_dir, "submission_cv_weighted.csv"), index=False)

    summary = {
        "run_name": cfg.run_name,
        "mode": "cv",
        "oof_logloss_cal": float(oof_logloss_cal),
        "oof_accuracy": float(oof_accuracy),
        "fold_mean_valid_logloss_cal": float(fold_metrics_df["valid_logloss_cal"].mean()),
        "fold_mean_temperature": float(fold_metrics_df["temperature"].mean()),
        "submission_equal": os.path.join(run_dir, "submission_cv_equal.csv"),
        "submission_weighted": os.path.join(run_dir, "submission_cv_weighted.csv"),
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
    parser = argparse.ArgumentParser(description="Render-consistency fusion trainer")
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_render_consistency")
    parser.add_argument("--run_name", type=str, default="render_consistency_fusion")

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
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup_pct", type=float, default=0.10)

    parser.add_argument("--grid_xy", type=int, default=16)
    parser.add_argument("--grid_z", type=int, default=20)
    parser.add_argument("--render_size", type=int, default=384)
    parser.add_argument("--render_branch_weight", type=float, default=0.35)
    parser.add_argument("--consistency_weight", type=float, default=0.05)
    parser.add_argument("--export_render_samples", type=int, default=0)

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
        render_size=ns.render_size,
        render_branch_weight=ns.render_branch_weight,
        consistency_weight=ns.consistency_weight,
        export_render_samples=ns.export_render_samples,
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

    export_render_samples(train_df, run_dir, "train", cfg.grid_xy, cfg.grid_z, cfg.render_size, max_samples=cfg.export_render_samples)
    export_render_samples(dev_df, run_dir, "dev", cfg.grid_xy, cfg.grid_z, cfg.render_size, max_samples=cfg.export_render_samples)
    export_render_samples(test_df, run_dir, "test", cfg.grid_xy, cfg.grid_z, cfg.render_size, max_samples=cfg.export_render_samples)

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
