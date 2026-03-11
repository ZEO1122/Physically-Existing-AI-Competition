import argparse
import contextlib
import copy
import io
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
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


# ============================================================
# Competition-aware overnight suite
# - respects relative paths
# - uses train/dev/test split from DACON summary
# - supports holdout tuning, CV, calibration, weighted ensemble,
#   domain-aware sampling, stronger aug, multi-backbone, multi-seed,
#   and optional video-motion auxiliary experiment.
# ============================================================


# -------------------------
# Global defaults
# -------------------------
DEFAULTS = {
    "DATA_ROOT": "./open",
    "SAVE_DIR": "./runs_dacon_236686_overnight",
    "TIME_BUDGET_HOURS": 6.5,
    "MIN_REMAINING_MINUTES_TO_START": 25.0,
    "SEED": 42,
    "NUM_WORKERS": 4,
    "AMP": True,
    "PIN_MEMORY": True,
    "NFOLDS": 5,
    "TTA_HFLIP": True,
    "CHECK_PATHS": True,
    "CHECK_TRAIN_VIDEO": False,
    "RUN_HOLDOUT_STAGE": True,
    "RUN_CV_STAGE": True,
    "RUN_FULL_STAGE": True,
    "RUN_VIDEO_AUX_STAGE": True,
    "VIDEO_CACHE_CSV": "video_motion_cache.csv",
    "HOLDOUT_TOP_K_FOR_CV": 2,
    "FULL_TOP_K": 2,
    "FINAL_SEEDS": [42, 3407],
    "FULL_BLEND_WEIGHT_MODE": "cv_calibrated_dev",  # or equal
    "SAVE_HARD_EXAMPLES_TOPK": 50,
    "DOMAIN_DEV_WEIGHT": 3.0,
    "ENABLE_CLASS_BALANCE": False,
    "DEFAULT_BATCH_SIZE": 16,
    "DEFAULT_IMG_SIZE": 384,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_FLOAT = {"stable": 0.0, "unstable": 1.0}
FLOAT_TO_LABEL = {0: "stable", 1: "unstable"}


# -------------------------
# Small helpers
# -------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
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


class TimeBudget:
    def __init__(self, hours: float, min_remaining_minutes_to_start: float = 25.0):
        self.start = time.time()
        self.hours = float(hours)
        self.min_remaining_minutes_to_start = float(min_remaining_minutes_to_start)

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start) / 3600.0

    @property
    def remaining_hours(self) -> float:
        return max(self.hours - self.elapsed_hours, 0.0)

    def can_start_new_run(self) -> bool:
        return self.remaining_hours * 60.0 >= self.min_remaining_minutes_to_start

    def state(self) -> Dict[str, float]:
        return {
            "elapsed_hours": self.elapsed_hours,
            "remaining_hours": self.remaining_hours,
            "budget_hours": self.hours,
        }


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
    loss = -np.sum(true * np.log(pred), axis=1)
    return float(np.mean(loss))


def binary_accuracy(y_true: np.ndarray, unstable_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = (np.asarray(unstable_prob, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
    return float((pred == y_true).mean())


def per_sample_logloss(y_true: np.ndarray, unstable_prob: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = clip_probs(unstable_prob, eps=eps).reshape(-1)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def safe_torch_save(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


# -------------------------
# Data loading / summary-aware paths
# -------------------------

def load_split_df(root_dir: str, split: str) -> pd.DataFrame:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")

    csv_name = "sample_submission.csv" if split == "test" else f"{split}.csv"
    df = pd.read_csv(os.path.join(root_dir, csv_name), dtype={"id": str})
    df["id"] = df["id"].astype(str)
    df["source"] = split

    split_dir = os.path.join(root_dir, split)
    df["folder"] = df["id"].map(lambda x: os.path.join(split_dir, x))
    df["front_path"] = df["folder"].map(lambda x: os.path.join(x, "front.png"))
    df["top_path"] = df["folder"].map(lambda x: os.path.join(x, "top.png"))
    df["video_path"] = None
    if split == "train":
        df["video_path"] = df["folder"].map(lambda x: os.path.join(x, "simulation.mp4"))

    if "label" in df.columns:
        df["label_float"] = df["label"].map(LABEL_TO_FLOAT).astype(np.float64)
    else:
        df["label_float"] = np.nan

    df["motion_score_raw"] = np.nan
    df["motion_score_norm"] = np.nan
    df["has_motion_target"] = False
    return df


def verify_paths(df: pd.DataFrame, check_video: bool = False, max_show: int = 10) -> None:
    missing = []
    for _, row in df.iterrows():
        if not os.path.exists(row["front_path"]):
            missing.append(row["front_path"])
        if not os.path.exists(row["top_path"]):
            missing.append(row["top_path"])
        if check_video and row["source"] == "train":
            vp = row.get("video_path", None)
            if isinstance(vp, str) and not os.path.exists(vp):
                missing.append(vp)
    if missing:
        preview = "\n".join(missing[:max_show])
        raise FileNotFoundError(f"총 {len(missing)}개 경로가 없습니다. 예시:\n{preview}")


def choose_stratify_target(df: pd.DataFrame, n_splits: int) -> pd.Series:
    joint = df["source"].astype(str) + "__" + df["label"].astype(str)
    counts = joint.value_counts()
    if len(counts) > 0 and counts.min() >= n_splits:
        return joint
    return df["label"].astype(str)


# -------------------------
# Optional video motion cache
# -------------------------

def try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None


def compute_video_motion_score(video_path: str, frame_count: int = 16, resize_hw: Tuple[int, int] = (64, 64)) -> Optional[float]:
    cv2 = try_import_cv2()
    if cv2 is None:
        return None
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1:
        cap.release()
        return None

    use_n = min(frame_count, total_frames)
    sample_idx = np.linspace(0, total_frames - 1, num=use_n).astype(int).tolist()
    frames = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize_hw, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()

    if len(frames) < 2:
        return None

    diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
    first_last = float(np.mean(np.abs(frames[-1] - frames[0])))
    temporal_std = float(np.std(np.stack(frames, axis=0), axis=0).mean())
    score = 0.5 * np.mean(diffs) + 0.3 * first_last + 0.2 * temporal_std
    return float(score)


def attach_video_motion_cache(train_df: pd.DataFrame, root_dir: str, save_dir: str, cache_csv_name: str) -> pd.DataFrame:
    cache_path = os.path.join(save_dir, cache_csv_name)
    cache_df = None
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path, dtype={"id": str})
    else:
        cv2 = try_import_cv2()
        if cv2 is None:
            print("[VideoAux] cv2를 찾지 못해 video motion cache 생성을 건너뜁니다.")
            return train_df

        print("[VideoAux] train/simulation.mp4 기반 motion score cache를 생성합니다...")
        rows = []
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="video-cache"):
            motion_score = compute_video_motion_score(row["video_path"])
            rows.append({
                "id": str(row["id"]),
                "motion_score_raw": np.nan if motion_score is None else float(motion_score),
            })
        cache_df = pd.DataFrame(rows)
        valid = cache_df["motion_score_raw"].dropna().astype(np.float64)
        if len(valid) > 0:
            mean = float(valid.mean())
            std = float(valid.std(ddof=0)) if float(valid.std(ddof=0)) > 1e-12 else 1.0
        else:
            mean, std = 0.0, 1.0
        cache_df["motion_score_norm"] = (cache_df["motion_score_raw"] - mean) / std
        cache_df["has_motion_target"] = cache_df["motion_score_raw"].notna()
        cache_df.to_csv(cache_path, index=False)
        save_json(os.path.join(save_dir, "video_motion_stats.json"), {"mean": mean, "std": std})
        print(f"[VideoAux] cache saved: {cache_path}")

    if cache_df is None or len(cache_df) == 0:
        return train_df

    merged = train_df.merge(
        cache_df[["id", "motion_score_raw", "motion_score_norm", "has_motion_target"]],
        on="id",
        how="left",
        suffixes=("", "_cache"),
    )

    for col in ["motion_score_raw", "motion_score_norm", "has_motion_target"]:
        cache_col = f"{col}_cache"
        if cache_col in merged.columns:
            merged[col] = merged[cache_col].where(merged[cache_col].notna(), merged[col])
            merged.drop(columns=[cache_col], inplace=True)

    merged["has_motion_target"] = merged["has_motion_target"].fillna(False).astype(bool)
    return merged


# -------------------------
# Augmentations
# -------------------------
class PairTrainTransform:
    def __init__(self, img_size: int, aug_profile: str = "light"):
        self.img_size = int(img_size)
        self.aug_profile = str(aug_profile)

    def _resize(self, img: Image.Image) -> Image.Image:
        return TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)

    def _maybe_blur(self, img: Image.Image, p: float, radius_min: float, radius_max: float) -> Image.Image:
        if random.random() < p:
            radius = random.uniform(radius_min, radius_max)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def _same_color(self, img: Image.Image, brightness: float, contrast: float, saturation: float, hue: float) -> Image.Image:
        img = TF.adjust_brightness(img, brightness)
        img = TF.adjust_contrast(img, contrast)
        img = TF.adjust_saturation(img, saturation)
        img = TF.adjust_hue(img, hue)
        return img

    def _jpeg_like(self, img: Image.Image, p: float, quality_min: int = 45, quality_max: int = 85) -> Image.Image:
        if random.random() >= p:
            return img
        buf = io.BytesIO()
        quality = int(random.randint(quality_min, quality_max))
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _geometric_params(self):
        if self.aug_profile == "strong":
            angle = random.uniform(-12.0, 12.0)
            max_shift = int(self.img_size * 0.07)
            translate = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
            scale = random.uniform(0.92, 1.08)
        else:
            angle = random.uniform(-8.0, 8.0)
            max_shift = int(self.img_size * 0.05)
            translate = (random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift))
            scale = random.uniform(0.95, 1.05)
        return angle, translate, scale

    def _photometric_params(self):
        if self.aug_profile == "strong":
            brightness = 1.0 + random.uniform(-0.25, 0.25)
            contrast = 1.0 + random.uniform(-0.25, 0.25)
            saturation = 1.0 + random.uniform(-0.25, 0.25)
            hue = random.uniform(-0.05, 0.05)
        else:
            brightness = 1.0 + random.uniform(-0.15, 0.15)
            contrast = 1.0 + random.uniform(-0.15, 0.15)
            saturation = 1.0 + random.uniform(-0.15, 0.15)
            hue = random.uniform(-0.03, 0.03)
        return brightness, contrast, saturation, hue

    def _post_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.aug_profile == "strong":
            if random.random() < 0.25:
                x = torchvision.transforms.RandomErasing(p=1.0, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")(x)
        else:
            if random.random() < 0.15:
                x = torchvision.transforms.RandomErasing(p=1.0, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random")(x)
        return x

    def __call__(self, front: Image.Image, top: Image.Image):
        front = self._resize(front)
        top = self._resize(top)

        if random.random() < 0.5:
            front = TF.hflip(front)
            top = TF.hflip(top)

        angle, translate, scale = self._geometric_params()
        front = TF.affine(front, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)
        top = TF.affine(top, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=0)

        brightness, contrast, saturation, hue = self._photometric_params()
        front = self._same_color(front, brightness, contrast, saturation, hue)
        top = self._same_color(top, brightness, contrast, saturation, hue)

        if self.aug_profile == "strong":
            if random.random() < 0.10:
                front = ImageOps.autocontrast(front)
                top = ImageOps.autocontrast(top)
            if random.random() < 0.05:
                front = ImageOps.grayscale(front).convert("RGB")
                top = ImageOps.grayscale(top).convert("RGB")
            front = self._maybe_blur(front, p=0.20, radius_min=0.2, radius_max=1.2)
            top = self._maybe_blur(top, p=0.20, radius_min=0.2, radius_max=1.2)
            front = self._jpeg_like(front, p=0.15)
            top = self._jpeg_like(top, p=0.15)
        else:
            front = self._maybe_blur(front, p=0.08, radius_min=0.1, radius_max=0.7)
            top = self._maybe_blur(top, p=0.08, radius_min=0.1, radius_max=0.7)

        front = TF.to_tensor(front)
        top = TF.to_tensor(top)
        front = TF.normalize(front, IMAGENET_MEAN, IMAGENET_STD)
        top = TF.normalize(top, IMAGENET_MEAN, IMAGENET_STD)
        front = self._post_tensor(front)
        top = self._post_tensor(top)
        return front, top


class PairEvalTransform:
    def __init__(self, img_size: int):
        self.img_size = int(img_size)

    def __call__(self, front: Image.Image, top: Image.Image):
        front = TF.resize(front, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        top = TF.resize(top, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        front = TF.to_tensor(front)
        top = TF.to_tensor(top)
        front = TF.normalize(front, IMAGENET_MEAN, IMAGENET_STD)
        top = TF.normalize(top, IMAGENET_MEAN, IMAGENET_STD)
        return front, top


# -------------------------
# Dataset
# -------------------------
class MultiViewDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        front = self._load_rgb(row["front_path"])
        top = self._load_rgb(row["top_path"])
        if self.transform is not None:
            front, top = self.transform(front, top)
        sample_id = str(row["id"])

        if self.is_test:
            return front, top, sample_id

        label = torch.tensor([float(row["label_float"] if not pd.isna(row["label_float"]) else LABEL_TO_FLOAT[row["label"]])], dtype=torch.float32)
        motion_target = float(row["motion_score_norm"]) if ("motion_score_norm" in row and pd.notna(row["motion_score_norm"])) else 0.0
        has_motion = bool(row.get("has_motion_target", False))
        return front, top, label, torch.tensor([motion_target], dtype=torch.float32), torch.tensor([1.0 if has_motion else 0.0], dtype=torch.float32), sample_id


# -------------------------
# Backbones / model
# -------------------------

def build_backbone(backbone_name: str, pretrained: bool = True):
    backbone_name = str(backbone_name)
    if backbone_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        try:
            base = models.efficientnet_v2_s(weights=weights)
        except Exception as e:
            print(f"[Warning] efficientnet_v2_s pretrained load 실패 -> random init 사용: {e}")
            base = models.efficientnet_v2_s(weights=None)
        feat_dim = base.classifier[1].in_features
        return {
            "name": backbone_name,
            "feat_dim": feat_dim,
            "features": base.features,
            "pool": base.avgpool,
            "kind": "cnn_pool",
        }

    if backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            base = models.convnext_tiny(weights=weights)
        except Exception as e:
            print(f"[Warning] convnext_tiny pretrained load 실패 -> random init 사용: {e}")
            base = models.convnext_tiny(weights=None)
        feat_dim = base.classifier[2].in_features
        return {
            "name": backbone_name,
            "feat_dim": feat_dim,
            "features": base.features,
            "pool": nn.AdaptiveAvgPool2d((1, 1)),
            "kind": "cnn_pool",
        }

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            base = models.resnet50(weights=weights)
        except Exception as e:
            print(f"[Warning] resnet50 pretrained load 실패 -> random init 사용: {e}")
            base = models.resnet50(weights=None)
        feat_dim = base.fc.in_features
        features = nn.Sequential(*list(base.children())[:-2])
        return {
            "name": backbone_name,
            "feat_dim": feat_dim,
            "features": features,
            "pool": nn.AdaptiveAvgPool2d((1, 1)),
            "kind": "cnn_pool",
        }

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class MultiViewFusionNet(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_v2_s", pretrained: bool = True, dropout: float = 0.3, motion_aux: bool = False):
        super().__init__()
        spec = build_backbone(backbone_name, pretrained=pretrained)
        self.backbone_name = backbone_name
        self.features = spec["features"]
        self.pool = spec["pool"]
        feat_dim = int(spec["feat_dim"])
        hidden = 512 if feat_dim >= 512 else 256

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
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.motion_aux_enabled = bool(motion_aux)
        if self.motion_aux_enabled:
            self.motion_head = nn.Sequential(
                nn.Linear(hidden * 4, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.motion_head = None

    def encode_views(self, front: torch.Tensor, top: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([front, top], dim=0)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        front_feat, top_feat = x.chunk(2, dim=0)
        return front_feat, top_feat

    def fuse(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        front_feat, top_feat = self.encode_views(front, top)
        front_feat = self.front_proj(front_feat)
        top_feat = self.top_proj(top_feat)
        fused = torch.cat([front_feat, top_feat, torch.abs(front_feat - top_feat), front_feat * top_feat], dim=1)
        return fused

    def forward(self, front: torch.Tensor, top: torch.Tensor):
        fused = self.fuse(front, top)
        logits = self.classifier(fused)
        motion_pred = self.motion_head(fused) if self.motion_aux_enabled else None
        return logits, motion_pred


# -------------------------
# Experiment config
# -------------------------
@dataclass
class ExperimentConfig:
    name: str
    protocol: str  # holdout_dev, cv_train_dev, full_train_dev
    backbone: str = "efficientnet_v2_s"
    pretrained: bool = True
    img_size: int = 384
    batch_size: int = 16
    epochs: int = 12
    nfolds: int = 5
    seed: int = 42
    dropout: float = 0.30
    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    mixup_alpha: float = 0.20
    mixup_prob: float = 0.40
    aug_profile: str = "light"
    tta_hflip: bool = True
    dev_oversample_factor: float = 1.0
    class_balance: bool = False
    patience: int = 4
    warmup_pct: float = 0.10
    motion_aux: bool = False
    motion_aux_weight: float = 0.12
    notes: str = ""
    selected_from: str = "manual"
    freeze_backbone_epochs: int = 0

    def short(self) -> Dict:
        d = asdict(self)
        return d


# -------------------------
# Sampler / loaders
# -------------------------

def build_train_sampler(df: pd.DataFrame, exp: ExperimentConfig) -> Optional[WeightedRandomSampler]:
    weights = np.ones(len(df), dtype=np.float64)

    if exp.dev_oversample_factor != 1.0 and "source" in df.columns:
        weights *= np.where(df["source"].astype(str).values == "dev", float(exp.dev_oversample_factor), 1.0)

    if exp.class_balance and "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        class_weight = {k: len(df) / max(v, 1) for k, v in counts.items()}
        weights *= df["label"].map(class_weight).astype(np.float64).values

    if np.allclose(weights, np.ones_like(weights)):
        return None

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(df), replacement=True)


def build_loader(df: pd.DataFrame, transform, exp: ExperimentConfig, device: torch.device, is_test: bool, shuffle: bool, sampler=None) -> DataLoader:
    ds = MultiViewDataset(df, transform=transform, is_test=is_test)
    generator = torch.Generator()
    generator.manual_seed(exp.seed)
    return DataLoader(
        ds,
        batch_size=exp.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=args_global.num_workers,
        pin_memory=(args_global.pin_memory and device.type == "cuda"),
        persistent_workers=(args_global.num_workers > 0),
        worker_init_fn=seed_worker_factory(exp.seed),
        generator=generator if not is_test else None,
        drop_last=False,
    )


# -------------------------
# Loss / calibration
# -------------------------

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
    fine = np.linspace(left, right, 81)
    for t in fine:
        p = sigmoid_np(logits / t)
        ll = dacon_logloss(labels, p)
        if ll < best_ll:
            best_t = float(t)
            best_ll = float(ll)

    return best_t, raw_ll, best_ll


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(max(temperature, 1e-6))
    return sigmoid_np(np.asarray(logits, dtype=np.float64) / temperature)


# -------------------------
# Train / validate / infer
# -------------------------

def maybe_freeze_backbone(model: MultiViewFusionNet, freeze: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = (not freeze)


class EpochOutput:
    def __init__(self):
        self.train_loss = math.nan
        self.valid_bce = math.nan
        self.valid_logloss = math.nan
        self.valid_acc = math.nan
        self.valid_logloss_cal = math.nan
        self.temperature = 1.0
        self.valid_logits = None
        self.valid_labels = None
        self.valid_ids = None
        self.valid_sources = None
        self.valid_probs_raw = None
        self.valid_probs_cal = None


def train_one_epoch(model, loader, optimizer, scheduler, scaler, exp: ExperimentConfig, device: torch.device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for front, top, y, motion_target, has_motion, _ in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        motion_target = motion_target.to(device, non_blocking=True)
        has_motion = has_motion.to(device, non_blocking=True)
        batch_size = front.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            if exp.mixup_alpha > 0 and random.random() < exp.mixup_prob:
                lam = np.random.beta(exp.mixup_alpha, exp.mixup_alpha)
                idx = torch.randperm(batch_size, device=device)
                mixed_front = lam * front + (1.0 - lam) * front[idx]
                mixed_top = lam * top + (1.0 - lam) * top[idx]
                y_a, y_b = y, y[idx]
                logits, motion_pred = model(mixed_front, mixed_top)
                loss = lam * classification_loss(logits, y_a, exp.label_smoothing) + (1.0 - lam) * classification_loss(logits, y_b, exp.label_smoothing)
                if exp.motion_aux and motion_pred is not None:
                    idx_has_motion = has_motion[idx]
                    mask = (has_motion > 0.5) & (idx_has_motion > 0.5)
                    if mask.any():
                        mixed_motion = lam * motion_target + (1.0 - lam) * motion_target[idx]
                        aux_loss = F.mse_loss(motion_pred[mask], mixed_motion[mask])
                        loss = loss + exp.motion_aux_weight * aux_loss
            else:
                logits, motion_pred = model(front, top)
                loss = classification_loss(logits, y, exp.label_smoothing)
                if exp.motion_aux and motion_pred is not None and has_motion.any():
                    mask = has_motion > 0.5
                    if mask.any():
                        aux_loss = F.mse_loss(motion_pred[mask], motion_target[mask])
                        loss = loss + exp.motion_aux_weight * aux_loss

        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, exp: ExperimentConfig, device: torch.device, use_amp: bool) -> EpochOutput:
    model.eval()
    out = EpochOutput()
    total_bce = 0.0
    total_count = 0

    logits_all = []
    labels_all = []
    ids_all = []
    sources_all = []

    for front, top, y, _, _, sample_ids in loader:
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = front.size(0)

        with autocast_context(device.type, use_amp):
            logits, _ = model(front, top)
            bce = F.binary_cross_entropy_with_logits(logits, y)

        total_bce += float(bce.item()) * batch_size
        total_count += batch_size
        logits_all.extend(logits.float().cpu().numpy().reshape(-1).tolist())
        labels_all.extend(y.float().cpu().numpy().reshape(-1).tolist())
        ids_all.extend(list(sample_ids))

    out.valid_bce = total_bce / max(total_count, 1)
    out.valid_logits = np.asarray(logits_all, dtype=np.float64)
    out.valid_labels = np.asarray(labels_all, dtype=np.float64)
    out.valid_ids = np.asarray(ids_all)
    out.valid_sources = None
    out.valid_probs_raw = sigmoid_np(out.valid_logits)
    out.valid_logloss = dacon_logloss(out.valid_labels, out.valid_probs_raw)
    out.valid_acc = binary_accuracy(out.valid_labels, out.valid_probs_raw)
    temp, raw_ll, cal_ll = fit_temperature_grid(out.valid_logits, out.valid_labels)
    out.temperature = temp
    out.valid_probs_cal = apply_temperature_to_logits(out.valid_logits, temp)
    out.valid_logloss_cal = cal_ll
    return out


@torch.no_grad()
def infer_logits(model, loader, device: torch.device, use_amp: bool, tta_hflip: bool) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    logits_all = []
    ids_all = []
    for front, top, sample_ids in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        with autocast_context(device.type, use_amp):
            logits, _ = model(front, top)
            if tta_hflip:
                logits_flip, _ = model(torch.flip(front, dims=[3]), torch.flip(top, dims=[3]))
                logits = (logits + logits_flip) / 2.0
        logits_all.extend(logits.float().cpu().numpy().reshape(-1).tolist())
        ids_all.extend(list(sample_ids))
    return np.asarray(logits_all, dtype=np.float64), ids_all


# -------------------------
# Metrics / analysis
# -------------------------

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


def save_hard_examples(df: pd.DataFrame, out_path: str, topk: int = 50) -> None:
    save_df = df.copy()
    save_df = save_df.sort_values("sample_logloss", ascending=False).head(topk)
    save_df.to_csv(out_path, index=False)


# -------------------------
# Core run helpers
# -------------------------

def build_optimizer_and_scheduler(model: MultiViewFusionNet, exp: ExperimentConfig, steps_per_epoch: int):
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": exp.backbone_lr},
            {"params": model.front_proj.parameters(), "lr": exp.learning_rate},
            {"params": model.top_proj.parameters(), "lr": exp.learning_rate},
            {"params": model.classifier.parameters(), "lr": exp.learning_rate},
        ] + ([{"params": model.motion_head.parameters(), "lr": exp.learning_rate}] if model.motion_aux_enabled and model.motion_head is not None else []),
        weight_decay=exp.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            exp.backbone_lr,
            exp.learning_rate,
            exp.learning_rate,
            exp.learning_rate,
        ] + ([exp.learning_rate] if model.motion_aux_enabled and model.motion_head is not None else []),
        epochs=max(exp.epochs, 1),
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=float(exp.warmup_pct),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def train_with_validation(
    exp: ExperimentConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
    fold_idx: Optional[int] = None,
) -> Dict:
    seed_everything(exp.seed + (0 if fold_idx is None else fold_idx * 1000))
    train_tf = PairTrainTransform(exp.img_size, aug_profile=exp.aug_profile)
    valid_tf = PairEvalTransform(exp.img_size)

    sampler = build_train_sampler(train_df, exp)
    train_loader = build_loader(train_df, train_tf, exp, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    valid_loader = build_loader(valid_df, valid_tf, exp, device=device, is_test=False, shuffle=False, sampler=None)
    test_loader = build_loader(test_df, valid_tf, exp, device=device, is_test=True, shuffle=False, sampler=None)

    model = MultiViewFusionNet(backbone_name=exp.backbone, pretrained=exp.pretrained, dropout=exp.dropout, motion_aux=exp.motion_aux).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, exp, steps_per_epoch=len(train_loader))
    scaler = build_grad_scaler(device.type, enabled=use_amp)

    best_score = float("inf")
    best_epoch = -1
    best_state = None
    best_epoch_output = None
    epochs_no_improve = 0

    epoch_records = []
    t0 = time.time()
    for epoch in range(1, exp.epochs + 1):
        maybe_freeze_backbone(model, freeze=(epoch <= exp.freeze_backbone_epochs))
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, exp, device, use_amp)
        valid_out = validate_one_epoch(model, valid_loader, exp, device, use_amp)
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

        fold_tag = f"fold {fold_idx}" if fold_idx is not None else "holdout"
        print(
            f"[{exp.name} | {fold_tag}] Epoch {epoch:02d}/{exp.epochs} | "
            f"TrainLoss {train_loss:.5f} | ValidBCE {valid_out.valid_bce:.5f} | "
            f"RawLL {valid_out.valid_logloss:.5f} | CalLL {valid_out.valid_logloss_cal:.5f} | Acc {valid_out.valid_acc:.5f} | T {valid_out.temperature:.3f}"
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
                "exp": exp.short(),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= exp.patience:
                print(f"[{exp.name}] early stopping at epoch {epoch} (patience={exp.patience})")
                break

    train_time = time.time() - t0
    ckpt_path = os.path.join(run_dir, f"best_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.pt")
    safe_torch_save(best_state, ckpt_path)
    pd.DataFrame(epoch_records).to_csv(
        os.path.join(run_dir, f"epochs_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        index=False,
    )

    best_model = MultiViewFusionNet(backbone_name=exp.backbone, pretrained=False, dropout=exp.dropout, motion_aux=exp.motion_aux).to(device)
    best_model.load_state_dict(best_state["model"])

    test_logits, test_ids = infer_logits(best_model, test_loader, device=device, use_amp=use_amp, tta_hflip=exp.tta_hflip)
    test_probs_raw = sigmoid_np(test_logits)
    test_probs_cal = apply_temperature_to_logits(test_logits, best_state["temperature"])

    valid_pred_df = valid_df[["id", "source", "label", "label_float", "front_path", "top_path"]].reset_index(drop=True).copy()
    valid_pred_df["logit"] = best_epoch_output.valid_logits
    valid_pred_df["unstable_prob_raw"] = best_epoch_output.valid_probs_raw
    valid_pred_df["unstable_prob"] = best_epoch_output.valid_probs_cal
    valid_pred_df["stable_prob"] = 1.0 - valid_pred_df["unstable_prob"]
    valid_pred_df["temperature"] = float(best_state["temperature"])
    valid_pred_df["sample_logloss"] = per_sample_logloss(valid_pred_df["label_float"].values, valid_pred_df["unstable_prob"].values)
    valid_pred_df.to_csv(
        os.path.join(run_dir, f"valid_predictions_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        index=False,
    )
    save_hard_examples(
        valid_pred_df,
        os.path.join(run_dir, f"hard_examples_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        topk=args_global.save_hard_examples_topk,
    )

    test_pred_df = pd.DataFrame({
        "id": test_ids,
        "logit": test_logits,
        "unstable_prob_raw": test_probs_raw,
        "unstable_prob": test_probs_cal,
        "stable_prob": 1.0 - test_probs_cal,
        "temperature": float(best_state["temperature"]),
    })
    test_pred_df.to_csv(
        os.path.join(run_dir, f"test_predictions_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        index=False,
    )

    metrics = {
        "exp_name": exp.name,
        "protocol": exp.protocol,
        "fold": None if fold_idx is None else int(fold_idx),
        "backbone": exp.backbone,
        "img_size": exp.img_size,
        "seed": exp.seed,
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_time),
        "temperature": float(best_state["temperature"]),
        "valid_logloss_raw": float(best_state["best_valid_logloss_raw"]),
        "valid_logloss_cal": float(best_state["best_valid_logloss_cal"]),
        "valid_accuracy": float(binary_accuracy(valid_df["label_float"].values, valid_pred_df["unstable_prob"].values)),
        "source_metrics": compute_source_metrics(valid_pred_df, prob_col="unstable_prob"),
        "checkpoint": ckpt_path,
        "pred_valid_csv": os.path.join(run_dir, f"valid_predictions_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        "pred_test_csv": os.path.join(run_dir, f"test_predictions_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
    }
    save_json(
        os.path.join(run_dir, f"result_{exp.name}{'' if fold_idx is None else f'_fold{fold_idx}'}.json"),
        metrics,
    )
    return {
        "metrics": metrics,
        "valid_df": valid_pred_df,
        "test_df": test_pred_df,
        "state": best_state,
    }


def run_holdout_experiment(exp: ExperimentConfig, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str, device: torch.device, use_amp: bool) -> Dict:
    exp_dir = os.path.join(out_dir, "holdout", exp.name)
    ensure_dir(exp_dir)
    print(f"\n===== HOLDOUT START: {exp.name} =====")
    return train_with_validation(exp, train_df, dev_df, test_df, exp_dir, device, use_amp, fold_idx=None)



def run_cv_experiment(exp: ExperimentConfig, full_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str, device: torch.device, use_amp: bool) -> Dict:
    exp_dir = os.path.join(out_dir, "cv", exp.name)
    ensure_dir(exp_dir)
    print(f"\n===== CV START: {exp.name} =====")

    strat_target = choose_stratify_target(full_df, exp.nfolds)
    skf = StratifiedKFold(n_splits=exp.nfolds, shuffle=True, random_state=exp.seed)

    oof_frames = []
    fold_metrics = []
    test_pred_frames = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(full_df, strat_target)):
        tr_df = full_df.iloc[tr_idx].reset_index(drop=True)
        va_df = full_df.iloc[va_idx].reset_index(drop=True)
        fold_exp = copy.deepcopy(exp)
        fold_exp.seed = exp.seed + fold_idx
        result = train_with_validation(fold_exp, tr_df, va_df, test_df, exp_dir, device, use_amp, fold_idx=fold_idx)
        valid_df = result["valid_df"].copy()
        valid_df["fold"] = fold_idx
        oof_frames.append(valid_df)
        fold_metrics.append(result["metrics"])
        test_fold = result["test_df"].copy()
        test_fold["fold"] = fold_idx
        test_fold["weight_raw"] = 1.0
        test_fold["weight_inv_cal_ll"] = 1.0 / max(result["metrics"]["valid_logloss_cal"], 1e-8)
        test_pred_frames.append(test_fold)

    fold_metrics_df = pd.DataFrame([
        {
            **{k: v for k, v in m.items() if k != "source_metrics"},
            "source_metrics": json.dumps(m["source_metrics"], ensure_ascii=False),
        }
        for m in fold_metrics
    ])
    fold_metrics_df.to_csv(os.path.join(exp_dir, "fold_metrics.csv"), index=False)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    oof_df = oof_df.sort_values(["source", "id"]).reset_index(drop=True)
    oof_df.to_csv(os.path.join(exp_dir, "oof_predictions.csv"), index=False)

    oof_full_ll = dacon_logloss(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    oof_full_acc = binary_accuracy(oof_df["label_float"].values, oof_df["unstable_prob"].values)
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

    sub_equal = blend_from_weights("weight_raw")
    sub_weighted = blend_from_weights("weight_inv_cal_ll")
    sub_equal.to_csv(os.path.join(exp_dir, "submission_cv_equal.csv"), index=False)
    sub_weighted.to_csv(os.path.join(exp_dir, "submission_cv_weighted.csv"), index=False)

    result_summary = {
        "exp_name": exp.name,
        "protocol": exp.protocol,
        "backbone": exp.backbone,
        "seed": exp.seed,
        "oof_logloss_cal": float(oof_full_ll),
        "oof_accuracy": float(oof_full_acc),
        "source_metrics": source_metrics,
        "fold_mean_valid_logloss_cal": float(fold_metrics_df["valid_logloss_cal"].mean()),
        "fold_mean_valid_logloss_raw": float(fold_metrics_df["valid_logloss_raw"].mean()),
        "fold_mean_temperature": float(fold_metrics_df["temperature"].mean()),
        "submission_equal": os.path.join(exp_dir, "submission_cv_equal.csv"),
        "submission_weighted": os.path.join(exp_dir, "submission_cv_weighted.csv"),
        "oof_csv": os.path.join(exp_dir, "oof_predictions.csv"),
        "fold_metrics_csv": os.path.join(exp_dir, "fold_metrics.csv"),
    }
    save_json(os.path.join(exp_dir, "cv_summary.json"), result_summary)
    print(
        f"[CV {exp.name}] OOF LogLoss {oof_full_ll:.6f} | OOF Acc {oof_full_acc:.6f} | "
        f"Dev LogLoss {source_metrics.get('dev', {}).get('logloss', math.nan):.6f}"
    )
    return {
        "summary": result_summary,
        "oof_df": oof_df,
        "test_equal_df": sub_equal,
        "test_weighted_df": sub_weighted,
        "fold_metrics_df": fold_metrics_df,
    }



def train_full_model(exp: ExperimentConfig, full_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str, device: torch.device, use_amp: bool, inherited_temp: float = 1.0, epochs_override: Optional[int] = None) -> Dict:
    exp_dir = os.path.join(out_dir, "full", exp.name)
    ensure_dir(exp_dir)
    print(f"\n===== FULL TRAIN START: {exp.name} =====")

    train_exp = copy.deepcopy(exp)
    if epochs_override is not None:
        train_exp.epochs = int(epochs_override)

    train_tf = PairTrainTransform(train_exp.img_size, aug_profile=train_exp.aug_profile)
    eval_tf = PairEvalTransform(train_exp.img_size)

    sampler = build_train_sampler(full_df, train_exp)
    train_loader = build_loader(full_df, train_tf, train_exp, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    test_loader = build_loader(test_df, eval_tf, train_exp, device=device, is_test=True, shuffle=False, sampler=None)

    model = MultiViewFusionNet(backbone_name=train_exp.backbone, pretrained=train_exp.pretrained, dropout=train_exp.dropout, motion_aux=train_exp.motion_aux).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_exp, steps_per_epoch=len(train_loader))
    scaler = build_grad_scaler(device.type, enabled=use_amp)

    epoch_records = []
    t0 = time.time()
    for epoch in range(1, train_exp.epochs + 1):
        maybe_freeze_backbone(model, freeze=(epoch <= train_exp.freeze_backbone_epochs))
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, train_exp, device, use_amp)
        epoch_records.append({"epoch": epoch, "train_loss": float(train_loss)})
        print(f"[{train_exp.name}] full-train epoch {epoch:02d}/{train_exp.epochs} | train_loss {train_loss:.5f}")

    train_seconds = time.time() - t0
    ckpt_path = os.path.join(exp_dir, f"full_model_{train_exp.name}.pt")
    safe_torch_save({
        "model": model.state_dict(),
        "exp": train_exp.short(),
        "temperature": float(inherited_temp),
        "epochs": int(train_exp.epochs),
    }, ckpt_path)
    pd.DataFrame(epoch_records).to_csv(os.path.join(exp_dir, f"epochs_{train_exp.name}.csv"), index=False)

    test_logits, test_ids = infer_logits(model, test_loader, device=device, use_amp=use_amp, tta_hflip=train_exp.tta_hflip)
    test_probs = apply_temperature_to_logits(test_logits, inherited_temp)
    submission = pd.DataFrame({
        "id": test_ids,
        "unstable_prob": test_probs,
        "stable_prob": 1.0 - test_probs,
    }).sort_values("id").reset_index(drop=True)
    submission_path = os.path.join(exp_dir, f"submission_{train_exp.name}.csv")
    submission.to_csv(submission_path, index=False)
    summary = {
        "exp_name": train_exp.name,
        "checkpoint": ckpt_path,
        "submission": submission_path,
        "epochs": int(train_exp.epochs),
        "temperature": float(inherited_temp),
        "train_seconds": float(train_seconds),
    }
    save_json(os.path.join(exp_dir, f"summary_{train_exp.name}.json"), summary)
    return {"summary": summary, "submission_df": submission}


# -------------------------
# Suite definition
# -------------------------

def make_holdout_suite(base_seed: int, enable_video_aux: bool) -> List[ExperimentConfig]:
    suite = [
        ExperimentConfig(
            name="hold_effv2s_base",
            protocol="holdout_dev",
            backbone="efficientnet_v2_s",
            img_size=384,
            batch_size=16,
            epochs=10,
            seed=base_seed,
            aug_profile="light",
            label_smoothing=0.05,
            mixup_alpha=0.20,
            mixup_prob=0.40,
            notes="baseline holdout",
        ),
        ExperimentConfig(
            name="hold_effv2s_nomix_ls0",
            protocol="holdout_dev",
            backbone="efficientnet_v2_s",
            img_size=384,
            batch_size=16,
            epochs=10,
            seed=base_seed,
            aug_profile="light",
            label_smoothing=0.00,
            mixup_alpha=0.00,
            mixup_prob=0.00,
            notes="check if smoothing/mixup hurt logloss",
        ),
        ExperimentConfig(
            name="hold_effv2s_mixlite",
            protocol="holdout_dev",
            backbone="efficientnet_v2_s",
            img_size=384,
            batch_size=16,
            epochs=10,
            seed=base_seed,
            aug_profile="light",
            label_smoothing=0.03,
            mixup_alpha=0.10,
            mixup_prob=0.25,
            notes="lighter calibration-friendly mix",
        ),
        ExperimentConfig(
            name="hold_effv2s_strongaug",
            protocol="holdout_dev",
            backbone="efficientnet_v2_s",
            img_size=384,
            batch_size=16,
            epochs=10,
            seed=base_seed,
            aug_profile="strong",
            label_smoothing=0.03,
            mixup_alpha=0.10,
            mixup_prob=0.25,
            notes="stronger photometric aug for dev/test domain",
        ),
        ExperimentConfig(
            name="hold_convnext_base",
            protocol="holdout_dev",
            backbone="convnext_tiny",
            img_size=320,
            batch_size=16,
            epochs=10,
            seed=base_seed,
            aug_profile="strong",
            label_smoothing=0.03,
            mixup_alpha=0.10,
            mixup_prob=0.25,
            notes="backbone diversity",
        ),
    ]
    if enable_video_aux:
        suite.append(
            ExperimentConfig(
                name="hold_effv2s_videoaux",
                protocol="holdout_dev",
                backbone="efficientnet_v2_s",
                img_size=384,
                batch_size=16,
                epochs=10,
                seed=base_seed,
                aug_profile="strong",
                label_smoothing=0.03,
                mixup_alpha=0.10,
                mixup_prob=0.25,
                motion_aux=True,
                motion_aux_weight=0.12,
                notes="image student with train video motion auxiliary target",
            )
        )
    return suite



def convert_holdout_to_cv(exp: ExperimentConfig) -> ExperimentConfig:
    new_exp = copy.deepcopy(exp)
    new_exp.protocol = "cv_train_dev"
    new_exp.name = exp.name.replace("hold_", "cv_")
    new_exp.epochs = 14
    new_exp.nfolds = args_global.nfolds
    new_exp.patience = 4
    new_exp.dev_oversample_factor = args_global.domain_dev_weight
    new_exp.class_balance = args_global.enable_class_balance
    new_exp.selected_from = "holdout_topk"
    return new_exp



def convert_cv_to_full(exp: ExperimentConfig, seed: int, tag: str) -> ExperimentConfig:
    new_exp = copy.deepcopy(exp)
    new_exp.protocol = "full_train_dev"
    new_exp.seed = int(seed)
    new_exp.name = f"full_{tag}_{exp.backbone}_seed{seed}"
    new_exp.selected_from = exp.name
    new_exp.patience = 999
    return new_exp


# -------------------------
# Selection / blending
# -------------------------

def rank_holdout_results(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        dev_ll = m["source_metrics"].get("dev", {}).get("logloss", m["valid_logloss_cal"])
        rows.append({
            "exp_name": m["exp_name"],
            "backbone": m["backbone"],
            "protocol": m["protocol"],
            "valid_logloss_raw": m["valid_logloss_raw"],
            "valid_logloss_cal": m["valid_logloss_cal"],
            "dev_logloss_cal": dev_ll,
            "valid_accuracy": m["valid_accuracy"],
            "temperature": m["temperature"],
            "train_seconds": m["train_seconds"],
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["dev_logloss_cal", "valid_logloss_cal", "valid_accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    return df



def select_top_holdout_configs(holdout_suite: List[ExperimentConfig], ranking_df: pd.DataFrame, top_k: int) -> List[ExperimentConfig]:
    if len(ranking_df) == 0:
        return []
    by_name = {exp.name: exp for exp in holdout_suite}
    picked = []
    used_backbones = set()
    for _, row in ranking_df.iterrows():
        name = row["exp_name"]
        exp = by_name[name]
        if exp.backbone not in used_backbones:
            picked.append(convert_holdout_to_cv(exp))
            used_backbones.add(exp.backbone)
        if len(picked) >= top_k:
            break
    if len(picked) < top_k:
        for _, row in ranking_df.iterrows():
            name = row["exp_name"]
            exp = by_name[name]
            candidate = convert_holdout_to_cv(exp)
            if all(candidate.name != p.name for p in picked):
                picked.append(candidate)
            if len(picked) >= top_k:
                break
    return picked[:top_k]



def rank_cv_results(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        s = r["summary"]
        dev_ll = s["source_metrics"].get("dev", {}).get("logloss", s["oof_logloss_cal"])
        rows.append({
            "exp_name": s["exp_name"],
            "backbone": s["backbone"],
            "oof_logloss_cal": s["oof_logloss_cal"],
            "oof_accuracy": s["oof_accuracy"],
            "dev_logloss_cal": dev_ll,
            "fold_mean_valid_logloss_cal": s["fold_mean_valid_logloss_cal"],
            "fold_mean_temperature": s["fold_mean_temperature"],
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values(["dev_logloss_cal", "oof_logloss_cal", "oof_accuracy"], ascending=[True, True, False]).reset_index(drop=True)
    return df



def weighted_blend_submissions(submission_infos: List[Tuple[pd.DataFrame, float]], out_path: str) -> pd.DataFrame:
    if not submission_infos:
        raise ValueError("No submission infos to blend")
    base_ids = submission_infos[0][0]["id"].astype(str).tolist()
    total_weight = 0.0
    stacked = np.zeros(len(base_ids), dtype=np.float64)
    for sub_df, weight in submission_infos:
        sub_df = sub_df.sort_values("id").reset_index(drop=True)
        if sub_df["id"].astype(str).tolist() != base_ids:
            raise ValueError("Submission id order mismatch during blending")
        weight = float(weight)
        total_weight += weight
        stacked += weight * sub_df["unstable_prob"].values.astype(np.float64)
    stacked /= max(total_weight, 1e-12)
    out_df = pd.DataFrame({
        "id": base_ids,
        "unstable_prob": stacked,
        "stable_prob": 1.0 - stacked,
    })
    out_df.to_csv(out_path, index=False)
    return out_df


# -------------------------
# Main suite
# -------------------------
class GlobalArgs:
    def __init__(self, ns):
        self.data_root = ns.data_root
        self.save_dir = ns.save_dir
        self.time_budget_hours = ns.time_budget_hours
        self.min_remaining_minutes_to_start = ns.min_remaining_minutes_to_start
        self.seed = ns.seed
        self.num_workers = ns.num_workers
        self.nfolds = ns.nfolds
        self.amp = ns.amp
        self.pin_memory = ns.pin_memory
        self.run_holdout_stage = ns.run_holdout_stage
        self.run_cv_stage = ns.run_cv_stage
        self.run_full_stage = ns.run_full_stage
        self.run_video_aux_stage = ns.run_video_aux_stage
        self.check_paths = ns.check_paths
        self.check_train_video = ns.check_train_video
        self.video_cache_csv = ns.video_cache_csv
        self.holdout_top_k_for_cv = ns.holdout_top_k_for_cv
        self.full_top_k = ns.full_top_k
        self.final_seeds = [int(x) for x in ns.final_seeds.split(",") if str(x).strip()]
        self.domain_dev_weight = ns.domain_dev_weight
        self.enable_class_balance = ns.enable_class_balance
        self.tta_hflip = ns.tta_hflip
        self.save_hard_examples_topk = ns.save_hard_examples_topk


args_global: GlobalArgs


def main():
    global args_global
    parser = argparse.ArgumentParser(description="Overnight DACON multiview experiment suite")
    parser.add_argument("--data_root", type=str, default=DEFAULTS["DATA_ROOT"])
    parser.add_argument("--save_dir", type=str, default=DEFAULTS["SAVE_DIR"])
    parser.add_argument("--time_budget_hours", type=float, default=DEFAULTS["TIME_BUDGET_HOURS"])
    parser.add_argument("--min_remaining_minutes_to_start", type=float, default=DEFAULTS["MIN_REMAINING_MINUTES_TO_START"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["SEED"])
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["NUM_WORKERS"])
    parser.add_argument("--nfolds", type=int, default=DEFAULTS["NFOLDS"])
    parser.add_argument("--amp", action="store_true", default=DEFAULTS["AMP"])
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--pin_memory", action="store_true", default=DEFAULTS["PIN_MEMORY"])
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--run_holdout_stage", action="store_true", default=DEFAULTS["RUN_HOLDOUT_STAGE"])
    parser.add_argument("--skip_holdout_stage", action="store_false", dest="run_holdout_stage")
    parser.add_argument("--run_cv_stage", action="store_true", default=DEFAULTS["RUN_CV_STAGE"])
    parser.add_argument("--skip_cv_stage", action="store_false", dest="run_cv_stage")
    parser.add_argument("--run_full_stage", action="store_true", default=DEFAULTS["RUN_FULL_STAGE"])
    parser.add_argument("--skip_full_stage", action="store_false", dest="run_full_stage")
    parser.add_argument("--run_video_aux_stage", action="store_true", default=DEFAULTS["RUN_VIDEO_AUX_STAGE"])
    parser.add_argument("--skip_video_aux_stage", action="store_false", dest="run_video_aux_stage")
    parser.add_argument("--check_paths", action="store_true", default=DEFAULTS["CHECK_PATHS"])
    parser.add_argument("--no_check_paths", action="store_false", dest="check_paths")
    parser.add_argument("--check_train_video", action="store_true", default=DEFAULTS["CHECK_TRAIN_VIDEO"])
    parser.add_argument("--video_cache_csv", type=str, default=DEFAULTS["VIDEO_CACHE_CSV"])
    parser.add_argument("--holdout_top_k_for_cv", type=int, default=DEFAULTS["HOLDOUT_TOP_K_FOR_CV"])
    parser.add_argument("--full_top_k", type=int, default=DEFAULTS["FULL_TOP_K"])
    parser.add_argument("--final_seeds", type=str, default=",".join(map(str, DEFAULTS["FINAL_SEEDS"])))
    parser.add_argument("--domain_dev_weight", type=float, default=DEFAULTS["DOMAIN_DEV_WEIGHT"])
    parser.add_argument("--enable_class_balance", action="store_true", default=DEFAULTS["ENABLE_CLASS_BALANCE"])
    parser.add_argument("--tta_hflip", action="store_true", default=DEFAULTS["TTA_HFLIP"])
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")
    parser.add_argument("--save_hard_examples_topk", type=int, default=DEFAULTS["SAVE_HARD_EXAMPLES_TOPK"])
    ns = parser.parse_args()
    args_global = GlobalArgs(ns)

    seed_everything(args_global.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args_global.amp and device.type == "cuda")

    run_dir = os.path.join(args_global.save_dir, now_str())
    ensure_dir(run_dir)
    budget = TimeBudget(args_global.time_budget_hours, args_global.min_remaining_minutes_to_start)

    train_df = load_split_df(args_global.data_root, "train")
    dev_df = load_split_df(args_global.data_root, "dev")
    test_df = load_split_df(args_global.data_root, "test")

    if args_global.check_paths:
        verify_paths(train_df, check_video=args_global.check_train_video)
        verify_paths(dev_df, check_video=False)
        verify_paths(test_df, check_video=False)

    if args_global.run_video_aux_stage:
        train_df = attach_video_motion_cache(train_df, args_global.data_root, run_dir, args_global.video_cache_csv)

    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    full_df["label_float"] = full_df["label_float"].astype(np.float64)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "use_amp": use_amp,
        "defaults": DEFAULTS,
        "args": vars(ns),
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "train_label_distribution": train_df["label"].value_counts().to_dict(),
        "dev_label_distribution": dev_df["label"].value_counts().to_dict(),
    }
    save_json(os.path.join(run_dir, "run_metadata.json"), metadata)

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)} | Test samples: {len(test_df)}")
    print(f"Full trainable samples (train+dev): {len(full_df)}")
    print(f"Time budget: {args_global.time_budget_hours:.2f}h")

    video_aux_available = bool(train_df.get('has_motion_target', pd.Series([False])).fillna(False).astype(bool).any())
    holdout_suite = make_holdout_suite(args_global.seed, enable_video_aux=video_aux_available)
    holdout_results: List[Dict] = []
    cv_results: List[Dict] = []
    full_results: List[Dict] = []

    if args_global.run_holdout_stage:
        for exp in holdout_suite:
            if not budget.can_start_new_run():
                print("[Budget] holdout stage stopped due to time budget")
                break
            try:
                res = run_holdout_experiment(exp, train_df, dev_df, test_df, run_dir, device, use_amp)
                holdout_results.append(res)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] {exp.name} failed and will be skipped: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

    holdout_ranking = rank_holdout_results(holdout_results)
    holdout_ranking_path = os.path.join(run_dir, "holdout_ranking.csv")
    holdout_ranking.to_csv(holdout_ranking_path, index=False)

    cv_suite: List[ExperimentConfig] = []
    if args_global.run_cv_stage and len(holdout_ranking) > 0:
        cv_suite = select_top_holdout_configs(holdout_suite, holdout_ranking, args_global.holdout_top_k_for_cv)
        # ensure the video aux experiment is included if it exists and was competitive enough
        if video_aux_available and "hold_effv2s_videoaux" in set(holdout_ranking["exp_name"].tolist()):
            best_names = set(holdout_ranking.head(max(args_global.holdout_top_k_for_cv + 1, 3))["exp_name"].tolist())
            if "hold_effv2s_videoaux" in best_names:
                video_cv = convert_holdout_to_cv(next(e for e in holdout_suite if e.name == "hold_effv2s_videoaux"))
                if all(video_cv.name != x.name for x in cv_suite):
                    cv_suite.append(video_cv)

        selected_cv_df = pd.DataFrame([x.short() for x in cv_suite]) if cv_suite else pd.DataFrame()
        selected_cv_df.to_csv(os.path.join(run_dir, "selected_cv_suite.csv"), index=False)

        for exp in cv_suite:
            if not budget.can_start_new_run():
                print("[Budget] cv stage stopped due to time budget")
                break
            try:
                res = run_cv_experiment(exp, full_df, test_df, run_dir, device, use_amp)
                cv_results.append(res)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] {exp.name} failed and will be skipped: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

    cv_ranking = rank_cv_results(cv_results)
    cv_ranking_path = os.path.join(run_dir, "cv_ranking.csv")
    cv_ranking.to_csv(cv_ranking_path, index=False)

    if args_global.run_full_stage and len(cv_ranking) > 0:
        top_cv_names = cv_ranking.head(args_global.full_top_k)["exp_name"].tolist()
        name_to_exp = {exp.name: exp for exp in cv_suite}
        full_suite = []
        for name in top_cv_names:
            parent = name_to_exp[name]
            for seed in args_global.final_seeds:
                full_suite.append(convert_cv_to_full(parent, seed=seed, tag=name.replace("cv_", "")))

        pd.DataFrame([x.short() for x in full_suite]).to_csv(os.path.join(run_dir, "selected_full_suite.csv"), index=False)

        cv_score_lookup = {r["summary"]["exp_name"]: r["summary"] for r in cv_results}
        for exp in full_suite:
            if not budget.can_start_new_run():
                print("[Budget] full stage stopped due to time budget")
                break
            parent_name = exp.selected_from
            parent_summary = cv_score_lookup.get(parent_name, None)
            inherited_temp = float(parent_summary["fold_mean_temperature"]) if parent_summary is not None else 1.0
            if parent_summary is not None:
                # mean best epoch from fold metrics, fallback to parent epochs
                fold_metrics_csv = parent_summary["fold_metrics_csv"]
                if os.path.exists(fold_metrics_csv):
                    fm = pd.read_csv(fold_metrics_csv)
                    if "best_epoch" in fm.columns:
                        epochs_override = int(max(4, round(float(fm["best_epoch"].mean()))))
                    else:
                        epochs_override = None
                else:
                    epochs_override = None
            else:
                epochs_override = None

            try:
                res = train_full_model(exp, full_df, test_df, run_dir, device, use_amp, inherited_temp=inherited_temp, epochs_override=epochs_override)
                full_results.append({
                    "exp": exp,
                    "result": res,
                    "parent_name": parent_name,
                    "parent_score": None if parent_summary is None else float(parent_summary["source_metrics"].get("dev", {}).get("logloss", parent_summary["oof_logloss_cal"])),
                })
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] {exp.name} failed and will be skipped: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

    artifact_summary = {
        "run_dir": run_dir,
        "holdout_ranking_csv": holdout_ranking_path,
        "cv_ranking_csv": cv_ranking_path,
        "budget": budget.state(),
        "holdout_experiments_ran": len(holdout_results),
        "cv_experiments_ran": len(cv_results),
        "full_experiments_ran": len(full_results),
    }

    # Final blends
    final_paths = {}
    if len(cv_results) > 0:
        cv_name_to_result = {r['summary']['exp_name']: r for r in cv_results}
        best_cv_name = str(cv_ranking.iloc[0]['exp_name']) if len(cv_ranking) > 0 else cv_results[0]['summary']['exp_name']
        best_cv = cv_name_to_result[best_cv_name]
        best_cv_sub = best_cv['test_weighted_df'].sort_values('id').reset_index(drop=True)
        best_cv_path = os.path.join(run_dir, "final_submission_best_cv_weighted.csv")
        best_cv_sub.to_csv(best_cv_path, index=False)
        final_paths["best_cv_weighted"] = best_cv_path

    if len(full_results) > 0:
        sub_infos = []
        for item in full_results:
            sub_df = item["result"]["submission_df"].sort_values("id").reset_index(drop=True)
            score = item["parent_score"]
            if score is None or args_global.final_seeds is None:
                weight = 1.0
            else:
                weight = 1.0 / max(float(score), 1e-8)
            sub_infos.append((sub_df, weight))
        final_full_path = os.path.join(run_dir, "final_submission_full_blend.csv")
        weighted_blend_submissions(sub_infos, final_full_path)
        final_paths["full_blend"] = final_full_path

    if len(full_results) > 0 and len(cv_results) > 0:
        cv_name_to_result = {r['summary']['exp_name']: r for r in cv_results}
        best_cv_name = str(cv_ranking.iloc[0]['exp_name']) if len(cv_ranking) > 0 else cv_results[0]['summary']['exp_name']
        cv_best_summary = cv_name_to_result[best_cv_name]
        cv_dev_score = float(cv_best_summary["summary"]["source_metrics"].get("dev", {}).get("logloss", cv_best_summary["summary"]["oof_logloss_cal"]))
        full_sub = pd.read_csv(final_paths["full_blend"], dtype={"id": str})
        cv_sub = pd.read_csv(final_paths["best_cv_weighted"], dtype={"id": str})
        blend_weight_full = 1.0 / max(cv_dev_score, 1e-8)
        blend_weight_cv = 1.0 / max(cv_dev_score, 1e-8)
        super_path = os.path.join(run_dir, "final_submission_super_blend.csv")
        weighted_blend_submissions([(full_sub, blend_weight_full), (cv_sub, blend_weight_cv)], super_path)
        final_paths["super_blend"] = super_path

    artifact_summary["final_paths"] = final_paths
    artifact_summary["budget_final"] = budget.state()
    save_json(os.path.join(run_dir, "artifact_summary.json"), artifact_summary)

    print("\n========== DONE ==========")
    print(f"Run dir: {run_dir}")
    print(f"Budget used: {budget.elapsed_hours:.2f}h / {args_global.time_budget_hours:.2f}h")
    if len(holdout_ranking) > 0:
        print("Holdout best:")
        print(holdout_ranking.head(5).to_string(index=False))
    if len(cv_ranking) > 0:
        print("\nCV best:")
        print(cv_ranking.head(5).to_string(index=False))
    if final_paths:
        print("\nFinal submissions:")
        for k, v in final_paths.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
