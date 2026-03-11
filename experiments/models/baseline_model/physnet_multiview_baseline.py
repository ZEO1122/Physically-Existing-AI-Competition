"""
PhysNet-inspired multi-view baseline for DACON structure stability.

What is adapted from the original PhysNet paper?
- Shared ResNet-34 trunk ending at the 7x7 spatial feature map.
- A PhysNet-style spatial decoder that upsamples coarse features to 56x56.
- A joint training setup with:
    1) binary fall / instability prediction
    2) optional auxiliary future motion-mask prediction

What is different from the original PhysNet?
- The competition input is two views (front / top), so we fuse two ResNet feature maps.
- The competition does not provide segmentation masks, but train split provides
  simulation.mp4. We convert that video into coarse binary motion heatmaps at
  multiple future times and use them as pseudo mask targets.
- Dev/Test do not contain simulation.mp4, so auxiliary supervision is only used
  when a video exists.

This script is intentionally standalone and keeps the same broad workflow as the
user-provided backbone baseline:
1) load train/dev/test CSVs and image paths
2) train a holdout or CV model
3) validate, calibrate with temperature scaling, and save predictions/submissions

Recommended first run:
python physnet_multiview_baseline.py --mode holdout --data_root ./open --run_name physnet_holdout
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_FLOAT = {"stable": 0.0, "unstable": 1.0}


# ============================================================
# Utilities
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)



def save_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
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
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
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



def parse_float_list(raw: str) -> Tuple[float, ...]:
    raw = str(raw).strip()
    if not raw:
        return tuple()
    return tuple(float(x.strip()) for x in raw.split(",") if str(x).strip())


# ============================================================
# Competition-aware data loading
# ============================================================

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
    df["video_path"] = df["folder"].map(lambda x: os.path.join(x, "simulation.mp4"))
    df["has_video"] = df["video_path"].map(os.path.exists)

    if "label" in df.columns:
        df["label_float"] = df["label"].map(LABEL_TO_FLOAT).astype(np.float64)
    else:
        df["label_float"] = np.nan
    return df



def verify_paths(df: pd.DataFrame, check_video_when_available: bool = False, max_show: int = 10) -> None:
    missing: List[str] = []
    for _, row in df.iterrows():
        if not os.path.exists(row["front_path"]):
            missing.append(row["front_path"])
        if not os.path.exists(row["top_path"]):
            missing.append(row["top_path"])
        if check_video_when_available and bool(row.get("has_video", False)) and not os.path.exists(row["video_path"]):
            missing.append(row["video_path"])
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
# Motion pseudo-target extraction from simulation.mp4
# ============================================================

def _video_cache_name(video_path: str, motion_size: int, timepoints: Sequence[float], threshold: float, blur_kernel: int) -> str:
    payload = {
        "video_path": os.path.abspath(video_path),
        "motion_size": int(motion_size),
        "timepoints": [float(x) for x in timepoints],
        "threshold": float(threshold),
        "blur_kernel": int(blur_kernel),
    }
    sig = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{sig}.npy"



def _prep_gray_frame(frame_bgr: np.ndarray, target_size: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)



def extract_motion_targets_from_video(
    video_path: str,
    motion_size: int,
    timepoints: Sequence[float],
    threshold: float = 0.10,
    blur_kernel: int = 5,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 8.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read_frame(frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0:
            frame_idx = 0
        if frame_count > 0:
            frame_idx = min(frame_idx, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    first_frame = read_frame(0)
    if first_frame is None:
        cap.release()
        raise RuntimeError(f"첫 프레임 읽기 실패: {video_path}")

    first_gray = _prep_gray_frame(first_frame, target_size=motion_size)
    outputs: List[np.ndarray] = []
    prev_gray = first_gray.copy()

    k = int(blur_kernel)
    if k % 2 == 0:
        k += 1
    if k < 1:
        k = 1

    for t in timepoints:
        frame_idx = int(round(float(t) * fps))
        frame = read_frame(frame_idx)
        if frame is None:
            gray = prev_gray.copy()
        else:
            gray = _prep_gray_frame(frame, target_size=motion_size)
            prev_gray = gray

        diff = cv2.absdiff(gray, first_gray) / 255.0
        if k > 1:
            diff = cv2.GaussianBlur(diff, (k, k), 0)
        mask = np.clip((diff - float(threshold)) / max(1.0 - float(threshold), 1e-6), 0.0, 1.0)
        outputs.append(mask.astype(np.float32))

    cap.release()
    return np.stack(outputs, axis=0).astype(np.float32)



def load_or_build_motion_targets(
    video_path: str,
    cache_dir: Optional[str],
    motion_size: int,
    timepoints: Sequence[float],
    threshold: float,
    blur_kernel: int,
) -> np.ndarray:
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"simulation.mp4 not found: {video_path}")

    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_path = os.path.join(
            cache_dir,
            _video_cache_name(video_path, motion_size, timepoints, threshold, blur_kernel),
        )
        if os.path.exists(cache_path):
            arr = np.load(cache_path)
            return arr.astype(np.float32)

    arr = extract_motion_targets_from_video(
        video_path=video_path,
        motion_size=motion_size,
        timepoints=timepoints,
        threshold=threshold,
        blur_kernel=blur_kernel,
    )

    if cache_path is not None:
        tmp_path = f"{cache_path}.tmp.{os.getpid()}.{time.time_ns()}.npy"
        np.save(tmp_path, arr.astype(np.float32))
        try:
            os.replace(tmp_path, cache_path)
        except Exception:
            for path in [tmp_path, tmp_path + ".npy"]:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
    return arr.astype(np.float32)


# ============================================================
# Transforms
# ============================================================
class PairTrainTransform:
    """
    Apply the same geometric / photometric transform to front and top images.
    If a motion target is present, apply the same geometric transform to it.
    """

    def __init__(self, img_size: int, motion_size: int, aug_profile: str = "light"):
        self.img_size = int(img_size)
        self.motion_size = int(motion_size)
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
        img.save(buf, format="JPEG", quality=int(random.randint(quality_min, quality_max)))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _geometric_params(self) -> Tuple[float, Tuple[int, int], float]:
        if self.aug_profile == "strong":
            angle = random.uniform(-12.0, 12.0)
            max_shift = int(self.img_size * 0.07)
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
            scale = random.uniform(0.92, 1.08)
        else:
            angle = random.uniform(-8.0, 8.0)
            max_shift = int(self.img_size * 0.05)
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
            scale = random.uniform(0.95, 1.05)
        return angle, translate, scale

    def _photometric_params(self) -> Tuple[float, float, float, float]:
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
            if random.random() < 0.20:
                x = torchvision.transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.12),
                    ratio=(0.3, 3.3),
                    value="random",
                )(x)
        else:
            if random.random() < 0.10:
                x = torchvision.transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.10),
                    ratio=(0.3, 3.3),
                    value="random",
                )(x)
        return x

    def _transform_motion_targets(
        self,
        motion_targets: Optional[np.ndarray],
        do_hflip: bool,
        angle: float,
        translate: Tuple[int, int],
        scale: float,
    ) -> Optional[torch.Tensor]:
        if motion_targets is None:
            return None
        x = torch.as_tensor(motion_targets, dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"motion_targets must be [T,H,W], got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # [T,1,H,W]

        if do_hflip:
            x = torch.flip(x, dims=[3])

        tx = int(round(float(translate[0]) * self.motion_size / max(self.img_size, 1)))
        ty = int(round(float(translate[1]) * self.motion_size / max(self.img_size, 1)))
        out = []
        for item in x:
            item = TF.affine(
                item,
                angle=angle,
                translate=(tx, ty),
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
            out.append(item.squeeze(0))
        return torch.stack(out, dim=0)

    def __call__(self, front: Image.Image, top: Image.Image, motion_targets: Optional[np.ndarray] = None):
        front = self._resize(front)
        top = self._resize(top)

        do_hflip = random.random() < 0.5
        if do_hflip:
            front = TF.hflip(front)
            top = TF.hflip(top)

        angle, translate, scale = self._geometric_params()
        front = TF.affine(
            front,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        top = TF.affine(
            top,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        motion_targets_t = self._transform_motion_targets(motion_targets, do_hflip, angle, translate, scale)

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
        return front, top, motion_targets_t


class PairEvalTransform:
    def __init__(self, img_size: int, motion_size: int):
        self.img_size = int(img_size)
        self.motion_size = int(motion_size)

    def __call__(self, front: Image.Image, top: Image.Image, motion_targets: Optional[np.ndarray] = None):
        front = TF.resize(front, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        top = TF.resize(top, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)
        front = TF.to_tensor(front)
        top = TF.to_tensor(top)
        front = TF.normalize(front, IMAGENET_MEAN, IMAGENET_STD)
        top = TF.normalize(top, IMAGENET_MEAN, IMAGENET_STD)
        motion_targets_t = None
        if motion_targets is not None:
            motion_targets_t = torch.as_tensor(motion_targets, dtype=torch.float32)
        return front, top, motion_targets_t


# ============================================================
# Dataset
# ============================================================
class MultiViewPhysNetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        is_test: bool = False,
        use_motion_aux: bool = False,
        motion_cache_dir: Optional[str] = None,
        motion_size: int = 56,
        motion_timepoints: Sequence[float] = (2.5, 5.0, 7.5, 10.0),
        motion_threshold: float = 0.10,
        motion_blur_kernel: int = 5,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.use_motion_aux = bool(use_motion_aux and (not is_test))
        self.motion_cache_dir = motion_cache_dir
        self.motion_size = int(motion_size)
        self.motion_timepoints = tuple(float(x) for x in motion_timepoints)
        self.motion_threshold = float(motion_threshold)
        self.motion_blur_kernel = int(motion_blur_kernel)
        self.num_motion_targets = len(self.motion_timepoints)

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGB")

    def _load_motion_targets(self, row: pd.Series) -> Tuple[Optional[np.ndarray], float]:
        if not self.use_motion_aux:
            return None, 0.0
        video_path = str(row.get("video_path", ""))
        if not video_path or not os.path.exists(video_path):
            return None, 0.0
        try:
            arr = load_or_build_motion_targets(
                video_path=video_path,
                cache_dir=self.motion_cache_dir,
                motion_size=self.motion_size,
                timepoints=self.motion_timepoints,
                threshold=self.motion_threshold,
                blur_kernel=self.motion_blur_kernel,
            )
            return arr.astype(np.float32), 1.0
        except Exception:
            return None, 0.0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        front = self._load_rgb(row["front_path"])
        top = self._load_rgb(row["top_path"])
        motion_targets, motion_valid = self._load_motion_targets(row)

        if self.transform is not None:
            front, top, motion_targets_t = self.transform(front, top, motion_targets)
        else:
            motion_targets_t = torch.as_tensor(motion_targets, dtype=torch.float32) if motion_targets is not None else None

        if motion_targets_t is None:
            motion_targets_t = torch.zeros(self.num_motion_targets, self.motion_size, self.motion_size, dtype=torch.float32)

        sample_id = str(row["id"])
        if self.is_test:
            return front, top, sample_id

        label = torch.tensor([float(row["label_float"])], dtype=torch.float32)
        motion_valid_t = torch.tensor([float(motion_valid)], dtype=torch.float32)
        return front, top, label, sample_id, motion_targets_t, motion_valid_t


# ============================================================
# Model
# ============================================================
class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class MultiViewPhysNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        dropout: float = 0.20,
        num_motion_targets: int = 4,
    ):
        super().__init__()
        backbone_name = str(backbone_name).lower()
        if backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            builder = models.resnet34
            feature_dim = 512
            feature_extractor = lambda base: nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            builder = models.resnet50
            feature_dim = 2048
            feature_extractor = lambda base: nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            builder = models.resnet101
            feature_dim = 2048
            feature_extractor = lambda base: nn.Sequential(*list(base.children())[:-2])
        elif backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            builder = models.convnext_tiny
            feature_dim = 768
            feature_extractor = lambda base: base.features
        elif backbone_name == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            builder = models.convnext_small
            feature_dim = 768
            feature_extractor = lambda base: base.features
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone_name = backbone_name
        try:
            base = builder(weights=weights)
        except Exception as e:
            print(f"[Warning] {backbone_name} pretrained load 실패 -> random init 사용: {e}")
            base = builder(weights=None)

        self.features = feature_extractor(base)
        self.fusion = nn.Sequential(
            ConvBNAct(feature_dim * 4, 512, k=1, p=0),
            ConvBNAct(512, 512, k=3),
        )
        self.context = nn.Sequential(
            ConvBNAct(512, 512, k=3),
            ConvBNAct(512, 512, k=3),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(512, 256, k=3),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(256, 128, k=3),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(128, 64, k=3),
        )
        self.motion_head = nn.Conv2d(64, int(num_motion_targets), kernel_size=1)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def encode_views(self, front: torch.Tensor, top: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([front, top], dim=0)
        x = self.features(x)
        front_map, top_map = x.chunk(2, dim=0)
        return front_map, top_map

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> Dict[str, torch.Tensor]:
        front_map, top_map = self.encode_views(front, top)
        fused = torch.cat(
            [front_map, top_map, torch.abs(front_map - top_map), front_map * top_map],
            dim=1,
        )
        fused = self.fusion(fused)
        context = self.context(fused)
        logit = self.classifier(context)

        x = self.up1(context)
        x = self.up2(x)
        x = self.up3(x)
        motion_logits = self.motion_head(x)
        return {
            "logit": logit,
            "motion_logits": motion_logits,
        }


# ============================================================
# Config
# ============================================================
@dataclass
class BaselineConfig:
    run_name: str
    mode: str = "holdout"  # holdout or cv
    data_root: str = "./open"
    save_dir: str = "./runs_physnet_baseline"

    backbone_name: str = "resnet34"
    pretrained: bool = True
    img_size: int = 224
    motion_size: int = 56
    batch_size: int = 24
    epochs: int = 16
    nfolds: int = 5
    seed: int = 42
    num_workers: int = 4

    learning_rate: float = 5e-4
    backbone_lr: float = 5e-5
    weight_decay: float = 1e-2
    dropout: float = 0.20
    label_smoothing: float = 0.02
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.0
    aug_profile: str = "strong"  # light or strong

    patience: int = 5
    warmup_pct: float = 0.10
    freeze_backbone_epochs: int = 1

    use_amp: bool = True
    pin_memory: bool = True
    tta_hflip: bool = True
    tta_scales: Tuple[float, ...] = (1.0,)
    temperature_scaling: bool = True
    dev_oversample_factor: float = 1.0
    class_balance: bool = False
    check_paths: bool = True
    save_hard_examples_topk: int = 50

    use_motion_aux: bool = True
    motion_loss_weight: float = 0.15
    motion_dice_weight: float = 0.25
    motion_timepoints: Tuple[float, ...] = (2.5, 5.0, 7.5, 10.0)
    motion_threshold: float = 0.10
    motion_blur_kernel: int = 5
    motion_cache_dir: str = "./cache_physnet_motion"

    def short(self) -> Dict[str, Any]:
        obj = asdict(self)
        obj["motion_timepoints"] = list(self.motion_timepoints)
        obj["tta_scales"] = list(self.tta_scales)
        return obj


# ============================================================
# Loader helpers
# ============================================================

def build_train_sampler(df: pd.DataFrame, cfg: BaselineConfig) -> Optional[WeightedRandomSampler]:
    weights = np.ones(len(df), dtype=np.float64)

    if cfg.dev_oversample_factor != 1.0 and "source" in df.columns:
        weights *= np.where(df["source"].astype(str).values == "dev", float(cfg.dev_oversample_factor), 1.0)

    if cfg.class_balance and "label" in df.columns:
        counts = df["label"].value_counts().to_dict()
        class_weight = {k: len(df) / max(v, 1) for k, v in counts.items()}
        weights *= df["label"].map(class_weight).astype(np.float64).values

    if np.allclose(weights, np.ones_like(weights)):
        return None

    weights_t = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(df), replacement=True)



def build_loader(
    df: pd.DataFrame,
    transform,
    cfg: BaselineConfig,
    device: torch.device,
    is_test: bool,
    shuffle: bool,
    sampler=None,
) -> DataLoader:
    ds = MultiViewPhysNetDataset(
        df=df,
        transform=transform,
        is_test=is_test,
        use_motion_aux=cfg.use_motion_aux,
        motion_cache_dir=cfg.motion_cache_dir,
        motion_size=cfg.motion_size,
        motion_timepoints=cfg.motion_timepoints,
        motion_threshold=cfg.motion_threshold,
        motion_blur_kernel=cfg.motion_blur_kernel,
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
# Loss / calibration
# ============================================================

def smooth_targets(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing



def classification_loss(logits: torch.Tensor, y: torch.Tensor, smoothing: float) -> torch.Tensor:
    y_sm = smooth_targets(y, smoothing=smoothing)
    return F.binary_cross_entropy_with_logits(logits, y_sm)



def motion_aux_loss(
    motion_logits: torch.Tensor,
    motion_targets: torch.Tensor,
    motion_valid: torch.Tensor,
    dice_weight: float = 0.25,
) -> torch.Tensor:
    valid_mask = motion_valid.reshape(-1) > 0.5
    if valid_mask.sum().item() == 0:
        return motion_logits.sum() * 0.0

    pred = motion_logits[valid_mask]            # [Nv, T, H, W]
    target = motion_targets[valid_mask]         # [Nv, T, H, W]
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)

    bce = F.binary_cross_entropy_with_logits(pred, target)
    if dice_weight <= 0:
        return bce

    prob = torch.sigmoid(pred)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()
    return bce + float(dice_weight) * dice



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


# ============================================================
# Train / evaluate / infer
# ============================================================

def maybe_freeze_backbone(model: MultiViewPhysNet, freeze: bool) -> None:
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
        self.valid_logits: Optional[np.ndarray] = None
        self.valid_labels: Optional[np.ndarray] = None
        self.valid_ids: Optional[np.ndarray] = None
        self.valid_probs_raw: Optional[np.ndarray] = None
        self.valid_probs_cal: Optional[np.ndarray] = None



def build_optimizer_and_scheduler(model: MultiViewPhysNet, cfg: BaselineConfig, steps_per_epoch: int):
    head_params = list(model.fusion.parameters())
    head_params += list(model.context.parameters())
    head_params += list(model.up1.parameters())
    head_params += list(model.up2.parameters())
    head_params += list(model.up3.parameters())
    head_params += list(model.motion_head.parameters())
    head_params += list(model.classifier.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.backbone_lr, cfg.learning_rate],
        epochs=max(cfg.epochs, 1),
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=float(cfg.warmup_pct),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler



def train_one_epoch(
    model: MultiViewPhysNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    cfg: BaselineConfig,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for front, top, y, _, motion_targets, motion_valid in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        motion_targets = motion_targets.to(device, non_blocking=True)
        motion_valid = motion_valid.to(device, non_blocking=True)
        batch_size = front.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            do_mixup = (
                cfg.mixup_alpha > 0
                and random.random() < cfg.mixup_prob
                and not bool(cfg.use_motion_aux)
            )

            if do_mixup:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                idx = torch.randperm(batch_size, device=device)
                mixed_front = lam * front + (1.0 - lam) * front[idx]
                mixed_top = lam * top + (1.0 - lam) * top[idx]
                y_a, y_b = y, y[idx]
                out = model(mixed_front, mixed_top)
                loss = lam * classification_loss(out["logit"], y_a, cfg.label_smoothing)
                loss = loss + (1.0 - lam) * classification_loss(out["logit"], y_b, cfg.label_smoothing)
            else:
                out = model(front, top)
                loss = classification_loss(out["logit"], y, cfg.label_smoothing)
                if cfg.use_motion_aux:
                    aux = motion_aux_loss(
                        motion_logits=out["motion_logits"],
                        motion_targets=motion_targets,
                        motion_valid=motion_valid,
                        dice_weight=cfg.motion_dice_weight,
                    )
                    loss = loss + float(cfg.motion_loss_weight) * aux

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
def validate_one_epoch(model: MultiViewPhysNet, loader: DataLoader, cfg: BaselineConfig, device: torch.device, use_amp: bool) -> EpochOutput:
    model.eval()
    out = EpochOutput()
    total_bce = 0.0
    total_count = 0

    logits_all: List[float] = []
    labels_all: List[float] = []
    ids_all: List[str] = []

    for front, top, y, sample_ids, _, _ in loader:
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = front.size(0)

        with autocast_context(device.type, use_amp):
            pred = model(front, top)
            logits = pred["logit"]
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


def apply_center_scale_tta(x: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if abs(scale - 1.0) < 1e-8:
        return x
    if scale <= 0.0 or scale > 1.0:
        raise ValueError(f"tta scale must be in (0, 1], got {scale}")
    h, w = x.shape[-2:]
    crop_h = max(1, int(round(h * scale)))
    crop_w = max(1, int(round(w * scale)))
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return TF.resized_crop(
        x,
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        size=[h, w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


@torch.no_grad()
def infer_logits(
    model: MultiViewPhysNet,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta_hflip: bool,
    tta_scales: Sequence[float],
) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    logits_all: List[float] = []
    ids_all: List[str] = []
    scales = tuple(float(s) for s in tta_scales) if tta_scales else (1.0,)
    for batch in tqdm(loader, leave=False):
        if len(batch) == 3:
            front, top, sample_ids = batch
        elif len(batch) >= 4:
            front, top, _, sample_ids = batch[:4]
        else:
            raise ValueError(f"Unexpected batch structure in infer_logits: {len(batch)} items")
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        with autocast_context(device.type, use_amp):
            logits_parts: List[torch.Tensor] = []
            for scale in scales:
                front_view = apply_center_scale_tta(front, scale)
                top_view = apply_center_scale_tta(top, scale)
                pred = model(front_view, top_view)
                logits_parts.append(pred["logit"])
                if tta_hflip:
                    pred_flip = model(torch.flip(front_view, dims=[3]), torch.flip(top_view, dims=[3]))
                    logits_parts.append(pred_flip["logit"])
            logits = torch.stack(logits_parts, dim=0).mean(dim=0)
        logits_all.extend(logits.float().cpu().numpy().reshape(-1).tolist())
        ids_all.extend(list(sample_ids))
    return np.asarray(logits_all, dtype=np.float64), ids_all



def compute_source_metrics(df: pd.DataFrame, prob_col: str = "unstable_prob") -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for source_name, sub in df.groupby("source"):
        if "label_float" not in sub.columns or sub["label_float"].isna().all():
            continue
        y = sub["label_float"].values.astype(np.float64)
        p = sub[prob_col].values.astype(np.float64)
        metrics[str(source_name)] = {
            "logloss": dacon_logloss(y, p),
            "accuracy": binary_accuracy(y, p),
            "n": int(len(sub)),
        }
    return metrics



def save_hard_examples(df: pd.DataFrame, out_path: str, topk: int = 50) -> None:
    save_df = df.sort_values("sample_logloss", ascending=False).head(topk).copy()
    save_df.to_csv(out_path, index=False)


# ============================================================
# Core run helpers
# ============================================================

def train_with_validation(
    cfg: BaselineConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
    fold_idx: Optional[int] = None,
) -> Dict[str, Any]:
    seed_everything(cfg.seed + (0 if fold_idx is None else fold_idx * 1000))
    train_tf = PairTrainTransform(cfg.img_size, cfg.motion_size, aug_profile=cfg.aug_profile)
    valid_tf = PairEvalTransform(cfg.img_size, cfg.motion_size)

    sampler = build_train_sampler(train_df, cfg)
    train_loader = build_loader(train_df, train_tf, cfg, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    valid_loader = build_loader(valid_df, valid_tf, cfg, device=device, is_test=False, shuffle=False, sampler=None)
    test_loader = build_loader(test_df, valid_tf, cfg, device=device, is_test=True, shuffle=False, sampler=None)

    model = MultiViewPhysNet(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        dropout=cfg.dropout,
        num_motion_targets=len(cfg.motion_timepoints),
    ).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, steps_per_epoch=len(train_loader))
    scaler = build_grad_scaler(device.type, enabled=use_amp)

    best_score = float("inf")
    best_epoch = -1
    best_state = None
    best_epoch_output: Optional[EpochOutput] = None
    epochs_no_improve = 0
    epoch_records: List[Dict[str, Any]] = []

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        maybe_freeze_backbone(model, freeze=(epoch <= cfg.freeze_backbone_epochs))
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

        fold_tag = f"fold {fold_idx}" if fold_idx is not None else "holdout"
        print(
            f"[{cfg.run_name} | {fold_tag}] Epoch {epoch:02d}/{cfg.epochs} | "
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

    train_time = time.time() - t0
    if best_state is None or best_epoch_output is None:
        raise RuntimeError("No valid checkpoint was produced.")

    suffix = "" if fold_idx is None else f"_fold{fold_idx}"
    ckpt_path = os.path.join(run_dir, f"best_{cfg.run_name}{suffix}.pt")
    torch.save(best_state, ckpt_path)
    pd.DataFrame(epoch_records).to_csv(os.path.join(run_dir, f"epochs_{cfg.run_name}{suffix}.csv"), index=False)

    best_model = MultiViewPhysNet(
        backbone_name=cfg.backbone_name,
        pretrained=False,
        dropout=cfg.dropout,
        num_motion_targets=len(cfg.motion_timepoints),
    ).to(device)
    best_model.load_state_dict(best_state["model"])

    valid_logits, valid_ids = infer_logits(
        best_model,
        valid_loader,
        device=device,
        use_amp=use_amp,
        tta_hflip=cfg.tta_hflip,
        tta_scales=cfg.tta_scales,
    )
    test_logits, test_ids = infer_logits(
        best_model,
        test_loader,
        device=device,
        use_amp=use_amp,
        tta_hflip=cfg.tta_hflip,
        tta_scales=cfg.tta_scales,
    )
    valid_probs_raw = sigmoid_np(valid_logits)
    valid_probs_cal = apply_temperature_to_logits(valid_logits, best_state["temperature"])
    test_probs_raw = sigmoid_np(test_logits)
    test_probs_cal = apply_temperature_to_logits(test_logits, best_state["temperature"])

    valid_pred_df = valid_df[["id", "source", "label", "label_float", "front_path", "top_path"]].reset_index(drop=True).copy()
    valid_pred_df["valid_infer_id"] = valid_ids
    valid_pred_df["logit"] = valid_logits
    valid_pred_df["unstable_prob_raw"] = valid_probs_raw
    valid_pred_df["unstable_prob"] = valid_probs_cal
    valid_pred_df["stable_prob"] = 1.0 - valid_pred_df["unstable_prob"]
    valid_pred_df["temperature"] = float(best_state["temperature"])
    valid_pred_df["sample_logloss"] = per_sample_logloss(valid_pred_df["label_float"].values, valid_pred_df["unstable_prob"].values)
    valid_pred_df = valid_pred_df.drop(columns=["valid_infer_id"])
    valid_pred_csv = os.path.join(run_dir, f"valid_predictions_{cfg.run_name}{suffix}.csv")
    valid_pred_df.to_csv(valid_pred_csv, index=False)
    save_hard_examples(valid_pred_df, os.path.join(run_dir, f"hard_examples_{cfg.run_name}{suffix}.csv"), topk=cfg.save_hard_examples_topk)

    test_pred_df = pd.DataFrame({
        "id": test_ids,
        "logit": test_logits,
        "unstable_prob_raw": test_probs_raw,
        "unstable_prob": test_probs_cal,
        "stable_prob": 1.0 - test_probs_cal,
        "temperature": float(best_state["temperature"]),
    })
    test_pred_csv = os.path.join(run_dir, f"test_predictions_{cfg.run_name}{suffix}.csv")
    test_pred_df.to_csv(test_pred_csv, index=False)

    metrics = {
        "run_name": cfg.run_name,
        "mode": cfg.mode,
        "fold": None if fold_idx is None else int(fold_idx),
        "backbone": f"physnet_{cfg.backbone_name}_multiview",
        "img_size": cfg.img_size,
        "motion_size": cfg.motion_size,
        "seed": cfg.seed,
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_time),
        "temperature": float(best_state["temperature"]),
        "valid_logloss_raw": float(dacon_logloss(valid_df["label_float"].values, valid_pred_df["unstable_prob_raw"].values)),
        "valid_logloss_cal": float(dacon_logloss(valid_df["label_float"].values, valid_pred_df["unstable_prob"].values)),
        "valid_accuracy": float(binary_accuracy(valid_df["label_float"].values, valid_pred_df["unstable_prob"].values)),
        "source_metrics": compute_source_metrics(valid_pred_df, prob_col="unstable_prob"),
        "checkpoint": ckpt_path,
        "pred_valid_csv": valid_pred_csv,
        "pred_test_csv": test_pred_csv,
    }
    save_json(os.path.join(run_dir, f"result_{cfg.run_name}{suffix}.json"), metrics)
    return {
        "metrics": metrics,
        "valid_df": valid_pred_df,
        "test_df": test_pred_df,
        "state": best_state,
    }


# ============================================================
# Holdout / CV runners
# ============================================================

def run_holdout(cfg: BaselineConfig, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict[str, Any]:
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
        "backbone": f"physnet_{cfg.backbone_name}_multiview",
        "submission": submission_path,
        "submission_alias": submission_alias_path,
        "valid_logloss_cal": float(result["metrics"]["valid_logloss_cal"]),
        "dev_logloss_cal": float(result["metrics"]["source_metrics"].get("dev", {}).get("logloss", result["metrics"]["valid_logloss_cal"])),
        "valid_accuracy": float(result["metrics"]["valid_accuracy"]),
    }
    save_json(os.path.join(run_dir, "holdout_summary.json"), summary)
    return {
        "summary": summary,
        "result": result,
        "submission_df": submission,
    }



def run_cv(cfg: BaselineConfig, full_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict[str, Any]:
    strat_target = choose_stratify_target(full_df, cfg.nfolds)
    skf = StratifiedKFold(n_splits=cfg.nfolds, shuffle=True, random_state=cfg.seed)

    oof_frames: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    test_pred_frames: List[pd.DataFrame] = []

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

    fold_metrics_df = pd.DataFrame([
        {**{k: v for k, v in m.items() if k != "source_metrics"}, "source_metrics": json.dumps(m["source_metrics"], ensure_ascii=False)}
        for m in fold_metrics
    ])
    fold_metrics_csv = os.path.join(run_dir, "fold_metrics.csv")
    fold_metrics_df.to_csv(fold_metrics_csv, index=False)

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    oof_df = oof_df.sort_values(["source", "id"]).reset_index(drop=True)
    oof_csv = os.path.join(run_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv, index=False)

    oof_logloss_cal = dacon_logloss(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    oof_accuracy = binary_accuracy(oof_df["label_float"].values, oof_df["unstable_prob"].values)
    source_metrics = compute_source_metrics(oof_df, prob_col="unstable_prob")

    test_stack = pd.concat(test_pred_frames, axis=0, ignore_index=True)

    def blend_from_weights(weight_col: str) -> pd.DataFrame:
        parts: List[Dict[str, Any]] = []
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
        "backbone": f"physnet_{cfg.backbone_name}_multiview",
        "oof_logloss_cal": float(oof_logloss_cal),
        "oof_accuracy": float(oof_accuracy),
        "source_metrics": source_metrics,
        "fold_mean_valid_logloss_cal": float(fold_metrics_df["valid_logloss_cal"].mean()),
        "fold_mean_valid_logloss_raw": float(fold_metrics_df["valid_logloss_raw"].mean()),
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
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PhysNet-inspired multi-view baseline")
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_physnet_baseline")
    parser.add_argument("--run_name", type=str, default="physnet_multiview")
    parser.add_argument("--backbone_name", choices=["resnet34", "resnet50", "resnet101", "convnext_tiny", "convnext_small"], default="resnet34")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--motion_size", type=int, default=56)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)
    parser.add_argument("--mixup_prob", type=float, default=0.0)
    parser.add_argument("--aug_profile", choices=["light", "strong"], default="strong")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup_pct", type=float, default=0.10)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=1)

    parser.add_argument("--dev_oversample_factor", type=float, default=1.0)
    parser.add_argument("--class_balance", action="store_true", default=False)
    parser.add_argument("--tta_hflip", action="store_true", default=True)
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")
    parser.add_argument("--tta_scales", type=str, default="1.0")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--temperature_scaling", action="store_true", default=True)
    parser.add_argument("--no_temperature_scaling", action="store_false", dest="temperature_scaling")
    parser.add_argument("--check_paths", action="store_true", default=True)
    parser.add_argument("--no_check_paths", action="store_false", dest="check_paths")
    parser.add_argument("--save_hard_examples_topk", type=int, default=50)

    parser.add_argument("--use_motion_aux", action="store_true", default=True)
    parser.add_argument("--no_use_motion_aux", action="store_false", dest="use_motion_aux")
    parser.add_argument("--motion_loss_weight", type=float, default=0.15)
    parser.add_argument("--motion_dice_weight", type=float, default=0.25)
    parser.add_argument("--motion_timepoints", type=str, default="2.5,5.0,7.5,10.0")
    parser.add_argument("--motion_threshold", type=float, default=0.10)
    parser.add_argument("--motion_blur_kernel", type=int, default=5)
    parser.add_argument("--motion_cache_dir", type=str, default="./cache_physnet_motion")
    return parser



def make_config_from_args(ns: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        run_name=ns.run_name,
        mode=ns.mode,
        data_root=ns.data_root,
        save_dir=ns.save_dir,
        backbone_name=ns.backbone_name,
        pretrained=True,
        img_size=ns.img_size,
        motion_size=ns.motion_size,
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
        mixup_alpha=ns.mixup_alpha,
        mixup_prob=ns.mixup_prob,
        aug_profile=ns.aug_profile,
        patience=ns.patience,
        warmup_pct=ns.warmup_pct,
        freeze_backbone_epochs=ns.freeze_backbone_epochs,
        use_amp=ns.use_amp,
        pin_memory=ns.pin_memory,
        tta_hflip=ns.tta_hflip,
        tta_scales=parse_float_list(ns.tta_scales),
        temperature_scaling=ns.temperature_scaling,
        dev_oversample_factor=ns.dev_oversample_factor,
        class_balance=ns.class_balance,
        check_paths=ns.check_paths,
        save_hard_examples_topk=ns.save_hard_examples_topk,
        use_motion_aux=ns.use_motion_aux,
        motion_loss_weight=ns.motion_loss_weight,
        motion_dice_weight=ns.motion_dice_weight,
        motion_timepoints=parse_float_list(ns.motion_timepoints),
        motion_threshold=ns.motion_threshold,
        motion_blur_kernel=ns.motion_blur_kernel,
        motion_cache_dir=ns.motion_cache_dir,
    )



def run_baseline(cfg: BaselineConfig) -> str:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp and device.type == "cuda")

    train_df = load_split_df(cfg.data_root, "train")
    dev_df = load_split_df(cfg.data_root, "dev")
    test_df = load_split_df(cfg.data_root, "test")
    full_df = pd.concat([train_df, dev_df], ignore_index=True)

    if cfg.check_paths:
        verify_paths(train_df, check_video_when_available=False)
        verify_paths(dev_df, check_video_when_available=False)
        verify_paths(test_df, check_video_when_available=False)

    run_dir = os.path.join(cfg.save_dir, f"{cfg.run_name}_{now_str()}")
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "run_config.json"), cfg.short())

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Backbone: physnet_{cfg.backbone_name}_multiview | Mode: {cfg.mode}")
    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)} | Test samples: {len(test_df)}")
    print(f"Motion aux: {cfg.use_motion_aux} | Motion timepoints: {cfg.motion_timepoints}")
    print(f"Run dir: {run_dir}")

    if cfg.mode == "holdout":
        result = run_holdout(cfg, train_df, dev_df, test_df, run_dir, device, use_amp)
        print("\n[Holdout Summary]")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif cfg.mode == "cv":
        result = run_cv(cfg, full_df, test_df, run_dir, device, use_amp)
        print("\n[CV Summary]")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")

    return run_dir



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = make_config_from_args(args)
    run_dir = run_baseline(cfg)
    print(f"\nDone. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
