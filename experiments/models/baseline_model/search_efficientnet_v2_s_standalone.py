"""
Standalone automatic hyperparameter search for one backbone.

이 파일은 다른 shared 모듈 없이 단독 실행되도록 만든 버전입니다.
즉, 이 파일 하나만 복사해도 아래 전체 흐름이 한 번에 동작합니다.

전체 플로우
1) train/dev/test CSV와 이미지 경로를 읽습니다.
2) holdout 또는 CV 모드로 학습/검증 파이프라인을 준비합니다.
3) 하이퍼파라미터 후보를 random/grid search로 생성합니다.
4) trial마다 학습 -> 검증 -> calibration -> 제출 CSV 생성까지 수행합니다.
5) 모든 trial 결과를 ranking CSV로 저장하고, 필요하면 상위 trial만 CV로 다시 검증합니다.

이 스크립트는 DACON 구조물 안정성 대회용으로 작성되었고,
입력은 front/top 2-view 이미지, 출력은 unstable_prob / stable_prob 입니다.
"""

TARGET_BACKBONE = "efficientnet_v2_s"
TARGET_BACKBONE_TITLE = "EfficientNet-V2-S"

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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

import itertools
import traceback

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


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


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
# Transforms
# ============================================================
class PairTrainTransform:
    # [Flow] 학습 시 front / top 두 이미지에 같은 기하 변환과 같은 색상 변환을 적용합니다.
    """
    나중에 데이터 전처리 / 증강 튜닝은 이 클래스부터 손대면 됩니다.
    - aug_profile='light'
    - aug_profile='strong'
    """
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
        img.save(buf, format="JPEG", quality=int(random.randint(quality_min, quality_max)))
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
                x = torchvision.transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.12),
                    ratio=(0.3, 3.3),
                    value="random",
                )(x)
        else:
            if random.random() < 0.15:
                x = torchvision.transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.10),
                    ratio=(0.3, 3.3),
                    value="random",
                )(x)
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


# ============================================================
# Dataset
# ============================================================
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
        label = torch.tensor([float(row["label_float"])], dtype=torch.float32)
        return front, top, label, sample_id


# ============================================================
# Model
# ============================================================

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
        return {"features": base.features, "pool": base.avgpool, "feat_dim": feat_dim}

    if backbone_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        try:
            base = models.convnext_tiny(weights=weights)
        except Exception as e:
            print(f"[Warning] convnext_tiny pretrained load 실패 -> random init 사용: {e}")
            base = models.convnext_tiny(weights=None)
        feat_dim = base.classifier[2].in_features
        return {"features": base.features, "pool": nn.AdaptiveAvgPool2d((1, 1)), "feat_dim": feat_dim}

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        try:
            base = models.resnet50(weights=weights)
        except Exception as e:
            print(f"[Warning] resnet50 pretrained load 실패 -> random init 사용: {e}")
            base = models.resnet50(weights=None)
        feat_dim = base.fc.in_features
        features = nn.Sequential(*list(base.children())[:-2])
        return {"features": features, "pool": nn.AdaptiveAvgPool2d((1, 1)), "feat_dim": feat_dim}

    raise ValueError(f"Unsupported backbone: {backbone_name}")


class MultiViewFusionNet(nn.Module):
    # [Flow] 두 뷰를 같은 backbone으로 인코딩하고,
    # front/top/abs-diff/elementwise-product를 이어 붙여 최종 이진 분류를 수행합니다.
    def __init__(self, backbone_name: str, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        spec = build_backbone(backbone_name, pretrained=pretrained)
        feat_dim = int(spec["feat_dim"])
        hidden = 512 if feat_dim >= 512 else 256

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
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_views(self, front: torch.Tensor, top: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([front, top], dim=0)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        front_feat, top_feat = x.chunk(2, dim=0)
        return front_feat, top_feat

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        front_feat, top_feat = self.encode_views(front, top)
        front_feat = self.front_proj(front_feat)
        top_feat = self.top_proj(top_feat)
        fused = torch.cat(
            [front_feat, top_feat, torch.abs(front_feat - top_feat), front_feat * top_feat],
            dim=1,
        )
        return self.classifier(fused)


# ============================================================
# Config
# ============================================================
@dataclass
class BaselineConfig:
    run_name: str
    backbone: str
    mode: str = "holdout"  # holdout or cv
    data_root: str = "./open"
    save_dir: str = "./runs_backbone_baselines"

    pretrained: bool = True
    img_size: int = 384
    batch_size: int = 16
    epochs: int = 12
    nfolds: int = 5
    seed: int = 42
    num_workers: int = 4

    learning_rate: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 1e-2
    dropout: float = 0.30
    label_smoothing: float = 0.03
    mixup_alpha: float = 0.10
    mixup_prob: float = 0.25
    aug_profile: str = "light"  # light or strong

    patience: int = 4
    warmup_pct: float = 0.10
    freeze_backbone_epochs: int = 0

    use_amp: bool = True
    pin_memory: bool = True
    tta_hflip: bool = True
    temperature_scaling: bool = True
    dev_oversample_factor: float = 1.0
    class_balance: bool = False
    check_paths: bool = True
    save_hard_examples_topk: int = 50

    def short(self) -> Dict:
        return asdict(self)


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


def build_loader(df: pd.DataFrame, transform, cfg: BaselineConfig, device: torch.device, is_test: bool, shuffle: bool, sampler=None) -> DataLoader:
    ds = MultiViewDataset(df, transform=transform, is_test=is_test)
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
        self.valid_probs_raw = None
        self.valid_probs_cal = None


def build_optimizer_and_scheduler(model: MultiViewFusionNet, cfg: BaselineConfig, steps_per_epoch: int):
    optimizer = optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": cfg.backbone_lr},
            {"params": model.front_proj.parameters(), "lr": cfg.learning_rate},
            {"params": model.top_proj.parameters(), "lr": cfg.learning_rate},
            {"params": model.classifier.parameters(), "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg.backbone_lr, cfg.learning_rate, cfg.learning_rate, cfg.learning_rate],
        epochs=max(cfg.epochs, 1),
        steps_per_epoch=max(steps_per_epoch, 1),
        pct_start=float(cfg.warmup_pct),
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, scaler, cfg: BaselineConfig, device: torch.device, use_amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for front, top, y, _ in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = front.size(0)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device.type, use_amp):
            if cfg.mixup_alpha > 0 and random.random() < cfg.mixup_prob:
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                idx = torch.randperm(batch_size, device=device)
                mixed_front = lam * front + (1.0 - lam) * front[idx]
                mixed_top = lam * top + (1.0 - lam) * top[idx]
                y_a, y_b = y, y[idx]
                logits = model(mixed_front, mixed_top)
                loss = lam * classification_loss(logits, y_a, cfg.label_smoothing) + (1.0 - lam) * classification_loss(logits, y_b, cfg.label_smoothing)
            else:
                logits = model(front, top)
                loss = classification_loss(logits, y, cfg.label_smoothing)

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
def validate_one_epoch(model, loader, cfg: BaselineConfig, device: torch.device, use_amp: bool) -> EpochOutput:
    model.eval()
    out = EpochOutput()
    total_bce = 0.0
    total_count = 0

    logits_all = []
    labels_all = []
    ids_all = []

    for front, top, y, sample_ids in loader:
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = front.size(0)

        with autocast_context(device.type, use_amp):
            logits = model(front, top)
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


@torch.no_grad()
def infer_logits(model, loader, device: torch.device, use_amp: bool, tta_hflip: bool) -> Tuple[np.ndarray, List[str]]:
    model.eval()
    logits_all = []
    ids_all = []
    for front, top, sample_ids in tqdm(loader, leave=False):
        front = front.to(device, non_blocking=True)
        top = top.to(device, non_blocking=True)
        with autocast_context(device.type, use_amp):
            logits = model(front, top)
            if tta_hflip:
                logits_flip = model(torch.flip(front, dims=[3]), torch.flip(top, dims=[3]))
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


def save_hard_examples(df: pd.DataFrame, out_path: str, topk: int = 50) -> None:
    save_df = df.sort_values("sample_logloss", ascending=False).head(topk).copy()
    save_df.to_csv(out_path, index=False)


# ============================================================
# Core run helpers
# ============================================================

def train_with_validation(
    # [Flow] 한 번의 학습-검증 루프를 수행하는 핵심 함수입니다.
    # - train loader / valid loader / test loader 생성
    # - epoch 반복 학습
    # - 가장 좋은 validation 성능 checkpoint 저장
    # - validation logits로 temperature scaling 적용
    # - test 예측 CSV까지 저장
    cfg: BaselineConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
    fold_idx: Optional[int] = None,
) -> Dict:
    seed_everything(cfg.seed + (0 if fold_idx is None else fold_idx * 1000))
    train_tf = PairTrainTransform(cfg.img_size, aug_profile=cfg.aug_profile)
    valid_tf = PairEvalTransform(cfg.img_size)

    sampler = build_train_sampler(train_df, cfg)
    train_loader = build_loader(train_df, train_tf, cfg, device=device, is_test=False, shuffle=(sampler is None), sampler=sampler)
    valid_loader = build_loader(valid_df, valid_tf, cfg, device=device, is_test=False, shuffle=False, sampler=None)
    test_loader = build_loader(test_df, valid_tf, cfg, device=device, is_test=True, shuffle=False, sampler=None)

    model = MultiViewFusionNet(backbone_name=cfg.backbone, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
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
    ckpt_path = os.path.join(run_dir, f"best_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.pt")
    torch.save(best_state, ckpt_path)
    pd.DataFrame(epoch_records).to_csv(
        os.path.join(run_dir, f"epochs_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        index=False,
    )

    best_model = MultiViewFusionNet(backbone_name=cfg.backbone, pretrained=False, dropout=cfg.dropout).to(device)
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
    save_hard_examples(
        valid_pred_df,
        os.path.join(run_dir, f"hard_examples_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        topk=cfg.save_hard_examples_topk,
    )

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
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "seed": cfg.seed,
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_time),
        "temperature": float(best_state["temperature"]),
        "valid_logloss_raw": float(best_state["best_valid_logloss_raw"]),
        "valid_logloss_cal": float(best_state["best_valid_logloss_cal"]),
        "valid_accuracy": float(binary_accuracy(valid_df["label_float"].values, valid_pred_df["unstable_prob"].values)),
        "source_metrics": compute_source_metrics(valid_pred_df, prob_col="unstable_prob"),
        "checkpoint": ckpt_path,
        "pred_valid_csv": valid_pred_csv,
        "pred_test_csv": test_pred_csv,
    }
    save_json(
        os.path.join(run_dir, f"result_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.json"),
        metrics,
    )
    return {
        "metrics": metrics,
        "valid_df": valid_pred_df,
        "test_df": test_pred_df,
        "state": best_state,
    }


# ============================================================
# Holdout / CV runners
# ============================================================

def run_holdout(cfg: BaselineConfig, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict:
    # [Flow] Holdout 실험
    # train 데이터로 학습하고, dev 데이터로 검증합니다.
    result = train_with_validation(cfg, train_df, dev_df, test_df, run_dir, device, use_amp, fold_idx=None)
    submission = result["test_df"][["id", "unstable_prob", "stable_prob"]].sort_values("id").reset_index(drop=True)
    submission_path = os.path.join(run_dir, "submission_holdout.csv")
    submission.to_csv(submission_path, index=False)
    summary = {
        "run_name": cfg.run_name,
        "mode": "holdout",
        "backbone": cfg.backbone,
        "submission": submission_path,
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


def run_cv(cfg: BaselineConfig, full_df: pd.DataFrame, test_df: pd.DataFrame, run_dir: str, device: torch.device, use_amp: bool) -> Dict:
    # [Flow] K-fold Cross Validation
    # full_df(train+dev)를 nfold로 나누고 각 fold에서 학습/검증을 반복합니다.
    # 마지막에는 fold별 test 예측을 equal / weighted 두 방식으로 앙상블합니다.
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

    fold_metrics_df = pd.DataFrame([
        {
            **{k: v for k, v in m.items() if k != "source_metrics"},
            "source_metrics": json.dumps(m["source_metrics"], ensure_ascii=False),
        }
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
    sub_equal_path = os.path.join(run_dir, "submission_cv_equal.csv")
    sub_weighted_path = os.path.join(run_dir, "submission_cv_weighted.csv")
    sub_equal.to_csv(sub_equal_path, index=False)
    sub_weighted.to_csv(sub_weighted_path, index=False)

    summary = {
        "run_name": cfg.run_name,
        "mode": "cv",
        "backbone": cfg.backbone,
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
# Presets and CLI
# ============================================================
PRESETS: Dict[str, Dict] = {
    "efficientnet_v2_s": {
        "backbone": "efficientnet_v2_s",
        "img_size": 384,
        "batch_size": 16,
        "epochs": 12,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.05,
        "mixup_alpha": 0.20,
        "mixup_prob": 0.40,
        "aug_profile": "light",
    },
    "convnext_tiny": {
        "backbone": "convnext_tiny",
        "img_size": 320,
        "batch_size": 16,
        "epochs": 12,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.10,
        "mixup_prob": 0.25,
        "aug_profile": "strong",
    },
    "resnet50": {
        "backbone": "resnet50",
        "img_size": 320,
        "batch_size": 16,
        "epochs": 12,
        "dropout": 0.30,
        "learning_rate": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.03,
        "mixup_alpha": 0.10,
        "mixup_prob": 0.25,
        "aug_profile": "light",
    },
}


def build_parser(default_preset_name: str) -> argparse.ArgumentParser:
    preset = PRESETS[default_preset_name]
    parser = argparse.ArgumentParser(description=f"Baseline trainer for {default_preset_name}")
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_backbone_baselines")
    parser.add_argument("--run_name", type=str, default=f"baseline_{default_preset_name}")

    parser.add_argument("--img_size", type=int, default=preset["img_size"])
    parser.add_argument("--batch_size", type=int, default=preset["batch_size"])
    parser.add_argument("--epochs", type=int, default=preset["epochs"])
    parser.add_argument("--nfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=preset["learning_rate"])
    parser.add_argument("--backbone_lr", type=float, default=preset["backbone_lr"])
    parser.add_argument("--weight_decay", type=float, default=preset["weight_decay"])
    parser.add_argument("--dropout", type=float, default=preset["dropout"])
    parser.add_argument("--label_smoothing", type=float, default=preset["label_smoothing"])
    parser.add_argument("--mixup_alpha", type=float, default=preset["mixup_alpha"])
    parser.add_argument("--mixup_prob", type=float, default=preset["mixup_prob"])
    parser.add_argument("--aug_profile", choices=["light", "strong"], default=preset["aug_profile"])

    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--warmup_pct", type=float, default=0.10)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)

    parser.add_argument("--dev_oversample_factor", type=float, default=1.0)
    parser.add_argument("--class_balance", action="store_true", default=False)
    parser.add_argument("--tta_hflip", action="store_true", default=True)
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--temperature_scaling", action="store_true", default=True)
    parser.add_argument("--no_temperature_scaling", action="store_false", dest="temperature_scaling")
    parser.add_argument("--check_paths", action="store_true", default=True)
    parser.add_argument("--no_check_paths", action="store_false", dest="check_paths")
    parser.add_argument("--save_hard_examples_topk", type=int, default=50)
    return parser


def make_config_from_args(default_preset_name: str, ns: argparse.Namespace) -> BaselineConfig:
    preset = PRESETS[default_preset_name]
    return BaselineConfig(
        run_name=ns.run_name,
        backbone=preset["backbone"],
        mode=ns.mode,
        data_root=ns.data_root,
        save_dir=ns.save_dir,
        pretrained=True,
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
        mixup_alpha=ns.mixup_alpha,
        mixup_prob=ns.mixup_prob,
        aug_profile=ns.aug_profile,
        patience=ns.patience,
        warmup_pct=ns.warmup_pct,
        freeze_backbone_epochs=ns.freeze_backbone_epochs,
        use_amp=ns.use_amp,
        pin_memory=ns.pin_memory,
        tta_hflip=ns.tta_hflip,
        temperature_scaling=ns.temperature_scaling,
        dev_oversample_factor=ns.dev_oversample_factor,
        class_balance=ns.class_balance,
        check_paths=ns.check_paths,
        save_hard_examples_topk=ns.save_hard_examples_topk,
    )


def run_baseline(cfg: BaselineConfig) -> str:
    # [Flow] 단일 실험 실행 함수
    # 1) 데이터 로드
    # 2) 경로 검증
    # 3) holdout 또는 CV 학습 수행
    # 4) summary / submission CSV 저장
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp and device.type == "cuda")

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
    print(f"Backbone: {cfg.backbone} | Mode: {cfg.mode}")
    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)} | Test samples: {len(test_df)}")
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


def run_with_preset(default_preset_name: str) -> None:
    parser = build_parser(default_preset_name)
    args = parser.parse_args()
    cfg = make_config_from_args(default_preset_name, args)
    run_dir = run_baseline(cfg)
    print(f"\nDone. Outputs saved to: {run_dir}")






# ============================================================
# Search spaces
# - Edit these lists freely to widen / narrow the search.
# - Random search uses all keys below.
# - Grid search uses DEFAULT_GRID_KEYS[backbone].
# ============================================================
SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    TARGET_BACKBONE: {
        "img_size": [320, 384],
        "batch_size": [16, 12, 8],
        "learning_rate": [5e-4, 1e-3, 1.5e-3],
        "backbone_lr": [5e-5, 1e-4, 2e-4],
        "weight_decay": [5e-3, 1e-2, 2e-2],
        "dropout": [0.20, 0.30, 0.40],
        "label_smoothing": [0.00, 0.03, 0.05, 0.10],
        "mixup_alpha": [0.00, 0.10, 0.20],
        "mixup_prob": [0.00, 0.25, 0.40],
        "aug_profile": ["light", "strong"],
        "freeze_backbone_epochs": [0, 1],
        "dev_oversample_factor": [1.0, 2.0, 3.0],
        "class_balance": [False, True],
    },
}

DEFAULT_GRID_KEYS: Dict[str, List[str]] = {
    TARGET_BACKBONE: [
        "img_size",
        "learning_rate",
        "label_smoothing",
        "mixup_alpha",
        "mixup_prob",
        "aug_profile",
        "dev_oversample_factor",
    ],
}

# ============================================================
# Small helpers
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TimeBudget:
    def __init__(self, hours: float):
        self.start = time.time()
        self.hours = float(hours)

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.start) / 3600.0

    @property
    def remaining_hours(self) -> float:
        return max(self.hours - self.elapsed_hours, 0.0)

    def can_start(self, min_remaining_minutes: float) -> bool:
        return self.remaining_hours * 60.0 >= float(min_remaining_minutes)

    def state(self) -> Dict[str, float]:
        return {
            "elapsed_hours": self.elapsed_hours,
            "remaining_hours": self.remaining_hours,
            "budget_hours": self.hours,
        }


def canonicalize_value(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        return round(float(v), 10)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    cand = {k: canonicalize_value(v) for k, v in candidate.items()}

    mix_alpha = float(cand.get("mixup_alpha", 0.0))
    mix_prob = float(cand.get("mixup_prob", 0.0))
    if mix_alpha <= 0.0 or mix_prob <= 0.0:
        cand["mixup_alpha"] = 0.0
        cand["mixup_prob"] = 0.0

    if float(cand.get("dev_oversample_factor", 1.0)) < 1.0:
        cand["dev_oversample_factor"] = 1.0

    return cand


def candidate_signature(backbone: str, candidate: Dict[str, Any], mode: str, epochs: int) -> str:
    payload = {
        "backbone": backbone,
        "mode": mode,
        "epochs": int(epochs),
        **normalize_candidate(candidate),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# ============================================================
# Candidate generation
# ============================================================

def sample_random_candidates(
    backbone: str,
    n_trials: int,
    seed: int,
    search_space: Dict[str, Sequence[Any]],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(search_space.keys())
    seen = set()
    out = []
    max_attempts = max(n_trials * 50, 200)

    for _ in range(max_attempts):
        cand = {k: rng.choice(list(search_space[k])) for k in keys}
        cand = normalize_candidate(cand)
        sig = candidate_signature(backbone, cand, mode="holdout", epochs=0)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= n_trials:
            break
    return out


def generate_grid_candidates(
    backbone: str,
    search_space: Dict[str, Sequence[Any]],
    grid_keys: Sequence[str],
    seed: int,
    max_trials: Optional[int] = None,
) -> List[Dict[str, Any]]:
    preset = PRESETS[backbone]
    keys = [k for k in grid_keys if k in search_space]
    value_lists = [list(search_space[k]) for k in keys]

    grid = []
    for values in itertools.product(*value_lists):
        cand = {k: preset.get(k) for k in search_space.keys() if k not in keys and k in preset}
        for k, v in zip(keys, values):
            cand[k] = v
        cand = normalize_candidate(cand)
        grid.append(cand)

    rng = random.Random(seed)
    rng.shuffle(grid)
    if max_trials is not None:
        grid = grid[: int(max_trials)]
    return grid


# ============================================================
# Trial building / parsing
# ============================================================

def make_config(
    backbone: str,
    mode: str,
    trial_name: str,
    data_root: str,
    save_dir: str,
    num_workers: int,
    trial_seed: int,
    epochs: int,
    nfolds: int,
    use_amp: bool,
    pin_memory: bool,
    tta_hflip: bool,
    check_paths: bool,
    search_params: Dict[str, Any],
) -> BaselineConfig:
    preset = PRESETS[backbone]
    cfg = BaselineConfig(
        run_name=trial_name,
        backbone=backbone,
        mode=mode,
        data_root=data_root,
        save_dir=save_dir,
        pretrained=True,
        img_size=int(search_params.get("img_size", preset["img_size"])),
        batch_size=int(search_params.get("batch_size", preset["batch_size"])),
        epochs=int(epochs),
        nfolds=int(nfolds),
        seed=int(trial_seed),
        num_workers=int(num_workers),
        learning_rate=float(search_params.get("learning_rate", preset["learning_rate"])),
        backbone_lr=float(search_params.get("backbone_lr", preset["backbone_lr"])),
        weight_decay=float(search_params.get("weight_decay", preset["weight_decay"])),
        dropout=float(search_params.get("dropout", preset["dropout"])),
        label_smoothing=float(search_params.get("label_smoothing", preset["label_smoothing"])),
        mixup_alpha=float(search_params.get("mixup_alpha", preset["mixup_alpha"])),
        mixup_prob=float(search_params.get("mixup_prob", preset["mixup_prob"])),
        aug_profile=str(search_params.get("aug_profile", preset["aug_profile"])),
        patience=4,
        warmup_pct=0.10,
        freeze_backbone_epochs=int(search_params.get("freeze_backbone_epochs", 0)),
        use_amp=bool(use_amp),
        pin_memory=bool(pin_memory),
        tta_hflip=bool(tta_hflip),
        temperature_scaling=True,
        dev_oversample_factor=float(search_params.get("dev_oversample_factor", 1.0)),
        class_balance=bool(search_params.get("class_balance", False)),
        check_paths=bool(check_paths),
        save_hard_examples_topk=50,
    )
    return cfg


def load_summary(run_dir: str, mode: str) -> Dict[str, Any]:
    if mode == "holdout":
        path = os.path.join(run_dir, "holdout_summary.json")
    elif mode == "cv":
        path = os.path.join(run_dir, "cv_summary.json")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"summary file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_trial(backbone: str, mode: str, run_dir: str, cfg: BaselineConfig, trial_idx: int, search_params: Dict[str, Any]) -> Dict[str, Any]:
    summary = load_summary(run_dir, mode)
    record: Dict[str, Any] = {
        "trial_idx": int(trial_idx),
        "status": "ok",
        "backbone": backbone,
        "mode": mode,
        "run_name": cfg.run_name,
        "run_dir": run_dir,
        "config_path": os.path.join(run_dir, "run_config.json"),
        "summary_path": os.path.join(run_dir, "holdout_summary.json" if mode == "holdout" else "cv_summary.json"),
        "epochs": int(cfg.epochs),
        "seed": int(cfg.seed),
        **normalize_candidate(search_params),
    }

    if mode == "holdout":
        record["objective"] = float(summary.get("dev_logloss_cal", summary.get("valid_logloss_cal", math.inf)))
        record["secondary"] = float(summary.get("valid_logloss_cal", math.inf))
        record["dev_logloss_cal"] = float(summary.get("dev_logloss_cal", math.inf))
        record["valid_logloss_cal"] = float(summary.get("valid_logloss_cal", math.inf))
        record["valid_accuracy"] = float(summary.get("valid_accuracy", math.nan))
    else:
        source_metrics = summary.get("source_metrics", {}) or {}
        dev_score = ((source_metrics.get("dev") or {}).get("logloss"))
        if dev_score is None:
            dev_score = summary.get("oof_logloss_cal", math.inf)
        record["objective"] = float(dev_score)
        record["secondary"] = float(summary.get("oof_logloss_cal", math.inf))
        record["dev_logloss_cal"] = float(dev_score)
        record["oof_logloss_cal"] = float(summary.get("oof_logloss_cal", math.inf))
        record["oof_accuracy"] = float(summary.get("oof_accuracy", math.nan))
        record["fold_mean_valid_logloss_cal"] = float(summary.get("fold_mean_valid_logloss_cal", math.inf))
        record["fold_mean_temperature"] = float(summary.get("fold_mean_temperature", math.nan))
    return record


# ============================================================
# Ranking / persistence
# ============================================================

def sort_history_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    ok = df[df["status"] == "ok"].copy()
    bad = df[df["status"] != "ok"].copy()

    if mode == "holdout":
        ok = ok.sort_values(["objective", "secondary", "valid_accuracy"], ascending=[True, True, False])
    else:
        sort_cols = ["objective", "secondary"]
        ascending = [True, True]
        if "oof_accuracy" in ok.columns:
            sort_cols.append("oof_accuracy")
            ascending.append(False)
        ok = ok.sort_values(sort_cols, ascending=ascending)

    bad = bad.sort_values(["trial_idx"])
    return pd.concat([ok, bad], axis=0, ignore_index=True)


def persist_search_tables(search_dir: str, history: List[Dict[str, Any]], mode: str) -> Tuple[str, str]:
    history_df = pd.DataFrame(history)
    history_path = os.path.join(search_dir, "search_history.csv")
    history_df.to_csv(history_path, index=False)

    ranking_df = sort_history_df(history_df, mode=mode)
    ranking_path = os.path.join(search_dir, "search_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False)
    return history_path, ranking_path


# ============================================================
# Trial execution
# ============================================================

def run_trial_with_oom_fallback(cfg: BaselineConfig, fallback_batches: Sequence[int]) -> Tuple[str, BaselineConfig]:
    # [Flow] GPU OOM이 나면 batch size를 줄여가며 같은 trial을 재시도합니다.
    tried = []
    batch_candidates = [cfg.batch_size] + [int(x) for x in fallback_batches if int(x) < int(cfg.batch_size)]
    batch_candidates = list(dict.fromkeys(batch_candidates))

    for bs in batch_candidates:
        trial_cfg = BaselineConfig(**asdict(cfg))
        trial_cfg.batch_size = int(bs)
        if bs != cfg.batch_size:
            trial_cfg.run_name = f"{cfg.run_name}_bs{bs}"
        try:
            run_dir = run_baseline(trial_cfg)
            return run_dir, trial_cfg
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            tried.append(bs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[OOM] {cfg.run_name} failed at batch_size={bs}; trying smaller batch if available...")
            continue

    raise RuntimeError(f"All batch sizes failed with OOM: {tried}")


# ============================================================
# Search driver
# ============================================================

def run_search(args: argparse.Namespace, default_backbone: Optional[str] = None) -> str:
    # [Flow] 자동 탐색의 메인 드라이버
    # 1) search 공간에서 후보 생성
    # 2) 시간 예산과 OOM fallback을 고려하며 trial 실행
    # 3) trial summary를 수집해 ranking CSV 생성
    # 4) 옵션에 따라 상위 holdout trial만 CV로 재검증
    backbone = default_backbone or args.backbone
    if backbone not in PRESETS:
        raise ValueError(f"Unknown backbone: {backbone}")

    search_dir = args.resume_dir or os.path.join(args.save_dir, f"search_{backbone}_{now_str()}")
    ensure_dir(search_dir)
    trial_save_root = os.path.join(search_dir, "trial_runs")
    ensure_dir(trial_save_root)

    budget = TimeBudget(args.time_budget_hours)
    fallback_batches = [int(x) for x in args.oom_batch_fallback.split(",") if str(x).strip()]

    search_space = SEARCH_SPACES[backbone]
    if args.search_method == "random":
        candidates = sample_random_candidates(
            backbone=backbone,
            n_trials=args.n_trials,
            seed=args.seed,
            search_space=search_space,
        )
    else:
        candidates = generate_grid_candidates(
            backbone=backbone,
            search_space=search_space,
            grid_keys=DEFAULT_GRID_KEYS[backbone],
            seed=args.seed,
            max_trials=args.max_grid_trials,
        )

    history: List[Dict[str, Any]] = []
    seen_signatures = set()
    existing_history_path = os.path.join(search_dir, "search_history.csv")
    if os.path.exists(existing_history_path):
        old = pd.read_csv(existing_history_path)
        if "signature" in old.columns:
            seen_signatures |= set(old["signature"].astype(str).tolist())
        history.extend(old.to_dict(orient="records"))
        print(f"[Resume] loaded {len(old)} previous trials from {existing_history_path}")

    save_json(
        os.path.join(search_dir, "search_plan.json"),
        {
            "backbone": backbone,
            "mode": args.mode,
            "search_method": args.search_method,
            "n_trials": args.n_trials,
            "max_grid_trials": args.max_grid_trials,
            "time_budget_hours": args.time_budget_hours,
            "min_remaining_minutes_to_start": args.min_remaining_minutes_to_start,
            "trial_epochs": args.trial_epochs,
            "nfolds": args.nfolds,
            "search_space": search_space,
            "grid_keys": DEFAULT_GRID_KEYS[backbone],
            "fallback_batches": fallback_batches,
        },
    )

    start_trial_idx = len(history) + 1
    print(f"[Search] backbone={backbone} | mode={args.mode} | search_method={args.search_method}")
    print(f"[Search] search_dir={search_dir}")
    print(f"[Search] candidates prepared={len(candidates)}")

    for local_idx, candidate in enumerate(candidates, start=1):
        signature = candidate_signature(backbone, candidate, mode=args.mode, epochs=args.trial_epochs)
        if signature in seen_signatures:
            continue
        if not budget.can_start(args.min_remaining_minutes_to_start):
            print("[Budget] search stopped before launching a new trial")
            break

        trial_idx = start_trial_idx
        start_trial_idx += 1
        seen_signatures.add(signature)
        trial_name = f"search_{backbone}_{args.mode}_t{trial_idx:03d}"
        cfg = make_config(
            backbone=backbone,
            mode=args.mode,
            trial_name=trial_name,
            data_root=args.data_root,
            save_dir=trial_save_root,
            num_workers=args.num_workers,
            trial_seed=args.seed + trial_idx,
            epochs=args.trial_epochs,
            nfolds=args.nfolds,
            use_amp=args.use_amp,
            pin_memory=args.pin_memory,
            tta_hflip=args.tta_hflip,
            check_paths=args.check_paths,
            search_params=candidate,
        )

        print("\n" + "=" * 90)
        print(f"[Trial {trial_idx}] {trial_name}")
        print(json.dumps({**normalize_candidate(candidate), "epochs": cfg.epochs, "mode": cfg.mode}, ensure_ascii=False, indent=2))

        t0 = time.time()
        try:
            run_dir, used_cfg = run_trial_with_oom_fallback(cfg, fallback_batches=fallback_batches)
            record = summarize_trial(backbone, args.mode, run_dir, used_cfg, trial_idx, {**candidate, "batch_size": used_cfg.batch_size})
            record["signature"] = signature
            record["elapsed_seconds"] = time.time() - t0
            history.append(record)
            print(f"[Trial {trial_idx}] objective={record['objective']:.6f} | elapsed={record['elapsed_seconds']:.1f}s")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            err_record = {
                "trial_idx": int(trial_idx),
                "status": "failed",
                "backbone": backbone,
                "mode": args.mode,
                "run_name": cfg.run_name,
                "run_dir": "",
                "config_path": "",
                "summary_path": "",
                "epochs": int(cfg.epochs),
                "seed": int(cfg.seed),
                "signature": signature,
                "elapsed_seconds": time.time() - t0,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": tb,
                **normalize_candidate(candidate),
            }
            history.append(err_record)
            print(f"[Trial {trial_idx}] FAILED: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        history_path, ranking_path = persist_search_tables(search_dir, history, mode=args.mode)
        print(f"[Search] updated history -> {history_path}")
        print(f"[Search] updated ranking -> {ranking_path}")

    ranking_df = pd.read_csv(os.path.join(search_dir, "search_ranking.csv")) if os.path.exists(os.path.join(search_dir, "search_ranking.csv")) else pd.DataFrame()
    ok_df = ranking_df[ranking_df["status"] == "ok"].copy() if len(ranking_df) else pd.DataFrame()
    if len(ok_df) > 0:
        best_row = ok_df.iloc[0].to_dict()
        save_json(os.path.join(search_dir, "best_config_holdout_or_cv.json"), best_row)

    cv_refine_records: List[Dict[str, Any]] = []
    if args.mode == "holdout" and args.refine_top_k_cv > 0 and len(ok_df) > 0:
        topk_df = ok_df.head(int(args.refine_top_k_cv)).copy()
        for rank_idx, row in enumerate(topk_df.itertuples(index=False), start=1):
            if not budget.can_start(args.min_remaining_minutes_to_start):
                print("[Budget] cv refine stopped before launching a new trial")
                break

            with open(getattr(row, "config_path"), "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            cv_cfg = BaselineConfig(**cfg_dict)
            cv_cfg.mode = "cv"
            cv_cfg.nfolds = int(args.nfolds)
            cv_cfg.epochs = int(args.cv_epochs if args.cv_epochs is not None else max(cv_cfg.epochs + 2, cv_cfg.epochs))
            if args.cv_dev_oversample_factor is not None:
                cv_cfg.dev_oversample_factor = float(args.cv_dev_oversample_factor)
            cv_cfg.save_dir = trial_save_root
            cv_cfg.run_name = f"{cv_cfg.run_name}_cvrefine_r{rank_idx}"
            cv_cfg.seed = int(cv_cfg.seed) + 10000 + rank_idx
            t0 = time.time()
            try:
                run_dir, used_cfg = run_trial_with_oom_fallback(cv_cfg, fallback_batches=fallback_batches)
                record = summarize_trial(backbone, "cv", run_dir, used_cfg, rank_idx, {
                    "parent_holdout_run": getattr(row, "run_name"),
                    "batch_size": used_cfg.batch_size,
                    "img_size": used_cfg.img_size,
                    "learning_rate": used_cfg.learning_rate,
                    "backbone_lr": used_cfg.backbone_lr,
                    "weight_decay": used_cfg.weight_decay,
                    "dropout": used_cfg.dropout,
                    "label_smoothing": used_cfg.label_smoothing,
                    "mixup_alpha": used_cfg.mixup_alpha,
                    "mixup_prob": used_cfg.mixup_prob,
                    "aug_profile": used_cfg.aug_profile,
                    "freeze_backbone_epochs": used_cfg.freeze_backbone_epochs,
                    "dev_oversample_factor": used_cfg.dev_oversample_factor,
                    "class_balance": used_cfg.class_balance,
                })
                record["elapsed_seconds"] = time.time() - t0
                record["parent_holdout_run"] = getattr(row, "run_name")
                cv_refine_records.append(record)
                print(f"[CV refine {rank_idx}] objective={record['objective']:.6f}")
            except Exception as e:  # noqa: BLE001
                cv_refine_records.append({
                    "trial_idx": rank_idx,
                    "status": "failed",
                    "backbone": backbone,
                    "mode": "cv",
                    "parent_holdout_run": getattr(row, "run_name"),
                    "run_name": cv_cfg.run_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "elapsed_seconds": time.time() - t0,
                })
                print(f"[CV refine {rank_idx}] FAILED: {type(e).__name__}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if cv_refine_records:
            cv_refine_df = pd.DataFrame(cv_refine_records)
            if "status" in cv_refine_df.columns:
                ok = cv_refine_df[cv_refine_df["status"] == "ok"].copy()
                bad = cv_refine_df[cv_refine_df["status"] != "ok"].copy()
                if len(ok) > 0:
                    ok = ok.sort_values(["objective", "secondary", "oof_accuracy"], ascending=[True, True, False])
                cv_refine_df = pd.concat([ok, bad], axis=0, ignore_index=True)
            cv_refine_path = os.path.join(search_dir, "cv_refine_ranking.csv")
            cv_refine_df.to_csv(cv_refine_path, index=False)
            ok = cv_refine_df[cv_refine_df["status"] == "ok"] if "status" in cv_refine_df.columns else pd.DataFrame()
            if len(ok) > 0:
                save_json(os.path.join(search_dir, "best_config_cv_refine.json"), ok.iloc[0].to_dict())

    save_json(
        os.path.join(search_dir, "search_summary.json"),
        {
            "backbone": backbone,
            "mode": args.mode,
            "search_method": args.search_method,
            "search_dir": search_dir,
            "history_csv": os.path.join(search_dir, "search_history.csv"),
            "ranking_csv": os.path.join(search_dir, "search_ranking.csv"),
            "cv_refine_csv": os.path.join(search_dir, "cv_refine_ranking.csv") if os.path.exists(os.path.join(search_dir, "cv_refine_ranking.csv")) else None,
            "budget": budget.state(),
            "n_trials_ok": int(sum(1 for x in history if x.get("status") == "ok")),
            "n_trials_failed": int(sum(1 for x in history if x.get("status") != "ok")),
        },
    )

    print("\n" + "=" * 90)
    print("[Search Done]")
    print(f"search_dir: {search_dir}")
    print(f"budget used: {budget.elapsed_hours:.2f}h / {args.time_budget_hours:.2f}h")
    if os.path.exists(os.path.join(search_dir, "search_ranking.csv")):
        df = pd.read_csv(os.path.join(search_dir, "search_ranking.csv"))
        ok = df[df["status"] == "ok"]
        if len(ok) > 0:
            print("[Top trials]")
            cols = [c for c in [
                "trial_idx", "run_name", "objective", "dev_logloss_cal", "valid_logloss_cal",
                "oof_logloss_cal", "oof_accuracy", "img_size", "batch_size", "learning_rate",
                "backbone_lr", "weight_decay", "dropout", "label_smoothing", "mixup_alpha",
                "mixup_prob", "aug_profile", "dev_oversample_factor", "class_balance"
            ] if c in ok.columns]
            print(ok[cols].head(10).to_string(index=False))
    return search_dir


# ============================================================
# CLI (standalone for EfficientNet-V2-S)
# ============================================================
def build_fixed_backbone_parser() -> argparse.ArgumentParser:
    """
    이 파일은 EfficientNet-V2-S 전용 탐색기입니다.
    따라서 backbone 인자는 따로 받지 않고, 내부에서 TARGET_BACKBONE을 고정 사용합니다.
    """
    parser = argparse.ArgumentParser(description="Standalone hyperparameter search for EfficientNet-V2-S")
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--search_method", choices=["random", "grid"], default="random")
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--max_grid_trials", type=int, default=24)
    parser.add_argument("--refine_top_k_cv", type=int, default=0)
    parser.add_argument("--trial_epochs", type=int, default=10)
    parser.add_argument("--cv_epochs", type=int, default=None)
    parser.add_argument("--nfolds", type=int, default=5)

    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_backbone_search")
    parser.add_argument("--resume_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")
    parser.add_argument("--pin_memory", action="store_true", default=True)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--tta_hflip", action="store_true", default=True)
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")
    parser.add_argument("--check_paths", action="store_true", default=True)
    parser.add_argument("--no_check_paths", action="store_false", dest="check_paths")

    parser.add_argument("--time_budget_hours", type=float, default=6.0)
    parser.add_argument("--min_remaining_minutes_to_start", type=float, default=20.0)
    parser.add_argument("--oom_batch_fallback", type=str, default="12,8,4")
    parser.add_argument("--cv_dev_oversample_factor", type=float, default=None)
    return parser


def main() -> None:
    # [Flow] 스크립트 진입점
    # 1) EfficientNet-V2-S 전용 parser 생성
    # 2) backbone 이름을 TARGET_BACKBONE으로 고정
    # 3) random/grid search 실행
    parser = build_fixed_backbone_parser()
    args = parser.parse_args()
    args.backbone = TARGET_BACKBONE
    search_dir = run_search(args, default_backbone=TARGET_BACKBONE)
    print(f"\nStandalone search finished for {TARGET_BACKBONE_TITLE}")
    print(f"Search dir: {search_dir}")


if __name__ == "__main__":
    main()
