"""
EfficientNet-V2-S 전용 학습 스크립트
- 사용자가 준 standalone baseline 코드를 바탕으로 정리한 버전입니다.
- 하이퍼파라미터는 추천값으로 고정하고, 전처리 / 증강 레버만 CLI에서 켜고 끌 수 있게 만들었습니다.
- 각 섹션마다 "무슨 역할을 하는지" 주석을 붙였습니다.

핵심 설계 원칙
1) backbone / optimizer / scheduler / epochs 등 학습 하이퍼파라미터는 고정
2) 전처리 / 증강 레버만 one-by-one ablation 가능
3) 기존 baseline의 holdout / cv / calibration / submission 흐름은 유지
4) base architecture(EfficientNet-V2-S + 2-view fusion)는 유지

예시 실행
# 추천 기본값으로 holdout 실행
python train_efficientnet_v2_s_aug_levers.py --data_root ./open --mode holdout

# checkerboard roll 보정만 끄고, crop을 wide로 변경
python train_efficientnet_v2_s_aug_levers.py --data_root ./open --mode holdout \
    --no_pp_grid_roll_align --pp_crop_mode wide

# lighting 관련 증강만 꺼서 ablation
python train_efficientnet_v2_s_aug_levers.py --data_root ./open --mode holdout \
    --no_aug_color_jitter --no_aug_shadow_gradient --no_aug_jpeg

# center crop은 유지하고, background deemphasis / shadow suppression만 켜기
python train_efficientnet_v2_s_aug_levers.py --data_root ./open --mode holdout \
    --pp_background_deemphasis --pp_shadow_suppress
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

# torchvision 일부 환경에서 fake NMS registration 오류를 피하기 위한 안전장치
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

# ------------------------------------------------------------
# Optional dependency
# - checkerboard 기반 roll 정규화는 OpenCV가 있을 때 더 안정적으로 동작합니다.
# - OpenCV가 없으면 해당 전처리는 자동으로 skip 됩니다.
# ------------------------------------------------------------
try:
    import cv2

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


# ------------------------------------------------------------
# Constants
# - ImageNet normalization은 pretrained EfficientNet 계열에서 일반적으로 사용하는 값입니다.
# - 대회 label을 float으로 바꾸기 위한 매핑도 같이 둡니다.
# ------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TO_FLOAT = {"stable": 0.0, "unstable": 1.0}
TARGET_BACKBONE = "efficientnet_v2_s"


# ------------------------------------------------------------
# Utilities
# - 파일 저장, seed 고정, 시간 문자열, AMP helper 등 공통 유틸리티입니다.
# - 원본 baseline의 안정적인 부분은 최대한 유지했습니다.
# ------------------------------------------------------------
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


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


# ------------------------------------------------------------
# Dataset loading
# - 대회 기본 폴더 구조(train/dev/test/id/front.png, top.png)를 그대로 읽습니다.
# - 원본 baseline의 입출력 스키마를 최대한 그대로 유지했습니다.
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Image helper functions
# - PIL <-> numpy 변환, 블러, fill color 추정 같은 low-level helper입니다.
# - 전처리 helper들이 재사용할 수 있게 따로 뽑았습니다.
# ------------------------------------------------------------
def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def estimate_fill_color(img: Image.Image, patch: int = 16) -> Tuple[int, int, int]:
    arr = pil_to_np(img)
    h, w = arr.shape[:2]
    p = max(4, min(patch, h // 6, w // 6))
    corners = np.concatenate(
        [
            arr[:p, :p].reshape(-1, 3),
            arr[:p, -p:].reshape(-1, 3),
            arr[-p:, :p].reshape(-1, 3),
            arr[-p:, -p:].reshape(-1, 3),
        ],
        axis=0,
    )
    color = np.median(corners, axis=0)
    return tuple(int(x) for x in color.tolist())


def blur_gray_map(x: np.ndarray, radius: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if radius <= 0:
        return x
    if CV2_AVAILABLE:
        k = int(max(3, round(radius * 4) * 2 + 1))
        return cv2.GaussianBlur(x, (k, k), sigmaX=radius)
    pil = Image.fromarray(np.uint8(np.clip(x, 0, 1) * 255), mode="L")
    out = pil.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return np.asarray(out, dtype=np.float32) / 255.0


def blur_rgb_array(x: np.ndarray, radius: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if radius <= 0:
        return x
    if CV2_AVAILABLE:
        k = int(max(3, round(radius * 4) * 2 + 1))
        return cv2.GaussianBlur(x, (k, k), sigmaX=radius)
    pil = np_to_pil(np.clip(x, 0, 1) * 255.0)
    out = pil.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    return pil_to_np(out).astype(np.float32) / 255.0


# ------------------------------------------------------------
# Objectness score map
# - 정교한 segmentation 대신, “구조물일 가능성이 높은 위치”를 soft score로 추정합니다.
# - colorful / edgy / center-biased 영역에 높은 점수를 주는 휴리스틱입니다.
# - center crop, background deemphasis, shadow suppression에서 공통으로 사용합니다.
# ------------------------------------------------------------
def compute_object_score_map(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    sat = hsv[..., 1]

    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    edge = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    smooth_gray = blur_gray_map(gray, radius=max(min(gray.shape) / 48.0, 2.0))
    contrast = np.abs(gray - smooth_gray)

    score = 0.55 * sat + 0.25 * edge + 0.20 * contrast
    score = blur_gray_map(score, radius=max(min(gray.shape) / 40.0, 3.0))

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    sigma_y = max(h * 0.28, 1.0)
    sigma_x = max(w * 0.28, 1.0)
    center_prior = np.exp(-(((yy - cy) ** 2) / (2.0 * sigma_y**2) + ((xx - cx) ** 2) / (2.0 * sigma_x**2)))

    score = 0.75 * norm01(score) + 0.25 * center_prior.astype(np.float32)
    return norm01(score)


# ------------------------------------------------------------
# Preprocessing helpers
# - 아래 함수들은 “추천했던 전처리”를 실제 레버로 분해한 버전입니다.
# - 각 레버는 PairTrainTransform / PairEvalTransform에서 on/off 됩니다.
# ------------------------------------------------------------
def approximate_checkerboard_roll_align(img: Image.Image, max_abs_deg: float = 12.0) -> Image.Image:
    """
    checkerboard 전체 homography 정규화 대신,
    가장 안전한 1차 정규화인 camera roll 보정만 수행합니다.

    동작 방식
    1) 긴 선분을 Hough로 찾음
    2) 각도를 90도 주기로 접어서 [-45, 45] 범위로 변환
    3) median/trimmed mean에 가까운 dominant roll을 추정
    4) 그 각도만큼 회전

    주의
    - OpenCV가 없으면 그대로 반환합니다.
    - 이 함수는 “정확한 viewpoint rectification”이 아니라 “작은 roll 보정”입니다.
    """
    if not CV2_AVAILABLE:
        return img

    arr = pil_to_np(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    h, w = gray.shape
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=max(int(min(h, w) * 0.15), 20),
        maxLineGap=max(int(min(h, w) * 0.03), 5),
    )
    if lines is None or len(lines) < 4:
        return img

    mapped_angles = []
    lengths = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line.tolist()]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < min(h, w) * 0.12:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = ((angle + 90.0) % 180.0) - 90.0
        mapped = ((angle + 45.0) % 90.0) - 45.0
        mapped_angles.append(mapped)
        lengths.append(length)

    if len(mapped_angles) < 4:
        return img

    mapped_angles = np.asarray(mapped_angles, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    q_lo, q_hi = np.quantile(mapped_angles, [0.15, 0.85])
    keep = (mapped_angles >= q_lo) & (mapped_angles <= q_hi)
    if keep.sum() >= 3:
        mapped_angles = mapped_angles[keep]
        lengths = lengths[keep]

    roll_deg = float(np.average(mapped_angles, weights=lengths))
    if abs(roll_deg) < 0.5 or abs(roll_deg) > max_abs_deg:
        return img

    fill = estimate_fill_color(img)
    return TF.rotate(
        img,
        angle=-roll_deg,
        interpolation=InterpolationMode.BILINEAR,
        fill=fill,
    )


def find_object_center_and_extent(score: np.ndarray) -> Tuple[int, int, int, int]:
    score = norm01(score)
    h, w = score.shape

    thr = max(float(np.quantile(score, 0.985)), float(score.mean() + score.std()))
    ys, xs = np.where(score >= thr)

    if len(xs) >= 16:
        weights = score[ys, xs] + 1e-6
        cx = int(np.average(xs, weights=weights))
        cy = int(np.average(ys, weights=weights))
        box_w = int(xs.max() - xs.min() + 1)
        box_h = int(ys.max() - ys.min() + 1)
    else:
        peak_idx = np.argmax(score)
        cy, cx = np.unravel_index(peak_idx, score.shape)
        box_w = max(int(w * 0.20), 16)
        box_h = max(int(h * 0.20), 16)

    return int(cx), int(cy), int(box_w), int(box_h)


def object_center_square_crop(img: Image.Image, crop_mode: str = "tight") -> Image.Image:
    """
    구조물이 대체로 중앙 부근에 있다는 점을 이용해 square crop을 수행합니다.

    crop_mode
    - tight: 구조물 위주로 더 타이트하게 자름
    - wide : 그림자 / 주변 맥락을 조금 더 살림
    """
    score = compute_object_score_map(img)
    cx, cy, box_w, box_h = find_object_center_and_extent(score)

    w, h = img.size
    min_side = min(w, h)
    base_ratio = 0.60 if crop_mode == "tight" else 0.78
    box_scale = 2.7 if crop_mode == "tight" else 3.4

    side_from_ratio = base_ratio * min_side
    side_from_box = max(box_w, box_h) * box_scale
    side = int(np.clip(max(side_from_ratio, side_from_box), min_side * 0.38, min_side * 0.95))

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    return img.crop((x1, y1, x2, y2))


def soften_background_with_object_prior(img: Image.Image) -> Image.Image:
    """
    checkerboard background를 hard-mask로 잘라내지 않고,
    object score를 이용해 바깥쪽 배경만 softly 약화합니다.

    장점
    - segmentation 실패 시에도 덜 공격적으로 동작
    - 바닥 패턴 / 넓은 빈 배경의 영향을 줄이기 쉬움
    """
    score = compute_object_score_map(img)
    score = norm01(score)
    h, w = score.shape

    q70 = float(np.quantile(score, 0.70))
    q98 = float(np.quantile(score, 0.98))
    denom = max(q98 - q70, 1e-6)
    mask = np.clip((score - q70) / denom, 0.0, 1.0)
    mask = blur_gray_map(mask, radius=max(min(h, w) / 22.0, 4.0))
    mask = np.clip(mask * 1.35, 0.0, 1.0)

    arr = pil_to_np(img).astype(np.float32) / 255.0
    bg_blur = blur_rgb_array(arr, radius=max(min(h, w) / 18.0, 3.0))
    bg_mean = arr.mean(axis=(0, 1), keepdims=True)
    bg = 0.55 * bg_blur + 0.45 * bg_mean

    keep = 0.35 + 0.65 * mask
    out = arr * keep[..., None] + bg * (1.0 - keep[..., None])
    return np_to_pil(out * 255.0)


def suppress_floor_shadows(img: Image.Image) -> Image.Image:
    """
    cast shadow를 완전히 지우지 않고, low-saturation / dark floor 영역만 살짝 들어올립니다.
    즉, 조명 정규화라기보다 “shadow 약화”에 가깝습니다.

    이 레버는 aggressive하게 쓰면 유용한 그림자 힌트까지 깎을 수 있으므로
    기본 추천값은 OFF로 두었습니다.
    """
    arr = pil_to_np(img).astype(np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    sat = hsv[..., 1]
    val = hsv[..., 2]
    score = compute_object_score_map(img)

    h, w = val.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    floor_region = yy >= h * 0.18
    shadow_mask = floor_region & (sat < 0.35) & (val < 0.75) & (score < 0.30)

    if not np.any(shadow_mask):
        return img

    target = np.clip(arr * 1.18 + 0.06, 0.0, 1.0)
    alpha = np.zeros_like(val, dtype=np.float32)
    alpha[shadow_mask] = 0.55
    alpha = blur_gray_map(alpha, radius=max(min(h, w) / 60.0, 2.0))

    out = arr * (1.0 - alpha[..., None]) + target * alpha[..., None]
    return np_to_pil(out * 255.0)


# ------------------------------------------------------------
# Augmentation helpers
# - 아래 함수들은 학습 시에만 사용하는 증강입니다.
# - front / top 두 뷰에는 가능한 한 같은 기하 / 같은 조명 변형을 적용합니다.
# ------------------------------------------------------------
def apply_same_color_jitter(img: Image.Image, brightness: float, contrast: float, saturation: float, hue: float) -> Image.Image:
    img = TF.adjust_brightness(img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    img = TF.adjust_hue(img, hue)
    return img


def jpeg_like(img: Image.Image, quality_min: int = 45, quality_max: int = 85) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(random.randint(quality_min, quality_max)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_lighting_gradient(img: Image.Image, strength: float) -> Image.Image:
    """
    광원 방향 변화 / 조도 기복을 흉내 내는 매우 가벼운 gradient illumination입니다.
    색상을 뒤집지 않고 전체 조도만 부드럽게 바꿉니다.
    """
    arr = pil_to_np(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = random.uniform(0.25 * w, 0.75 * w)
    cy = random.uniform(0.25 * h, 0.75 * h)
    dx = random.uniform(-1.0, 1.0)
    dy = random.uniform(-1.0, 1.0)
    norm = max(math.hypot(dx, dy), 1e-6)
    dx /= norm
    dy /= norm

    plane = dx * ((xx - cx) / max(w / 2.0, 1.0)) + dy * ((yy - cy) / max(h / 2.0, 1.0))
    plane = plane / max(np.max(np.abs(plane)), 1e-6)
    mult = 1.0 + float(strength) * plane
    mult = np.clip(mult, 0.70, 1.30)

    out = np.clip(arr * mult[..., None], 0.0, 1.0)
    return np_to_pil(out * 255.0)


def apply_glare_spot(img: Image.Image, amplitude: float, sigma_ratio: float) -> Image.Image:
    """
    sample 이미지의 specular-like glare를 약하게 흉내 내는 가우시안 bright spot입니다.
    너무 강하면 라벨 신호를 망가뜨릴 수 있어 기본값은 OFF입니다.
    """
    arr = pil_to_np(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0 = random.uniform(0.12 * w, 0.88 * w)
    y0 = random.uniform(0.12 * h, 0.88 * h)
    sigma = max(min(h, w) * sigma_ratio, 3.0)
    spot = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))

    out = np.clip(arr + amplitude * spot[..., None], 0.0, 1.0)
    return np_to_pil(out * 255.0)


# ------------------------------------------------------------
# Fixed config
# - 하이퍼파라미터는 “추천값”으로 고정합니다.
# - 사용자는 전처리 / 증강 레버만 만질 수 있고,
#   backbone / lr / wd / dropout / epochs 등은 여기에서 통일합니다.
# ------------------------------------------------------------
@dataclass
class FixedConfig:
    run_name: str
    mode: str = "holdout"  # holdout or cv
    data_root: str = "./open"
    save_dir: str = "./runs_aug_levers"

    # ---------- 고정 학습 하이퍼파라미터 ----------
    backbone: str = TARGET_BACKBONE
    pretrained: bool = True
    img_size: int = 384
    batch_size: int = 12
    epochs: int = 12
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
    freeze_backbone_epochs: int = 0

    # ---------- 고정 후처리 / 추론 ----------
    use_amp: bool = True
    pin_memory: bool = True
    temperature_scaling: bool = True
    check_paths: bool = True
    save_hard_examples_topk: int = 50

    # ---------- 전처리 레버 ----------
    pp_grid_roll_align: bool = True
    pp_center_crop: bool = True
    pp_crop_mode: str = "tight"  # tight or wide
    pp_background_deemphasis: bool = False
    pp_shadow_suppress: bool = False

    # ---------- 학습 증강 레버 ----------
    aug_hflip: bool = True
    aug_affine: bool = True
    aug_color_jitter: bool = True
    aug_shadow_gradient: bool = True
    aug_glare_spot: bool = False
    aug_autocontrast: bool = False
    aug_grayscale: bool = False
    aug_blur: bool = True
    aug_jpeg: bool = True
    aug_random_erasing: bool = False
    aug_mixup: bool = False

    # ---------- 고정 증강 강도 ----------
    mixup_alpha_fixed: float = 0.10
    mixup_prob_fixed: float = 0.25
    tta_hflip: bool = True

    def short(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------
# Transform classes
# - evaluation에서는 전처리만 수행합니다.
# - training에서는 전처리 + pair-wise augmentation을 수행합니다.
# - geometry/color 계열은 가능한 한 front / top에 동일 파라미터를 적용합니다.
# ------------------------------------------------------------
class PairTrainTransform:
    def __init__(self, cfg: FixedConfig):
        self.cfg = cfg
        self.img_size = int(cfg.img_size)
        self.reasing = torchvision.transforms.RandomErasing(
            p=1.0,
            scale=(0.02, 0.08),
            ratio=(0.4, 2.5),
            value="random",
        )

    def _resize(self, img: Image.Image) -> Image.Image:
        return TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)

    def _preprocess_single(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")

        if self.cfg.pp_grid_roll_align:
            img = approximate_checkerboard_roll_align(img)

        if self.cfg.pp_shadow_suppress:
            img = suppress_floor_shadows(img)

        if self.cfg.pp_center_crop:
            img = object_center_square_crop(img, crop_mode=self.cfg.pp_crop_mode)

        if self.cfg.pp_background_deemphasis:
            img = soften_background_with_object_prior(img)

        img = self._resize(img)
        return img

    def _apply_pairwise_geometry(self, front: Image.Image, top: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.cfg.aug_hflip and random.random() < 0.5:
            front = TF.hflip(front)
            top = TF.hflip(top)

        if self.cfg.aug_affine:
            angle = random.uniform(-6.0, 6.0)
            max_shift = int(self.img_size * 0.04)
            translate = (
                random.randint(-max_shift, max_shift),
                random.randint(-max_shift, max_shift),
            )
            scale = random.uniform(0.96, 1.04)

            front = TF.affine(
                front,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=estimate_fill_color(front),
            )
            top = TF.affine(
                top,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=estimate_fill_color(top),
            )

        return front, top

    def _apply_pairwise_photometric(self, front: Image.Image, top: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.cfg.aug_color_jitter:
            brightness = 1.0 + random.uniform(-0.12, 0.12)
            contrast = 1.0 + random.uniform(-0.12, 0.12)
            saturation = 1.0 + random.uniform(-0.10, 0.10)
            hue = random.uniform(-0.02, 0.02)
            front = apply_same_color_jitter(front, brightness, contrast, saturation, hue)
            top = apply_same_color_jitter(top, brightness, contrast, saturation, hue)

        if self.cfg.aug_shadow_gradient and random.random() < 0.35:
            strength = random.uniform(0.08, 0.16)
            if random.random() < 0.5:
                strength = -strength
            front = apply_lighting_gradient(front, strength)
            top = apply_lighting_gradient(top, strength)

        if self.cfg.aug_glare_spot and random.random() < 0.12:
            amplitude = random.uniform(0.08, 0.18)
            sigma_ratio = random.uniform(0.05, 0.12)
            front = apply_glare_spot(front, amplitude=amplitude, sigma_ratio=sigma_ratio)
            top = apply_glare_spot(top, amplitude=amplitude, sigma_ratio=sigma_ratio)

        if self.cfg.aug_autocontrast and random.random() < 0.10:
            front = ImageOps.autocontrast(front)
            top = ImageOps.autocontrast(top)

        if self.cfg.aug_grayscale and random.random() < 0.04:
            front = ImageOps.grayscale(front).convert("RGB")
            top = ImageOps.grayscale(top).convert("RGB")

        if self.cfg.aug_blur and random.random() < 0.12:
            radius = random.uniform(0.10, 0.70)
            front = front.filter(ImageFilter.GaussianBlur(radius=radius))
            top = top.filter(ImageFilter.GaussianBlur(radius=radius))

        if self.cfg.aug_jpeg and random.random() < 0.12:
            front = jpeg_like(front)
            top = jpeg_like(top)

        return front, top

    def _post_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.aug_random_erasing and random.random() < 0.15:
            x = self.reasing(x)
        return x

    def __call__(self, front: Image.Image, top: Image.Image):
        front = self._preprocess_single(front)
        top = self._preprocess_single(top)

        front, top = self._apply_pairwise_geometry(front, top)
        front, top = self._apply_pairwise_photometric(front, top)

        front = TF.to_tensor(front)
        top = TF.to_tensor(top)
        front = TF.normalize(front, IMAGENET_MEAN, IMAGENET_STD)
        top = TF.normalize(top, IMAGENET_MEAN, IMAGENET_STD)
        front = self._post_tensor(front)
        top = self._post_tensor(top)
        return front, top


class PairEvalTransform:
    def __init__(self, cfg: FixedConfig):
        self.cfg = cfg
        self.img_size = int(cfg.img_size)

    def _resize(self, img: Image.Image) -> Image.Image:
        return TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BICUBIC)

    def _preprocess_single(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")

        if self.cfg.pp_grid_roll_align:
            img = approximate_checkerboard_roll_align(img)

        if self.cfg.pp_shadow_suppress:
            img = suppress_floor_shadows(img)

        if self.cfg.pp_center_crop:
            img = object_center_square_crop(img, crop_mode=self.cfg.pp_crop_mode)

        if self.cfg.pp_background_deemphasis:
            img = soften_background_with_object_prior(img)

        img = self._resize(img)
        return img

    def __call__(self, front: Image.Image, top: Image.Image):
        front = self._preprocess_single(front)
        top = self._preprocess_single(top)
        front = TF.to_tensor(front)
        top = TF.to_tensor(top)
        front = TF.normalize(front, IMAGENET_MEAN, IMAGENET_STD)
        top = TF.normalize(top, IMAGENET_MEAN, IMAGENET_STD)
        return front, top


# ------------------------------------------------------------
# Dataset
# - front / top 두 장을 읽는 구조는 원본 baseline과 동일합니다.
# - transform은 PairTrainTransform / PairEvalTransform을 받습니다.
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Model
# - 두 뷰를 같은 EfficientNet-V2-S backbone으로 인코딩합니다.
# - front/top/abs-diff/elementwise-product fusion은 원본 baseline 구조를 유지했습니다.
# ------------------------------------------------------------
def build_backbone(backbone_name: str, pretrained: bool = True):
    if backbone_name != TARGET_BACKBONE:
        raise ValueError(f"Only {TARGET_BACKBONE} is supported in this script. Got: {backbone_name}")

    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    try:
        base = models.efficientnet_v2_s(weights=weights)
    except Exception as e:
        print(f"[Warning] efficientnet_v2_s pretrained load 실패 -> random init 사용: {e}")
        base = models.efficientnet_v2_s(weights=None)

    feat_dim = base.classifier[1].in_features
    return {
        "features": base.features,
        "pool": base.avgpool,
        "feat_dim": feat_dim,
    }


class MultiViewFusionNet(nn.Module):
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


# ------------------------------------------------------------
# DataLoader helpers
# - train/valid/test loader를 만드는 보조 함수입니다.
# - 이번 스크립트는 class-balance, oversample 같은 탐색용 레버는 제거했습니다.
# ------------------------------------------------------------
def build_loader(
    df: pd.DataFrame,
    transform,
    cfg: FixedConfig,
    device: torch.device,
    is_test: bool,
    shuffle: bool,
) -> DataLoader:
    ds = MultiViewDataset(df, transform=transform, is_test=is_test)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker_factory(cfg.seed),
        generator=generator if not is_test else None,
        drop_last=False,
    )


# ------------------------------------------------------------
# Loss / calibration
# - BCE with logits + label smoothing
# - validation logits에는 temperature scaling을 적용할 수 있습니다.
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Train / validate / infer
# - 원본 baseline 흐름을 유지하되,
#   mixup은 augmentation 레버로 on/off 되게 변경했습니다.
# ------------------------------------------------------------
def maybe_freeze_backbone(model: MultiViewFusionNet, freeze: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = not freeze


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


def build_optimizer_and_scheduler(model: MultiViewFusionNet, cfg: FixedConfig, steps_per_epoch: int):
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    cfg: FixedConfig,
    device: torch.device,
    use_amp: bool,
) -> float:
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
            if cfg.aug_mixup and random.random() < cfg.mixup_prob_fixed:
                lam = np.random.beta(cfg.mixup_alpha_fixed, cfg.mixup_alpha_fixed)
                idx = torch.randperm(batch_size, device=device)
                mixed_front = lam * front + (1.0 - lam) * front[idx]
                mixed_top = lam * top + (1.0 - lam) * top[idx]
                y_a, y_b = y, y[idx]
                logits = model(mixed_front, mixed_top)
                loss = lam * classification_loss(logits, y_a, cfg.label_smoothing) + (1.0 - lam) * classification_loss(
                    logits, y_b, cfg.label_smoothing
                )
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
def validate_one_epoch(model, loader, cfg: FixedConfig, device: torch.device, use_amp: bool) -> EpochOutput:
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


# ------------------------------------------------------------
# Core training helper
# - 한 번의 train/valid/test 학습 흐름을 담당합니다.
# - best checkpoint 저장, calibration, validation/test prediction 저장까지 여기서 처리합니다.
# ------------------------------------------------------------
def train_with_validation(
    cfg: FixedConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
    fold_idx: Optional[int] = None,
) -> Dict:
    seed_everything(cfg.seed + (0 if fold_idx is None else fold_idx * 1000))

    train_tf = PairTrainTransform(cfg)
    valid_tf = PairEvalTransform(cfg)

    train_loader = build_loader(train_df, train_tf, cfg, device=device, is_test=False, shuffle=True)
    valid_loader = build_loader(valid_df, valid_tf, cfg, device=device, is_test=False, shuffle=False)
    test_loader = build_loader(test_df, valid_tf, cfg, device=device, is_test=True, shuffle=False)

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

        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_bce": float(valid_out.valid_bce),
                "valid_logloss_raw": float(valid_out.valid_logloss),
                "valid_logloss_cal": float(valid_out.valid_logloss_cal),
                "valid_acc": float(valid_out.valid_acc),
                "temperature": float(valid_out.temperature),
            }
        )

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
    valid_pred_df["sample_logloss"] = per_sample_logloss(
        valid_pred_df["label_float"].values,
        valid_pred_df["unstable_prob"].values,
    )
    valid_pred_csv = os.path.join(run_dir, f"valid_predictions_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv")
    valid_pred_df.to_csv(valid_pred_csv, index=False)
    save_hard_examples(
        valid_pred_df,
        os.path.join(run_dir, f"hard_examples_{cfg.run_name}{'' if fold_idx is None else f'_fold{fold_idx}'}.csv"),
        topk=cfg.save_hard_examples_topk,
    )

    test_pred_df = pd.DataFrame(
        {
            "id": test_ids,
            "logit": test_logits,
            "unstable_prob_raw": test_probs_raw,
            "unstable_prob": test_probs_cal,
            "stable_prob": 1.0 - test_probs_cal,
            "temperature": float(best_state["temperature"]),
        }
    )
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


# ------------------------------------------------------------
# Holdout / CV
# - holdout: train으로 학습하고 dev로 검증
# - cv: train+dev 전체를 n-fold로 OOF 검증
# ------------------------------------------------------------
def run_holdout(
    cfg: FixedConfig,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
) -> Dict:
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


def run_cv(
    cfg: FixedConfig,
    full_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: str,
    device: torch.device,
    use_amp: bool,
) -> Dict:
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

    fold_metrics_df = pd.DataFrame(
        [
            {
                **{k: v for k, v in m.items() if k != "source_metrics"},
                "source_metrics": json.dumps(m["source_metrics"], ensure_ascii=False),
            }
            for m in fold_metrics
        ]
    )
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
            parts.append(
                {
                    "id": sid,
                    "unstable_prob": float(np.sum(w * p)),
                    "stable_prob": float(np.sum(w * (1.0 - p))),
                }
            )
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


# ------------------------------------------------------------
# Runner
# - 데이터 로드 -> 경로 확인 -> holdout 또는 cv 실행
# - 추천 고정 하이퍼파라미터 + 사용자가 선택한 레버 설정을 JSON으로 같이 저장합니다.
# ------------------------------------------------------------
def run_experiment(cfg: FixedConfig) -> str:
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
    print(f"OpenCV available: {CV2_AVAILABLE}")
    print(f"Run dir: {run_dir}")
    print("[Lever Summary]")
    print(
        json.dumps(
            {
                "pp_grid_roll_align": cfg.pp_grid_roll_align,
                "pp_center_crop": cfg.pp_center_crop,
                "pp_crop_mode": cfg.pp_crop_mode,
                "pp_background_deemphasis": cfg.pp_background_deemphasis,
                "pp_shadow_suppress": cfg.pp_shadow_suppress,
                "aug_hflip": cfg.aug_hflip,
                "aug_affine": cfg.aug_affine,
                "aug_color_jitter": cfg.aug_color_jitter,
                "aug_shadow_gradient": cfg.aug_shadow_gradient,
                "aug_glare_spot": cfg.aug_glare_spot,
                "aug_autocontrast": cfg.aug_autocontrast,
                "aug_grayscale": cfg.aug_grayscale,
                "aug_blur": cfg.aug_blur,
                "aug_jpeg": cfg.aug_jpeg,
                "aug_random_erasing": cfg.aug_random_erasing,
                "aug_mixup": cfg.aug_mixup,
                "tta_hflip": cfg.tta_hflip,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

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

    print(f"\nDone. Outputs saved to: {run_dir}")
    return run_dir


# ------------------------------------------------------------
# CLI
# - 사용자가 직접 조절할 수 있는 것은 “레버” 뿐입니다.
# - 하이퍼파라미터는 parser에 노출하지 않고 FixedConfig 기본값을 그대로 사용합니다.
# - 각 전처리 / 증강은 --foo / --no_foo 형태로 하나씩 ablation 가능합니다.
# ------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EfficientNet-V2-S trainer with preprocessing / augmentation levers")

    # ----- 실행 모드 / 입출력 경로 -----
    parser.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    parser.add_argument("--data_root", type=str, default="./open")
    parser.add_argument("--save_dir", type=str, default="./runs_aug_levers")
    parser.add_argument("--run_name", type=str, default="baseline_efficientnet_v2_s_aug_levers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--nfolds", type=int, default=5)

    # ----- 전처리 레버 -----
    parser.add_argument("--pp_grid_roll_align", action="store_true", default=True)
    parser.add_argument("--no_pp_grid_roll_align", action="store_false", dest="pp_grid_roll_align")

    parser.add_argument("--pp_center_crop", action="store_true", default=True)
    parser.add_argument("--no_pp_center_crop", action="store_false", dest="pp_center_crop")
    parser.add_argument("--pp_crop_mode", choices=["tight", "wide"], default="tight")

    parser.add_argument("--pp_background_deemphasis", action="store_true", default=False)
    parser.add_argument("--no_pp_background_deemphasis", action="store_false", dest="pp_background_deemphasis")

    parser.add_argument("--pp_shadow_suppress", action="store_true", default=False)
    parser.add_argument("--no_pp_shadow_suppress", action="store_false", dest="pp_shadow_suppress")

    # ----- 학습 증강 레버 -----
    parser.add_argument("--aug_hflip", action="store_true", default=True)
    parser.add_argument("--no_aug_hflip", action="store_false", dest="aug_hflip")

    parser.add_argument("--aug_affine", action="store_true", default=True)
    parser.add_argument("--no_aug_affine", action="store_false", dest="aug_affine")

    parser.add_argument("--aug_color_jitter", action="store_true", default=True)
    parser.add_argument("--no_aug_color_jitter", action="store_false", dest="aug_color_jitter")

    parser.add_argument("--aug_shadow_gradient", action="store_true", default=True)
    parser.add_argument("--no_aug_shadow_gradient", action="store_false", dest="aug_shadow_gradient")

    parser.add_argument("--aug_glare_spot", action="store_true", default=False)
    parser.add_argument("--no_aug_glare_spot", action="store_false", dest="aug_glare_spot")

    parser.add_argument("--aug_autocontrast", action="store_true", default=False)
    parser.add_argument("--no_aug_autocontrast", action="store_false", dest="aug_autocontrast")

    parser.add_argument("--aug_grayscale", action="store_true", default=False)
    parser.add_argument("--no_aug_grayscale", action="store_false", dest="aug_grayscale")

    parser.add_argument("--aug_blur", action="store_true", default=True)
    parser.add_argument("--no_aug_blur", action="store_false", dest="aug_blur")

    parser.add_argument("--aug_jpeg", action="store_true", default=True)
    parser.add_argument("--no_aug_jpeg", action="store_false", dest="aug_jpeg")

    parser.add_argument("--aug_random_erasing", action="store_true", default=False)
    parser.add_argument("--no_aug_random_erasing", action="store_false", dest="aug_random_erasing")

    parser.add_argument("--aug_mixup", action="store_true", default=False)
    parser.add_argument("--no_aug_mixup", action="store_false", dest="aug_mixup")

    # ----- 추론 증강 레버 -----
    parser.add_argument("--tta_hflip", action="store_true", default=True)
    parser.add_argument("--no_tta_hflip", action="store_false", dest="tta_hflip")

    return parser


# ------------------------------------------------------------
# Config builder
# - parser에서 받은 레버만 FixedConfig에 주입합니다.
# - 학습 하이퍼파라미터는 여기서도 바꾸지 않습니다.
# ------------------------------------------------------------
def make_config_from_args(ns: argparse.Namespace) -> FixedConfig:
    return FixedConfig(
        run_name=ns.run_name,
        mode=ns.mode,
        data_root=ns.data_root,
        save_dir=ns.save_dir,
        seed=ns.seed,
        num_workers=ns.num_workers,
        nfolds=ns.nfolds,
        pp_grid_roll_align=ns.pp_grid_roll_align,
        pp_center_crop=ns.pp_center_crop,
        pp_crop_mode=ns.pp_crop_mode,
        pp_background_deemphasis=ns.pp_background_deemphasis,
        pp_shadow_suppress=ns.pp_shadow_suppress,
        aug_hflip=ns.aug_hflip,
        aug_affine=ns.aug_affine,
        aug_color_jitter=ns.aug_color_jitter,
        aug_shadow_gradient=ns.aug_shadow_gradient,
        aug_glare_spot=ns.aug_glare_spot,
        aug_autocontrast=ns.aug_autocontrast,
        aug_grayscale=ns.aug_grayscale,
        aug_blur=ns.aug_blur,
        aug_jpeg=ns.aug_jpeg,
        aug_random_erasing=ns.aug_random_erasing,
        aug_mixup=ns.aug_mixup,
        tta_hflip=ns.tta_hflip,
    )


# ------------------------------------------------------------
# Entry point
# - parser -> config -> run_experiment 흐름입니다.
# ------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = make_config_from_args(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
