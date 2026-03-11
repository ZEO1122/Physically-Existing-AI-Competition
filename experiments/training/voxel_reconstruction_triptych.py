"""
Voxel-like 3D reconstruction prototype for the DACON structure stability dataset.

이 스크립트는 "정확한 물리 기반 3D 복원"이 아니라,
front / top 두 장의 이미지를 이용해 "정규화된 pseudo-voxel occupancy"를 만들고,
이를 3D 이미지로 렌더링하여 사람이 확인할 수 있게 저장하는 실용형 프로토타입입니다.

핵심 아이디어
1) 이미지 중심부에 있는 구조물을 GrabCut으로 분리합니다.
2) top 뷰는 footprint(바닥 점유)로, front 뷰는 x-z silhouette(가로-높이 형상)로 봅니다.
3) 두 마스크를 공통 격자로 맞춘 뒤, 간단한 space carving 방식으로 occupancy volume을 만듭니다.
4) top 뷰의 색을 column color로 써서 3D voxel 이미지를 렌더링합니다.
5) front / top / reconstructed-3D triptych를 샘플별로 저장하고, split별 contact sheet도 만듭니다.

주의
- 이 코드는 checkerboard + 고정된 voxel-like 배치를 활용한 "시각화용" 재구성입니다.
- occlusion, 내부 빈 공간, 뒤쪽 지지 구조까지 정확히 복원하지는 못합니다.
- 그래도 split별 샘플을 빠르게 점검하고, 후속 3D 특징 추출의 출발점으로 쓰기 좋습니다.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# 공통 유틸
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


# ============================================================
# 데이터 로딩
# baseline 코드의 open.zip 구조를 그대로 따릅니다.
# ============================================================



def is_competition_root(path: str) -> bool:
    """
    baseline 코드와 동일한 open 폴더 구조인지 확인합니다.
    필요 파일: train.csv, dev.csv, sample_submission.csv, train/dev/test 디렉토리
    """
    if not path:
        return False
    csv_ok = all(os.path.exists(os.path.join(path, name)) for name in ["train.csv", "dev.csv", "sample_submission.csv"])
    dir_ok = all(os.path.isdir(os.path.join(path, name)) for name in ["train", "dev", "test"])
    return bool(csv_ok and dir_ok)


def resolve_data_root(data_root: str) -> str:
    """
    실행 위치가 달라도 baseline 코드에서 쓰는 open 폴더를 최대한 자동으로 찾습니다.

    예시
    - 현재 작업 폴더가 프로젝트 루트일 때: ./open
    - 현재 작업 폴더가 model_file/baseline_model 일 때: ../../open
    - 스크립트 파일 기준 상대 경로로 실행할 때도 탐색 가능
    """
    raw = str(data_root).strip()
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []

    def add(p: str) -> None:
        if p and p not in candidates:
            candidates.append(os.path.abspath(p))

    # 사용자가 준 경로 우선
    add(raw)
    add(os.path.join(cwd, raw))
    add(os.path.join(script_dir, raw))

    # baseline_model 같은 하위 폴더에서 실행하는 경우까지 고려
    for base in [cwd, script_dir]:
        add(os.path.join(base, 'open'))
        add(os.path.join(base, '..', 'open'))
        add(os.path.join(base, '..', '..', 'open'))
        add(os.path.join(base, '..', '..', '..', 'open'))
        add(os.path.join(base, raw, 'open'))

    for cand in candidates:
        if is_competition_root(cand):
            return cand

    preview = "\n".join(candidates[:12])
    raise FileNotFoundError(
        'open 데이터 루트를 찾지 못했습니다.\n'
        f'입력값: {data_root}\n'
        '아래 후보들을 확인했지만 baseline 구조(train.csv/dev.csv/sample_submission.csv + train/dev/test)가 없었습니다.\n'
        f'{preview}'
    )


def load_split_df(root_dir: str, split: str) -> pd.DataFrame:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")

    csv_name = "sample_submission.csv" if split == "test" else f"{split}.csv"
    csv_path = os.path.join(root_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"id": str})
    df["id"] = df["id"].astype(str)
    df["source"] = split

    split_dir = os.path.join(root_dir, split)
    df["folder"] = df["id"].map(lambda x: os.path.join(split_dir, x))
    df["front_path"] = df["folder"].map(lambda x: os.path.join(x, "front.png"))
    df["top_path"] = df["folder"].map(lambda x: os.path.join(x, "top.png"))
    if "label" not in df.columns:
        df["label"] = "unknown"
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


# ============================================================
# 재구성 설정
# ============================================================
@dataclass
class ReconConfig:
    grid_xy: int = 10
    grid_z: int = 14
    num_per_split: int = 10
    seed: int = 42
    save_masks: bool = True
    use_top_convex_hull: bool = True
    top_shadow_saturation_quantile: float = 35.0
    top_shadow_saturation_floor: float = 12.0
    grabcut_iters: int = 5
    center_rect_frac: Tuple[float, float, float, float] = (0.18, 0.18, 0.64, 0.64)
    crop_pad_frac: float = 0.14
    panel_size: int = 384
    dpi: int = 150
    view_elev: float = 24.0
    view_azim: float = -55.0


# ============================================================
# 1) 전경 분리
# - 기본은 center-biased GrabCut
# - top view는 shadow 억제를 위해 saturation 기반 후처리를 한 번 더 수행
# ============================================================

def center_rect(h: int, w: int, rect_frac: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    rx, ry, rw, rh = rect_frac
    return (int(w * rx), int(h * ry), int(w * rw), int(h * rh))


def keep_best_component(mask: np.ndarray) -> np.ndarray:
    """
    여러 연결요소가 있을 때 이미지 중심에 가깝고 면적이 큰 전경을 남깁니다.
    checkerboard 그림자/광원 flare 같은 잡성분을 줄이기 위한 간단한 휴리스틱입니다.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num <= 1:
        return mask_u8.astype(bool)

    h, w = mask_u8.shape
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best_idx, best_score = 1, -1e18

    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        c = cents[i]
        dist = float(np.linalg.norm(c - center))
        score = area - 25.0 * dist
        if score > best_score:
            best_idx, best_score = i, score

    return (labels == best_idx)


def fallback_saliency_mask(rgb: np.ndarray) -> np.ndarray:
    """
    GrabCut이 잘 안 먹는 환경을 위한 약한 fallback입니다.
    배경 checkerboard보다 구조물이 대체로 더 colorful + edge-dense 하다는 점을 이용합니다.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    sat = hsv[:, :, 1].astype(np.float32)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.0)
    sat = cv2.GaussianBlur(sat, (0, 0), 1.0)

    sat_thr = np.percentile(sat, 80)
    grad_thr = np.percentile(grad, 85)
    mask = ((sat >= sat_thr) | (grad >= grad_thr)).astype(np.uint8) * 255

    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * 0.32, w * 0.32
    center_prior = (((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2) <= 1.0
    mask &= center_prior.astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return keep_best_component(mask > 0)


def grabcut_center_mask(img: Image.Image, cfg: ReconConfig) -> np.ndarray:
    rgb = pil_to_np(img)
    h, w = rgb.shape[:2]
    rect = center_rect(h, w, cfg.center_rect_frac)

    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd, fgd, cfg.grabcut_iters, cv2.GC_INIT_WITH_RECT)
        m = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
        m = cv2.morphologyEx(m * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m = keep_best_component(m > 0)

        area_ratio = float(m.mean())
        if area_ratio < 0.0005 or area_ratio > 0.45:
            return fallback_saliency_mask(rgb)
        return m
    except Exception:
        return fallback_saliency_mask(rgb)



def maybe_stabilize_top_mask_with_hull(mask: np.ndarray, use_hull: bool) -> np.ndarray:
    """
    top footprint가 거의 convex한 경우에는 convex hull로 작은 notch를 메워
    shadow/segmentation artifact에 덜 민감하게 만듭니다.
    """
    if not use_hull:
        return mask

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / max(hull_area, 1.0)

    # footprint가 이미 거의 convex할 때만 hull을 씁니다.
    if solidity >= 0.90:
        out = np.zeros_like(mask_u8)
        cv2.drawContours(out, [hull], -1, 1, thickness=-1)
        return out.astype(bool)
    return mask



def segment_top_mask(img: Image.Image, cfg: ReconConfig) -> np.ndarray:
    """
    top view는 shadow가 footprint에 붙기 쉬워서,
    GrabCut 이후 low-saturation 영역을 한 번 더 깎아냅니다.
    """
    rgb = pil_to_np(img)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    m = grabcut_center_mask(img, cfg)
    sat = hsv[:, :, 1].astype(np.float32)
    masked_sat = sat[m]
    if masked_sat.size > 0:
        sat_thr = max(cfg.top_shadow_saturation_floor, float(np.percentile(masked_sat, cfg.top_shadow_saturation_quantile)))
        filtered = m & (sat >= sat_thr)
        filtered = cv2.morphologyEx((filtered.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        filtered = keep_best_component(filtered > 0)
        if filtered.sum() > max(25, int(m.sum() * 0.35)):
            m = filtered

    m = maybe_stabilize_top_mask_with_hull(m, cfg.use_top_convex_hull)
    return m



def segment_front_mask(img: Image.Image, cfg: ReconConfig) -> np.ndarray:
    return grabcut_center_mask(img, cfg)


# ============================================================
# 2) 뷰 정규화
# - top: minAreaRect 기반 회전 정렬 후 crop
# - front: tight crop
# ============================================================

def tight_bbox(mask: np.ndarray, pad_frac: float = 0.12) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w = mask.shape[1]
    h = mask.shape[0]
    pw = int((x1 - x0) * pad_frac)
    ph = int((y1 - y0) * pad_frac)
    x0 = max(0, x0 - pw)
    y0 = max(0, y0 - ph)
    x1 = min(w, x1 + pw)
    y1 = min(h, y1 + ph)
    return x0, y0, x1, y1



def crop_by_bbox(mask: np.ndarray, rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    return mask[y0:y1, x0:x1], rgb[y0:y1, x0:x1]



def rotate_align_top(mask: np.ndarray, rgb: np.ndarray, cfg: ReconConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    top footprint를 minAreaRect로 잡고, 긴/짧은 축이 이미지 축과 나란해지도록 회전합니다.
    완전한 checkerboard homography는 아니지만, footprint를 공통 격자에 태우기 쉽게 만듭니다.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask, rgb, 0.0

    pts = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (rw, rh), angle = rect

    if rw < rh:
        rot_angle = angle
    else:
        rot_angle = angle + 90.0

    h, w = mask.shape
    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    rot_mask = cv2.warpAffine((mask.astype(np.uint8) * 255), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    rot_rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    bbox = tight_bbox(rot_mask, pad_frac=cfg.crop_pad_frac)
    crop_mask, crop_rgb = crop_by_bbox(rot_mask, rot_rgb, bbox)
    return crop_mask, crop_rgb, float(rot_angle)



def crop_front(mask: np.ndarray, rgb: np.ndarray, cfg: ReconConfig) -> Tuple[np.ndarray, np.ndarray]:
    bbox = tight_bbox(mask, pad_frac=cfg.crop_pad_frac)
    return crop_by_bbox(mask, rgb, bbox)


# ============================================================
# 3) 공통 격자 변환 + occupancy fusion
# - top -> x-y footprint
# - front -> x-z silhouette
# - 두 projection을 합쳐 pseudo-voxel volume 생성
# ============================================================

def resize_mask(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    m = cv2.resize((mask.astype(np.uint8) * 255), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return (m > 127)



def resize_rgb(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)



def normalize_top_to_grid(mask: np.ndarray, rgb: np.ndarray, cfg: ReconConfig) -> Tuple[np.ndarray, np.ndarray]:
    top_xy = resize_mask(mask, cfg.grid_xy, cfg.grid_xy)
    top_rgb = resize_rgb(rgb, cfg.grid_xy, cfg.grid_xy)
    return top_xy, top_rgb



def normalize_front_to_grid(mask: np.ndarray, cfg: ReconConfig) -> np.ndarray:
    """
    front silhouette는 x-z plane으로 보며,
    row 0이 top / 마지막 row가 바닥인 이미지를 받습니다.
    내부에 작은 segmentation hole이 있더라도 열 단위로 아래 방향 fill을 수행해
    '실제 column'에 가까운 형태로 만듭니다.
    """
    front_xz = resize_mask(mask, cfg.grid_xy, cfg.grid_z)
    filled = np.zeros_like(front_xz, dtype=bool)
    for x in range(cfg.grid_xy):
        ys = np.where(front_xz[:, x])[0]
        if len(ys) > 0:
            y_top = int(ys.min())
            filled[y_top:, x] = True
    if filled.sum() == 0:
        filled = front_xz
    return filled



def choose_top_orientation(top_xy: np.ndarray, top_rgb: np.ndarray, front_xz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    top 뷰는 front 기준 x축이 90도 ambiguity를 가질 수 있습니다.
    그래서 top을 [0도, 90도] 두 경우로 보고,
    x projection이 front x projection과 더 잘 맞는 쪽을 선택합니다.
    좌우 반전도 같이 검사합니다.
    """
    fx = front_xz.any(axis=0).astype(np.uint8)

    candidates = []
    for k in [0, 1]:
        b = np.rot90(top_xy, k)
        c = np.rot90(top_rgb, k)
        tx = b.any(axis=0).astype(np.uint8)
        same = float((tx == fx).mean())
        flip = float((tx[::-1] == fx).mean())
        if flip > same:
            candidates.append((flip, np.fliplr(b), np.fliplr(c), {"rot90": float(k), "flip_lr": 1.0, "score": flip}))
        else:
            candidates.append((same, b, c, {"rot90": float(k), "flip_lr": 0.0, "score": same}))

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_bin, best_rgb, info = candidates[0]
    return best_bin, best_rgb, info



def carve_voxel_volume(top_xy: np.ndarray, front_xz: np.ndarray) -> np.ndarray:
    """
    occupancy[z, y, x] = top[y, x] AND front[z, x]

    이 식은 'top footprint와 front silhouette을 동시에 만족하는 최대 부피'를 의미합니다.
    내부 빈 공간은 알 수 없기 때문에 완전 복원은 아니지만,
    구조물의 대략적인 외형과 지지 형태를 보기에는 충분합니다.
    """
    grid_z, grid_xy = front_xz.shape
    occ = np.zeros((grid_z, grid_xy, grid_xy), dtype=bool)

    # 이미지 row=0은 맨 위이므로, voxel z=0을 바닥으로 쓰기 위해 뒤집습니다.
    front_bottom_up = np.flipud(front_xz)

    for z in range(grid_z):
        for x in range(grid_xy):
            if front_bottom_up[z, x]:
                occ[z, :, x] = top_xy[:, x]

    # 간단한 지지성 pruning:
    # 위층 voxel은 바로 아래층의 같은 칸 또는 4-neighborhood 중 하나와 연결되어 있을 때만 유지합니다.
    supported = np.zeros_like(occ, dtype=bool)
    supported[0] = occ[0]
    neighbors = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    for z in range(1, grid_z):
        for y in range(grid_xy):
            for x in range(grid_xy):
                if not occ[z, y, x]:
                    continue
                ok = False
                for dy, dx in neighbors:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < grid_xy and 0 <= xx < grid_xy and supported[z - 1, yy, xx]:
                        ok = True
                        break
                if ok:
                    supported[z, y, x] = True
    return supported


# ============================================================
# 4) 렌더링
# - top 색을 각 column 색으로 써서 사람이 보기 쉬운 3D 이미지를 만듭니다.
# ============================================================

def build_facecolors(top_rgb: np.ndarray, occ: np.ndarray) -> np.ndarray:
    grid_z, grid_y, grid_x = occ.shape
    facecolors = np.zeros(occ.shape + (4,), dtype=np.float32)
    for z in range(grid_z):
        shade = 0.78 + 0.22 * (z / max(grid_z - 1, 1))
        for y in range(grid_y):
            for x in range(grid_x):
                c = top_rgb[y, x].astype(np.float32) / 255.0
                facecolors[z, y, x, :3] = np.clip(c * shade, 0.0, 1.0)
                facecolors[z, y, x, 3] = 1.0
    return facecolors



def render_voxel_image(occ: np.ndarray, top_rgb: np.ndarray, cfg: ReconConfig) -> np.ndarray:
    facecolors = build_facecolors(top_rgb, occ)

    fig = plt.figure(figsize=(4.5, 4.5), dpi=cfg.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(
        occ,
        facecolors=facecolors,
        edgecolor=(0.18, 0.18, 0.18, 0.25),
        linewidth=0.55,
    )
    ax.view_init(elev=cfg.view_elev, azim=cfg.view_azim)
    ax.set_box_aspect((occ.shape[2], occ.shape[1], occ.shape[0]))
    ax.set_axis_off()
    plt.tight_layout(pad=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ============================================================
# 5) triptych / contact sheet 저장
# ============================================================

def resize_for_panel(rgb: np.ndarray, panel_size: int) -> np.ndarray:
    return np.array(Image.fromarray(rgb).resize((panel_size, panel_size), Image.Resampling.BICUBIC))



def make_triptych(front_rgb: np.ndarray, top_rgb: np.ndarray, recon_rgb: np.ndarray,
                  title: str, subtitle: str, cfg: ReconConfig) -> np.ndarray:
    front_vis = resize_for_panel(front_rgb, cfg.panel_size)
    top_vis = resize_for_panel(top_rgb, cfg.panel_size)
    recon_vis = resize_for_panel(recon_rgb, cfg.panel_size)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=cfg.dpi)
    for ax, img, ttl in zip(
        axes,
        [front_vis, top_vis, recon_vis],
        ["front view", "top view", "reconstructed 3D"],
    ):
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))



def build_contact_sheet(image_paths: Sequence[str], save_path: str, columns: int = 2, gap: int = 16) -> None:
    imgs = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not imgs:
        return

    w = max(img.width for img in imgs)
    h = max(img.height for img in imgs)
    cols = max(1, columns)
    rows = int(math.ceil(len(imgs) / cols))

    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), color=(248, 248, 248))
    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = gap + c * (w + gap)
        y = gap + r * (h + gap)
        sheet.paste(img.resize((w, h), Image.Resampling.BICUBIC), (x, y))
    sheet.save(save_path)


# ============================================================
# 6) 샘플 단위 처리
# ============================================================

def process_pair(front_img: Image.Image, top_img: Image.Image, cfg: ReconConfig) -> Dict[str, np.ndarray]:
    front_rgb = pil_to_np(front_img)
    top_rgb = pil_to_np(top_img)

    front_mask = segment_front_mask(front_img, cfg)
    top_mask = segment_top_mask(top_img, cfg)

    front_mask_crop, front_crop_rgb = crop_front(front_mask, front_rgb, cfg)
    top_mask_crop, top_crop_rgb, top_rot_angle = rotate_align_top(top_mask, top_rgb, cfg)

    front_xz = normalize_front_to_grid(front_mask_crop, cfg)
    top_xy, top_grid_rgb = normalize_top_to_grid(top_mask_crop, top_crop_rgb, cfg)
    top_xy, top_grid_rgb, orient_info = choose_top_orientation(top_xy, top_grid_rgb, front_xz)

    occ = carve_voxel_volume(top_xy, front_xz)
    recon_rgb = render_voxel_image(occ, top_grid_rgb, cfg)

    return {
        "front_rgb": front_rgb,
        "top_rgb": top_rgb,
        "front_mask": (front_mask.astype(np.uint8) * 255),
        "top_mask": (top_mask.astype(np.uint8) * 255),
        "front_crop_rgb": front_crop_rgb,
        "top_crop_rgb": top_crop_rgb,
        "front_xz": (front_xz.astype(np.uint8) * 255),
        "top_xy": (top_xy.astype(np.uint8) * 255),
        "occ": occ.astype(np.uint8),
        "recon_rgb": recon_rgb,
        "top_rotation_angle": np.array([top_rot_angle], dtype=np.float32),
        "orientation_info": np.array([
            orient_info.get("rot90", 0.0),
            orient_info.get("flip_lr", 0.0),
            orient_info.get("score", 0.0),
        ], dtype=np.float32),
    }


# ============================================================
# 7) split 단위 처리
# ============================================================

def pick_rows(df: pd.DataFrame, num_per_split: int, seed: int) -> pd.DataFrame:
    n = min(num_per_split, len(df))
    if n <= 0:
        return df.iloc[:0].copy()
    rng = random.Random(seed)
    idx = list(df.index)
    rng.shuffle(idx)
    idx = idx[:n]
    return df.loc[idx].reset_index(drop=True)



def save_optional_masks(sample_dir: str, outputs: Dict[str, np.ndarray]) -> None:
    np_to_pil(outputs["front_mask"]).save(os.path.join(sample_dir, "front_mask.png"))
    np_to_pil(outputs["top_mask"]).save(os.path.join(sample_dir, "top_mask.png"))
    np_to_pil(outputs["front_xz"]).save(os.path.join(sample_dir, "front_xz_grid.png"))
    np_to_pil(outputs["top_xy"]).save(os.path.join(sample_dir, "top_xy_grid.png"))



def process_split(root_dir: str, split: str, save_dir: str, cfg: ReconConfig) -> pd.DataFrame:
    df = load_split_df(root_dir, split)
    verify_paths(df)
    picked = pick_rows(df, cfg.num_per_split, cfg.seed + hash(split) % 1000)

    split_dir = os.path.join(save_dir, split)
    ensure_dir(split_dir)

    rows = []
    triptych_paths: List[str] = []

    for _, row in picked.iterrows():
        sample_id = str(row["id"])
        label = str(row.get("label", "unknown"))
        sample_dir = os.path.join(split_dir, sample_id)
        ensure_dir(sample_dir)

        front_img = load_rgb(row["front_path"])
        top_img = load_rgb(row["top_path"])
        outputs = process_pair(front_img, top_img, cfg)

        # 3D occupancy와 렌더 이미지 저장
        np.save(os.path.join(sample_dir, "occupancy.npy"), outputs["occ"])
        np.save(os.path.join(sample_dir, "orientation_info.npy"), outputs["orientation_info"])
        np_to_pil(outputs["recon_rgb"]).save(os.path.join(sample_dir, "reconstructed_3d.png"))

        if cfg.save_masks:
            save_optional_masks(sample_dir, outputs)

        # 원본 front / top도 함께 저장해 두면 나중에 보고서 작성이 편합니다.
        front_img.save(os.path.join(sample_dir, "front_original.png"))
        top_img.save(os.path.join(sample_dir, "top_original.png"))

        title = f"split={split} | id={sample_id} | label={label}"
        subtitle = f"grid=({cfg.grid_xy}, {cfg.grid_xy}, {cfg.grid_z}) | pseudo-voxel reconstruction"
        panel = make_triptych(outputs["front_rgb"], outputs["top_rgb"], outputs["recon_rgb"], title, subtitle, cfg)
        panel_path = os.path.join(sample_dir, f"{sample_id}_triptych.png")
        np_to_pil(panel).save(panel_path)
        triptych_paths.append(panel_path)

        rows.append({
            "split": split,
            "id": sample_id,
            "label": label,
            "panel_path": panel_path,
            "recon_path": os.path.join(sample_dir, "reconstructed_3d.png"),
            "occ_path": os.path.join(sample_dir, "occupancy.npy"),
        })

    # split별 contact sheet도 한 장으로 저장합니다.
    build_contact_sheet(
        image_paths=triptych_paths,
        save_path=os.path.join(split_dir, f"{split}_triptych_contact_sheet.png"),
        columns=2,
        gap=16,
    )

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(os.path.join(split_dir, f"{split}_triptych_meta.csv"), index=False)
    return meta_df


# ============================================================
# 8) single pair 데모 모드
# - 대용량 데이터셋이 없어도 알고리즘 동작을 즉시 확인할 수 있게 넣은 모드입니다.
# ============================================================

def process_single_pair(front_path: str, top_path: str, save_dir: str, cfg: ReconConfig) -> None:
    ensure_dir(save_dir)
    front_img = load_rgb(front_path)
    top_img = load_rgb(top_path)
    outputs = process_pair(front_img, top_img, cfg)

    np_to_pil(outputs["recon_rgb"]).save(os.path.join(save_dir, "single_reconstructed_3d.png"))
    np.save(os.path.join(save_dir, "single_occupancy.npy"), outputs["occ"])
    if cfg.save_masks:
        np_to_pil(outputs["front_mask"]).save(os.path.join(save_dir, "single_front_mask.png"))
        np_to_pil(outputs["top_mask"]).save(os.path.join(save_dir, "single_top_mask.png"))

    panel = make_triptych(
        outputs["front_rgb"],
        outputs["top_rgb"],
        outputs["recon_rgb"],
        title="single pair demo",
        subtitle=f"grid=({cfg.grid_xy}, {cfg.grid_xy}, {cfg.grid_z}) | pseudo-voxel reconstruction",
        cfg=cfg,
    )
    np_to_pil(panel).save(os.path.join(save_dir, "single_triptych.png"))


# ============================================================
# 9) CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pseudo-voxel reconstruction + triptych saver")
    parser.add_argument("--data_root", type=str, default="./open", help="open.zip 압축 해제 루트 경로")
    parser.add_argument("--save_dir", type=str, default="./voxel_recon_outputs", help="출력 저장 경로")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "dev", "test"], help="처리할 split 목록")
    parser.add_argument("--num_per_split", type=int, default=10, help="split별 저장할 샘플 개수")
    parser.add_argument("--grid_xy", type=int, default=10, help="x/y 방향 pseudo-voxel 해상도")
    parser.add_argument("--grid_z", type=int, default=14, help="높이 방향 pseudo-voxel 해상도")
    parser.add_argument("--seed", type=int, default=42, help="샘플 선택 시드")
    parser.add_argument("--no_save_masks", action="store_true", help="중간 마스크/격자 이미지를 저장하지 않음")
    parser.add_argument("--no_top_convex_hull", action="store_true", help="top footprint smoothing에 convex hull을 사용하지 않음")

    # single pair 데모 모드
    parser.add_argument("--single_front", type=str, default=None, help="단일 front 이미지 경로")
    parser.add_argument("--single_top", type=str, default=None, help="단일 top 이미지 경로")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = ReconConfig(
        grid_xy=args.grid_xy,
        grid_z=args.grid_z,
        num_per_split=args.num_per_split,
        seed=args.seed,
        save_masks=not args.no_save_masks,
        use_top_convex_hull=not args.no_top_convex_hull,
    )

    ensure_dir(args.save_dir)

    # single pair만 넣으면 바로 데모 결과를 만들 수 있습니다.
    if args.single_front and args.single_top:
        process_single_pair(args.single_front, args.single_top, args.save_dir, cfg)
        print(f"[Done] single pair 결과 저장 완료: {args.save_dir}")
        return

    resolved_root = resolve_data_root(args.data_root)
    print(f'[Info] resolved data_root: {resolved_root}')

    all_meta = []
    for split in args.splits:
        meta_df = process_split(resolved_root, split, args.save_dir, cfg)
        all_meta.append(meta_df)
        print(f"[Done] split={split} | saved={len(meta_df)}")

    if all_meta:
        pd.concat(all_meta, axis=0, ignore_index=True).to_csv(os.path.join(args.save_dir, "all_triptych_meta.csv"), index=False)
        print(f"[Done] 전체 메타 저장: {os.path.join(args.save_dir, 'all_triptych_meta.csv')}")


if __name__ == "__main__":
    main()
