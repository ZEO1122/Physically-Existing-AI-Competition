# gradcam_multiview.py
import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

import matplotlib
matplotlib.use("Agg")  # 서버/CLI 환경에서도 저장 가능
import matplotlib.pyplot as plt


# -------------------------
# Model (baseline과 동일)
# -------------------------
class MultiViewResNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, views):
        # views: [front_batch, top_batch]
        f1 = self.feature_extractor(views[0]).view(views[0].size(0), -1)
        f2 = self.feature_extractor(views[1]).view(views[1].size(0), -1)
        combined = torch.cat((f1, f2), dim=1)
        return self.classifier(combined)


# -------------------------
# Utils: id 폴더 매칭(0 padding 대응)
# -------------------------
def infer_folder_id_width(split_dir: str) -> Optional[int]:
    if not os.path.isdir(split_dir):
        return None
    dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    if not dirs:
        return None
    numeric = [d for d in dirs if d.isdigit()]
    if len(numeric) < max(10, len(dirs)//2):
        return None
    lengths = [len(d) for d in numeric]
    # 가장 흔한 길이
    vals, cnts = np.unique(lengths, return_counts=True)
    most = int(vals[np.argmax(cnts)])
    if cnts.max() / max(1, len(numeric)) >= 0.8:
        return most
    return None


def resolve_sample_folder(split_dir: str, sample_id):
    """
    다양한 폴더 네이밍을 자동으로 매칭:
    - 그대로 (예: TRAIN_0001 / 0001 / 1)
    - TRAIN_{id}
    - TRAIN_{id:04d}
    - {id:04d}
    - zfill 추정(기존 로직)
    """
    sid_raw = str(sample_id)

    # 1) 그대로 존재하면 OK
    cand = os.path.join(split_dir, sid_raw)
    if os.path.isdir(cand):
        return cand, sid_raw

    # 2) 숫자라면 4자리/기타 변형 시도
    if sid_raw.isdigit():
        sid4 = sid_raw.zfill(4)

        # TRAIN_0123
        cand2 = os.path.join(split_dir, f"TRAIN_{sid4}")
        if os.path.isdir(cand2):
            return cand2, f"TRAIN_{sid4}"

        # 0123 (혹시 prefix 없는 경우)
        cand3 = os.path.join(split_dir, sid4)
        if os.path.isdir(cand3):
            return cand3, sid4

        # TRAIN_123 (혹시 0-padding 없는 경우)
        cand4 = os.path.join(split_dir, f"TRAIN_{sid_raw}")
        if os.path.isdir(cand4):
            return cand4, f"TRAIN_{sid_raw}"

    # 3) 기존: 폴더명 보고 zfill 폭 추정
    width = infer_folder_id_width(split_dir)
    if width is not None and sid_raw.isdigit():
        sid2 = sid_raw.zfill(width)

        cand5 = os.path.join(split_dir, sid2)
        if os.path.isdir(cand5):
            return cand5, sid2

        cand6 = os.path.join(split_dir, f"TRAIN_{sid2}")
        if os.path.isdir(cand6):
            return cand6, f"TRAIN_{sid2}"

    return None, sid_raw


# -------------------------
# Grad-CAM for multi-call (front/top) shared backbone
# -------------------------
class GradCAMMultiCall:
    """
    target_layer가 한 forward에 여러 번 호출되는 경우(여기서는 front/top 2번),
    activation/gradient를 '호출 순서대로' 각각 저장해서 CAM을 2장 생성한다.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: List[torch.Tensor] = []
        self.gradients: List[Optional[torch.Tensor]] = []
        self.handle = None

    def _forward_hook(self, module, inp, out):
        # out: [B,C,H,W]
        self.activations.append(out.detach())
        self.gradients.append(None)
        idx = len(self.activations) - 1

        def _save_grad(grad):
            self.gradients[idx] = grad.detach()

        out.register_hook(_save_grad)

    def __enter__(self):
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None

    def compute(self, views: List[torch.Tensor], target: str, img_size: int) -> List[np.ndarray]:
        """
        views: [front_batch, top_batch], 각각 [1,3,H,W]
        target: "unstable" (logit) or "stable" (-logit)
        img_size: CAM upsample size
        return: [cam_front, cam_top] (각각 0~1 normalized numpy [H,W])
        """
        self.model.zero_grad(set_to_none=True)
        self.activations.clear()
        self.gradients.clear()

        logits = self.model(views)            # [1,1]
        logit = logits.view(-1)[0]            # scalar

        if target == "unstable":
            score = logit
        elif target == "stable":
            # stable 확률은 sigmoid(-logit)이므로 stable 관점 CAM은 -logit을 타깃으로 둠
            score = -logit
        else:
            raise ValueError("target must be 'unstable' or 'stable'")

        score.backward(retain_graph=False)

        cams: List[np.ndarray] = []
        for act, grad in zip(self.activations, self.gradients):
            if grad is None:
                raise RuntimeError("Gradient was not captured. Check hooks/target layer.")

            # act, grad: [1,C,H,W]
            # weights: [1,C,1,1]
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1, keepdim=True)      # [1,1,H,W]
            cam = F.relu(cam)

            cam = F.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)
            cam = cam[0, 0]  # [H,W]

            cam_np = cam.cpu().numpy()
            cam_np = cam_np - cam_np.min()
            cam_np = cam_np / (cam_np.max() + 1e-8)
            cams.append(cam_np)

        return cams, float(logit.detach().cpu().item())


# -------------------------
# Visualization helpers
# -------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def tensor_to_rgb01(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [3,H,W], normalized (imagenet)
    return: [H,W,3] in 0..1
    """
    x = img_t.detach().cpu().float().numpy()
    x = (x * IMAGENET_STD[:, None, None]) + IMAGENET_MEAN[:, None, None]
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))
    return x

def save_cam_figure(
    out_path: str,
    front_rgb: np.ndarray,
    top_rgb: np.ndarray,
    cam_front: np.ndarray,
    cam_top: np.ndarray,
    sample_id: str,
    target: str,
    logit: float,
    prob_unstable: float,
    alpha: float = 0.35,
):
    """
    front/top 각각 원본+오버레이를 한 장 그림으로 저장
    """
    fig = plt.figure(figsize=(12, 7))

    # front
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(front_rgb)
    ax1.set_title(f"FRONT (input)")
    ax1.axis("off")

    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(front_rgb)
    ax2.imshow(cam_front, cmap="jet", alpha=alpha)
    ax2.set_title(f"FRONT Grad-CAM ({target})")
    ax2.axis("off")

    # top
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(top_rgb)
    ax3.set_title(f"TOP (input)")
    ax3.axis("off")

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(top_rgb)
    ax4.imshow(cam_top, cmap="jet", alpha=alpha)
    ax4.set_title(f"TOP Grad-CAM ({target})")
    ax4.axis("off")

    fig.suptitle(
        f"id={sample_id} | logit(unstable)={logit:.4f} | p(unstable)={prob_unstable:.4f}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./open", help="dataset root (default: ./open)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "dev", "test"])
    parser.add_argument("--id", type=str, required=True, help="sample id (csv id)")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="checkpoint path (default: best.pt)")
    parser.add_argument("--out", type=str, default="./gradcam_out", help="output dir")
    parser.add_argument("--img_size", type=int, default=224, help="input size (default 224)")
    parser.add_argument("--target", type=str, default="unstable", choices=["unstable", "stable"],
                        help="CAM target: unstable(logit) or stable(-logit)")
    parser.add_argument("--layer", type=str, default="layer4",
                        choices=["layer1", "layer2", "layer3", "layer4"],
                        help="ResNet target layer for CAM (default: layer4)")
    parser.add_argument("--alpha", type=float, default=0.35, help="overlay alpha")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # split dir
    split_dir = os.path.join(args.root, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split directory not found: {split_dir}")

    folder, resolved = resolve_sample_folder(split_dir, args.id)
    if folder is None:
        raise FileNotFoundError(f"folder not found for id={args.id} in {split_dir}")

    front_path = os.path.join(folder, "front.png")
    top_path   = os.path.join(folder, "top.png")
    if not os.path.isfile(front_path) or not os.path.isfile(top_path):
        raise FileNotFoundError(f"front/top not found: {front_path}, {top_path}")

    # preprocess (val/test와 동일: Resize + Normalize)
    preprocess = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])

    # load images
    front_pil = Image.open(front_path).convert("RGB")
    top_pil   = Image.open(top_path).convert("RGB")

    front_t = preprocess(front_pil).unsqueeze(0).to(device)  # [1,3,H,W]
    top_t   = preprocess(top_pil).unsqueeze(0).to(device)

    # model
    model = MultiViewResNet().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # target layer
    target_layer = getattr(model.backbone, args.layer)

    with GradCAMMultiCall(model, target_layer) as cam_engine:
        cams, logit = cam_engine.compute([front_t, top_t], target=args.target, img_size=args.img_size)

    if len(cams) != 2:
        raise RuntimeError(f"Expected 2 CAMs (front/top), got {len(cams)}. "
                           f"Hook layer might not be called twice.")

    cam_front, cam_top = cams
    prob_unstable = float(torch.sigmoid(torch.tensor(logit)).item())

    # make display images from the exact model input tensors (unnormalize)
    front_rgb = tensor_to_rgb01(front_t[0])  # [H,W,3]
    top_rgb   = tensor_to_rgb01(top_t[0])

    out_path = os.path.join(args.out, f"gradcam_{args.split}_id{args.id}_resolved{resolved}_{args.target}_{args.layer}.png")
    save_cam_figure(
        out_path=out_path,
        front_rgb=front_rgb,
        top_rgb=top_rgb,
        cam_front=cam_front,
        cam_top=cam_top,
        sample_id=str(args.id),
        target=args.target,
        logit=logit,
        prob_unstable=prob_unstable,
        alpha=args.alpha,
    )

    print("DONE ✅")
    print(" saved:", out_path)
    print(f" id={args.id} (resolved={resolved}) | logit(unstable)={logit:.6f} | p(unstable)={prob_unstable:.6f}")


if __name__ == "__main__":
    main()