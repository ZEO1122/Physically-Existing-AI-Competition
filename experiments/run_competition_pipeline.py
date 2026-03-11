from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


CODEX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODEX_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data_reconstrcture" / "open"
OUTPUT_ROOT = CODEX_ROOT / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

HYBRID_SCRIPT = PROJECT_ROOT / "data_reconstrcture" / "train_hybrid_structure_aware_gated_simaux.py"
PHYSNET_SCRIPT = PROJECT_ROOT / "model_file" / "baseline_model" / "physnet_multiview_baseline.py"
ANALYZE_SCRIPT = CODEX_ROOT / "analyze_dataset.py"


def run(cmd: list[str]) -> int:
    print("Running command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(cmd, check=False).returncode


def train_hybrid(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(HYBRID_SCRIPT),
        "--mode",
        args.mode,
        "--data_root",
        str(DATA_ROOT),
        "--save_dir",
        str(OUTPUT_ROOT / "hybrid"),
        "--run_name",
        args.run_name,
        "--img_size",
        str(args.img_size),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--nfolds",
        str(args.nfolds),
        "--num_workers",
        str(args.num_workers),
        "--learning_rate",
        str(args.learning_rate),
        "--backbone_lr",
        str(args.backbone_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--label_smoothing",
        str(args.label_smoothing),
        "--aug_profile",
        args.aug_profile,
        "--crop_profile",
        args.crop_profile,
        "--crop_margin_ratio",
        str(args.crop_margin_ratio),
        "--crop_min_side_ratio",
        str(args.crop_min_side_ratio),
        "--patience",
        str(args.patience),
        "--geom_aux_weight",
        str(args.geom_aux_weight),
        "--sim_aux_weight",
        str(args.sim_aux_weight),
    ]
    if args.no_amp:
        cmd.append("--no_use_amp")
    if args.no_tta:
        cmd.append("--no_tta_hflip")
    return run(cmd)


def train_physnet(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(PHYSNET_SCRIPT),
        "--mode",
        args.mode,
        "--backbone_name",
        args.backbone_name,
        "--data_root",
        str(DATA_ROOT),
        "--save_dir",
        str(OUTPUT_ROOT / "physnet"),
        "--run_name",
        args.run_name,
        "--seed",
        str(args.seed),
        "--img_size",
        str(args.img_size),
        "--motion_size",
        str(args.motion_size),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--nfolds",
        str(args.nfolds),
        "--num_workers",
        str(args.num_workers),
        "--learning_rate",
        str(args.learning_rate),
        "--backbone_lr",
        str(args.backbone_lr),
        "--weight_decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--label_smoothing",
        str(args.label_smoothing),
        "--aug_profile",
        args.aug_profile,
        "--patience",
        str(args.patience),
        "--motion_loss_weight",
        str(args.motion_loss_weight),
        "--motion_dice_weight",
        str(args.motion_dice_weight),
        "--motion_timepoints",
        args.motion_timepoints,
        "--tta_scales",
        args.tta_scales,
    ]
    if args.no_amp:
        cmd.append("--no_use_amp")
    if args.no_tta:
        cmd.append("--no_tta_hflip")
    return run(cmd)


def run_analysis(_: argparse.Namespace) -> int:
    return run([sys.executable, str(ANALYZE_SCRIPT)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Competition helper runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Run dataset analysis")
    p.set_defaults(func=run_analysis)

    p = sub.add_parser("train-hybrid", help="Run recommended hybrid model")
    p.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    p.add_argument("--run_name", default="codex_hybrid")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--aug_profile", choices=["base", "strong_domain", "crop_tuned"], default="base")
    p.add_argument("--crop_profile", choices=["none", "mask_bbox"], default="none")
    p.add_argument("--crop_margin_ratio", type=float, default=0.18)
    p.add_argument("--crop_min_side_ratio", type=float, default=0.42)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--geom_aux_weight", type=float, default=0.10)
    p.add_argument("--sim_aux_weight", type=float, default=0.08)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_tta", action="store_true")
    p.set_defaults(func=train_hybrid)

    p = sub.add_parser("train-physnet", help="Run motion-aux baseline")
    p.add_argument("--mode", choices=["holdout", "cv"], default="holdout")
    p.add_argument("--run_name", default="codex_physnet")
    p.add_argument("--backbone_name", choices=["resnet34", "resnet50", "resnet101", "convnext_tiny", "convnext_small"], default="resnet34")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--motion_size", type=int, default=56)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--nfolds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--backbone_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--label_smoothing", type=float, default=0.02)
    p.add_argument("--aug_profile", choices=["light", "strong"], default="strong")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--motion_loss_weight", type=float, default=0.15)
    p.add_argument("--motion_dice_weight", type=float, default=0.25)
    p.add_argument("--motion_timepoints", default="2.5,5.0,7.5,10.0")
    p.add_argument("--tta_scales", default="1.0")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--no_tta", action="store_true")
    p.set_defaults(func=train_physnet)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
