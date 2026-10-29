"""
Unified multi-dataset training script for image deepfake detection.

Trains ImageDeepfakeModel (EfficientNet-B0) or EnhancedImageDeepfakeModel
(EfficientNet-B4 + optional FFT branch) on any combination of:
  - CIFAKE      (data/cifake/)
  - ArtiFact    (data/artifact/)
  - GenImage    (data/genimage/)

Checkpoint naming convention:
  checkpoints/image_deepfake/<arch>_<datasets>_best.pth
  checkpoints/image_deepfake/<arch>_<datasets>_final.pth

Label convention (must match ImageDeepfakeModel and api/pipeline.py):
  Class 0 = FAKE  (AI-generated / manipulated)
  Class 1 = REAL  (authentic photograph)

Usage:
  cd c:\\Projects\\Multimodal_AV_fraud_and_deepfake
  python train/train_enhanced_image.py --datasets cifake artifact --model b4
  python train/train_enhanced_image.py --datasets cifake --model b0 --no-pretrained
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from train.cifake_dataset import CIFAKEDataset
from models.image_model import ImageDeepfakeModel, EnhancedImageDeepfakeModel
from train.metrics import calculate_metrics, print_metrics_summary

# Optional dataset imports — skip gracefully if the dataset module is missing
try:
    from train.artifact_dataset import ArtifactDataset
    _HAS_ARTIFACT = True
except ImportError:
    ArtifactDataset = None
    _HAS_ARTIFACT = False
    print("WARNING: train/artifact_dataset.py not found — 'artifact' dataset unavailable.")

try:
    from train.genimage_dataset import GenImageDataset
    _HAS_GENIMAGE = True
except ImportError:
    GenImageDataset = None
    _HAS_GENIMAGE = False
    print("WARNING: train/genimage_dataset.py not found — 'genimage' dataset unavailable.")

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Default directories ───────────────────────────────────────────────────────
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "image_deepfake"


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train image deepfake classifier on multiple datasets."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["cifake", "artifact", "genimage"],
        default=["cifake"],
        help="One or more datasets to train on (default: cifake).",
    )
    p.add_argument(
        "--model",
        choices=["b0", "b4"],
        default="b0",
        help="Backbone: b0 (EfficientNet-B0) or b4 (EfficientNet-B4). Default: b0.",
    )
    p.add_argument(
        "--use_fft_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FFT frequency branch in EnhancedImageDeepfakeModel (B4 only). Default: True.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the classifier head (default: 1e-4).",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4; use 0 on Windows if issues occur).",
    )
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load weights from an existing checkpoint before training (default: True).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to an existing checkpoint to fine-tune from.",
    )
    p.add_argument(
        "--data_root_cifake",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cifake"),
        help="Path to the CIFAKE root folder.",
    )
    p.add_argument(
        "--data_root_artifact",
        type=str,
        default=str(PROJECT_ROOT / "data" / "artifact"),
        help="Path to the ArtiFact root folder.",
    )
    p.add_argument(
        "--data_root_genimage",
        type=str,
        default=str(PROJECT_ROOT / "data" / "genimage"),
        help="Path to the GenImage root folder.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(CKPT_DIR),
        help="Directory to save checkpoints.",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_class_weights(combined_dist: dict) -> torch.Tensor:
    """
    Compute balanced class weights from combined class distribution dict.
    Returns a tensor suitable for nn.CrossEntropyLoss(weight=...).
    Index 0 = FAKE, Index 1 = REAL.
    """
    n_fake = combined_dist.get("FAKE", 0)
    n_real = combined_dist.get("REAL", 0)
    total = n_fake + n_real
    weight_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    weight_real = total / (2.0 * n_real) if n_real > 0 else 1.0
    return torch.tensor([weight_fake, weight_real], dtype=torch.float32)


def sum_distributions(datasets_list: list) -> dict:
    """
    Sum class_distribution() across all datasets in the list.
    Each dataset must implement class_distribution() -> {'FAKE': int, 'REAL': int}.
    For ConcatDataset we iterate the underlying datasets attribute.
    """
    total = {"FAKE": 0, "REAL": 0}
    for ds in datasets_list:
        dist = ds.class_distribution()
        for k in ("FAKE", "REAL"):
            total[k] = total.get(k, 0) + dist.get(k, 0)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    lr = args.lr
    wd = args.weight_decay
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Model-dependent image size
    image_size = 380 if args.model == "b4" else 160

    # Checkpoint filenames derived from arch + sorted dataset list
    datasets_str = "_".join(sorted(args.datasets))
    model_arch = "efficientnet_b4" if args.model == "b4" else "efficientnet_b0"
    best_name = f"{model_arch}_{datasets_str}_best.pth"
    final_name = f"{model_arch}_{datasets_str}_final.pth"
    best_ckpt_path = output_dir / best_name
    final_ckpt_path = output_dir / final_name

    print("=" * 70)
    print("ENHANCED IMAGE DEEPFAKE TRAINING — Multi-Dataset")
    print("=" * 70)
    print(f"Device        : {DEVICE}")
    print(f"Datasets      : {args.datasets}")
    print(f"Model arch    : {model_arch}")
    if args.model == "b4":
        print(f"FFT branch    : {args.use_fft_branch}")
    print(f"Image size    : {image_size}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Weight decay  : {wd}")
    print(f"Num workers   : {num_workers}")
    print(f"Pretrained    : {args.pretrained}")
    print(f"Checkpoint in : {args.checkpoint or '(none)'}")
    print(f"Best ckpt out : {best_ckpt_path}")
    print(f"Final ckpt out: {final_ckpt_path}")
    print("=" * 70)

    # ── 1. Build per-dataset train/val splits ─────────────────────────────
    train_parts = []
    val_parts = []

    for ds_name in args.datasets:
        if ds_name == "cifake":
            train_parts.append(
                CIFAKEDataset(
                    root=args.data_root_cifake,
                    split="train",
                    image_size=image_size,
                    training=True,
                )
            )
            val_parts.append(
                CIFAKEDataset(
                    root=args.data_root_cifake,
                    split="test",
                    image_size=image_size,
                    training=False,
                )
            )

        elif ds_name == "artifact":
            if not _HAS_ARTIFACT:
                print(
                    "WARNING: Skipping 'artifact' — train/artifact_dataset.py not available."
                )
                continue
            train_parts.append(
                ArtifactDataset(
                    root=args.data_root_artifact,
                    image_size=image_size,
                    training=True,
                )
            )
            val_parts.append(
                ArtifactDataset(
                    root=args.data_root_artifact,
                    image_size=image_size,
                    training=False,
                )
            )

        elif ds_name == "genimage":
            if not _HAS_GENIMAGE:
                print(
                    "WARNING: Skipping 'genimage' — train/genimage_dataset.py not available."
                )
                continue
            train_parts.append(
                GenImageDataset(
                    root=args.data_root_genimage,
                    image_size=image_size,
                    training=True,
                )
            )
            val_parts.append(
                GenImageDataset(
                    root=args.data_root_genimage,
                    image_size=image_size,
                    training=False,
                )
            )

    if not train_parts:
        print("ERROR: No datasets were loaded. Exiting.")
        sys.exit(1)

    # Combine datasets
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
        val_dataset = val_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)
        val_dataset = ConcatDataset(val_parts)

    # Print combined class distribution
    combined_train_dist = sum_distributions(train_parts)
    combined_val_dist = sum_distributions(val_parts)
    print(f"\nCombined Train distribution : {combined_train_dist}")
    print(f"Combined Val distribution   : {combined_val_dist}")
    print(f"Total train samples         : {len(train_dataset)}")
    print(f"Total val samples           : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")

    # ── 2. Model ───────────────────────────────────────────────────────────
    if args.model == "b4":
        model = EnhancedImageDeepfakeModel(use_fft_branch=args.use_fft_branch)
        print(f"\nModel: EnhancedImageDeepfakeModel (B4, fft={args.use_fft_branch})")

        # Freeze early blocks; unfreeze last 4
        for param in model.backbone.features[:-4].parameters():
            param.requires_grad = False
        for param in model.backbone.features[-4:].parameters():
            param.requires_grad = True
        if model.fft_branch is not None:
            for param in model.fft_branch.parameters():
                param.requires_grad = True
        for param in model.classifier_head.parameters():
            param.requires_grad = True

    else:
        model = ImageDeepfakeModel()
        print("\nModel: ImageDeepfakeModel (B0)")

        # Freeze early blocks; unfreeze last 4
        for param in model.backbone.features[:-4].parameters():
            param.requires_grad = False
        for param in model.backbone.features[-4:].parameters():
            param.requires_grad = True
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

    # Load checkpoint if requested
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"\nLoading checkpoint from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            state_dict = (
                ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            )
            model.load_state_dict(state_dict, strict=False)
            print("  Checkpoint weights loaded (strict=False).")
        else:
            print(f"\nWARNING: --checkpoint path not found: {ckpt_path}. Starting from ImageNet weights.")
    elif args.pretrained:
        # Try to warm-start from existing best checkpoint for this arch
        default_ckpt = output_dir / f"{model_arch}_image_best.pth"
        if default_ckpt.exists():
            print(f"\nLoading pretrained weights from: {default_ckpt}")
            ckpt = torch.load(default_ckpt, map_location=DEVICE, weights_only=False)
            state_dict = (
                ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            )
            model.load_state_dict(state_dict, strict=False)
            print("  Pretrained weights loaded (strict=False).")
        else:
            print(
                f"\nINFO: --pretrained set but no default checkpoint found at {default_ckpt}. "
                "Starting from ImageNet weights."
            )

    model.to(DEVICE)

    # ── 3. Loss with balanced class weights ───────────────────────────────
    class_weights = compute_class_weights(combined_train_dist).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(
        f"\nClass weights — FAKE: {class_weights[0]:.4f}, REAL: {class_weights[1]:.4f}"
    )

    # ── 4. Optimizer & Scheduler ──────────────────────────────────────────
    if args.model == "b4":
        opt_param_groups = [
            {"params": model.backbone.features[-4:].parameters(), "lr": lr * 0.1},
            {"params": model.classifier_head.parameters(), "lr": lr},
        ]
        if model.fft_branch is not None:
            opt_param_groups.append(
                {"params": model.fft_branch.parameters(), "lr": lr}
            )
    else:
        opt_param_groups = [
            {"params": model.backbone.features[-4:].parameters(), "lr": lr * 0.1},
            {"params": model.backbone.classifier.parameters(), "lr": lr},
        ]

    optimizer = torch.optim.AdamW(opt_param_groups, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    epochs_no_improve = 0
    patience = 5
    val_acc = 0.0
    val_f1 = 0.0
    val_auc = 0.0
    epoch = -1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training loop...")
    for epoch in range(epochs):

        # ── Train phase ────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for tensors, labels in train_loader:
            tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(tensors)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_acc = train_correct / train_total

        # ── Validation phase ───────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true_val, y_pred_val, y_prob_val = [], [], []
        with torch.no_grad():
            for tensors, labels in val_loader:
                tensors, labels = tensors.to(DEVICE), labels.to(DEVICE)
                outputs = model(tensors)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true_val.extend(labels.cpu().tolist())
                y_pred_val.extend(preds.cpu().tolist())
                # Probability of class 1 (REAL) for AUC — standard convention
                probs = torch.softmax(outputs, dim=1)
                y_prob_val.extend(probs[:, 1].cpu().tolist())

        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1:02d}/{epochs}]  "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%  |  "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%",
            flush=True,
        )

        try:
            val_metrics = calculate_metrics(y_true_val, y_pred_val, y_prob_val)
            val_f1 = val_metrics["f1_score"]
            val_auc = val_metrics["roc_auc"]
            print(
                f"         F1: {val_f1:.4f} | AUC: {val_auc:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"         (metrics error: {e})", flush=True)
            val_f1 = 0.0
            val_auc = 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "val_auc": val_auc,
                    "epoch": epoch + 1,
                    "datasets": args.datasets,
                    "model_arch": model_arch,
                    "threshold": 0.5,
                },
                best_ckpt_path,
            )
            print(
                f"  [SAVED] Best checkpoint → {best_ckpt_path}  "
                f"(Val Acc: {val_acc*100:.2f}%  F1: {val_f1:.4f}  AUC: {val_auc:.4f})",
                flush=True,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"\nEarly stopping: no improvement for {patience} consecutive epochs.",
                    flush=True,
                )
                break

    # Save final checkpoint regardless of whether it is the best.
    if epochs > 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "epoch": epoch + 1,
                "datasets": args.datasets,
                "model_arch": model_arch,
                "threshold": 0.5,
            },
            final_ckpt_path,
        )
        print(f"\n[SAVED] Final checkpoint → {final_ckpt_path}", flush=True)

    print("=" * 70, flush=True)
    print(
        f"Training complete.  Best Val Acc: {best_val_acc*100:.2f}%  "
        f"F1: {best_val_f1:.4f}  AUC: {best_val_auc:.4f}",
        flush=True,
    )
    print(f"Best checkpoint  : {best_ckpt_path}", flush=True)
    print(f"Final checkpoint : {final_ckpt_path}", flush=True)
    print("=" * 70, flush=True)

    # ── Post-training instructions ────────────────────────────────────────
    print(
        "\nTo use the new checkpoint in the API:\n"
        "  Edit api/pipeline.py and update IMAGE_CHECKPOINT to:\n"
        f"    {best_ckpt_path}\n"
        "  Then restart the API server.\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
