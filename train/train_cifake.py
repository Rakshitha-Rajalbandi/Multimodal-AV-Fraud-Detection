"""
CIFAKE Retraining Script for ImageDeepfakeModel (EfficientNet-B0).

Retrains the image classifier on the CIFAKE benchmark dataset (120,000 train images,
20,000 test images). Saves best checkpoint to:
  checkpoints/image_deepfake/efficientnet_b0_cifake_best.pth

Label convention (must match ImageDeepfakeModel and api/pipeline.py):
  Class 0 = FAKE  (AI-generated / manipulated)
  Class 1 = REAL  (authentic photograph)

Training configuration (mirrors train/train_image_classifier.py):
  IMAGE_SIZE   = 160
  Backbone freeze: all features[:-4] frozen; last 4 blocks + classifier trained
  Optimizer: AdamW (backbone last 4 blocks lr=1e-5, classifier head lr=2e-4)
  Scheduler: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
  Gradient clipping: max_norm=1.0
  Loss: CrossEntropyLoss with balanced class weights

Usage:
  cd c:\\Projects\\Multimodal_AV_fraud_and_deepfake
  python train/train_cifake.py
  python train/train_cifake.py --epochs 15 --batch_size 32
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.cifake_dataset import CIFAKEDataset
from models.image_model import ImageDeepfakeModel
from models.image_model import EnhancedImageDeepfakeModel
from train.metrics import calculate_metrics, print_metrics_summary

# ── Fixed architecture constants ────────────────────────────────────────────
IMAGE_SIZE = 160

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint paths
BEST_CKPT_NAME  = "efficientnet_b0_cifake_best.pth"
FINAL_CKPT_NAME = "efficientnet_b0_cifake_final.pth"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "image_deepfake"

# Pretrained checkpoint to warm-start from (original face-swap model)
PRETRAINED_CKPT = CKPT_DIR / "efficientnet_b0_image_best.pth"


def parse_args():
    p = argparse.ArgumentParser(
        description="Retrain ImageDeepfakeModel on CIFAKE dataset."
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
        "--learning_rate",
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
        help=(
            "Load weights from the existing checkpoint "
            "checkpoints/image_deepfake/efficientnet_b0_image_best.pth "
            "before retraining (default: True). "
            "Pass --no-pretrained to train from ImageNet weights only."
        ),
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cifake"),
        help="Path to the CIFAKE root folder (default: <project_root>/data/cifake).",
    )
    p.add_argument(
        "--model",
        choices=["b0", "b4"],
        default="b0",
        help="Backbone architecture: b0 (EfficientNet-B0, IMAGE_SIZE=160) or b4 (EfficientNet-B4, IMAGE_SIZE=380). Default: b0.",
    )
    p.add_argument(
        "--use_fft_branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable FFT frequency branch in EnhancedImageDeepfakeModel (B4 only). Default: True.",
    )
    return p.parse_args()


def compute_class_weights(dataset):
    """
    Compute balanced class weights from the dataset's label distribution.
    Returns a tensor suitable for nn.CrossEntropyLoss(weight=...).
    """
    dist = dataset.class_distribution()
    n_fake = dist["FAKE"]
    n_real = dist["REAL"]
    total = n_fake + n_real

    # weight_i = total / (num_classes * count_i)
    weight_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
    weight_real = total / (2.0 * n_real) if n_real > 0 else 1.0

    # Index 0 = FAKE, Index 1 = REAL
    return torch.tensor([weight_fake, weight_real], dtype=torch.float32)


def main():
    args = parse_args()
    cifake_root = Path(args.data_root)
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    wd = args.weight_decay
    num_workers = args.num_workers

    # Model-dependent configuration
    if args.model == "b4":
        image_size = 380
        best_ckpt_name  = "efficientnet_b4_cifake_best.pth"
        final_ckpt_name = "efficientnet_b4_cifake_final.pth"
    else:
        image_size = IMAGE_SIZE   # 160
        best_ckpt_name  = BEST_CKPT_NAME
        final_ckpt_name = FINAL_CKPT_NAME

    print("=" * 65)
    print("CIFAKE RETRAINING — EfficientNet-B0 ImageDeepfakeModel")
    print("=" * 65)
    print(f"Device        : {DEVICE}")
    print(f"CIFAKE root   : {cifake_root.resolve()}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Weight decay  : {wd}")
    print(f"Num workers   : {num_workers}")
    print(f"Pretrained    : {args.pretrained}")
    print(f"Best ckpt     : {CKPT_DIR / best_ckpt_name}")
    print(f"Final ckpt    : {CKPT_DIR / final_ckpt_name}")
    print("=" * 65)

    # ── 1. Datasets ────────────────────────────────────────────────────────
    train_dataset = CIFAKEDataset(
        root=cifake_root, split="train", image_size=image_size, training=True
    )
    val_dataset = CIFAKEDataset(
        root=cifake_root, split="test", image_size=image_size, training=False
    )

    print(f"Train distribution : {train_dataset.class_distribution()}")
    print(f"Val distribution   : {val_dataset.class_distribution()}")

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
        print(f"Model          : EnhancedImageDeepfakeModel (B4, fft={args.use_fft_branch})")
        # Freeze same pattern: features[0..4] frozen, features[5..8] trainable
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
        print("Model          : ImageDeepfakeModel (B0)")
        # Warm-start from existing checkpoint if requested
        if args.pretrained and PRETRAINED_CKPT.exists():
            print(f"\nLoading pretrained weights from: {PRETRAINED_CKPT}")
            ckpt = torch.load(PRETRAINED_CKPT, map_location=DEVICE, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("  Pretrained weights loaded (strict=False).")
        elif args.pretrained:
            print(f"\nWARNING: --pretrained set but checkpoint not found at {PRETRAINED_CKPT}. Training from ImageNet weights only.")
        for param in model.backbone.features[:-4].parameters():
            param.requires_grad = False
        for param in model.backbone.features[-4:].parameters():
            param.requires_grad = True
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True

    model.to(DEVICE)

    # ── 3. Loss with balanced class weights ───────────────────────────────
    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\nClass weights — FAKE: {class_weights[0]:.4f}, REAL: {class_weights[1]:.4f}")

    # ── 4. Optimizer & Scheduler ──────────────────────────────────────────
    if args.model == "b4":
        opt_param_groups = [
            {"params": model.backbone.features[-4:].parameters(), "lr": lr * 0.1},
            {"params": model.classifier_head.parameters(), "lr": lr},
        ]
        if model.fft_branch is not None:
            opt_param_groups.append({"params": model.fft_branch.parameters(), "lr": lr})
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
    epochs_no_improve = 0
    patience = 5
    # Initialised here so the final-checkpoint save outside the loop is safe
    # even when --epochs 0 is passed (loop body would never execute otherwise).
    val_acc = 0.0
    epoch = -1

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path  = CKPT_DIR / best_ckpt_name
    final_ckpt_path = CKPT_DIR / final_ckpt_name

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

            train_loss    += loss.item()
            preds          = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

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
                loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                preds        = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                y_true_val.extend(labels.cpu().tolist())
                y_pred_val.extend(preds.cpu().tolist())
                # prob of class 1 (REAL) for AUC — standard convention
                probs = torch.softmax(outputs, dim=1)
                y_prob_val.extend(probs[:, 1].cpu().tolist())

        val_acc        = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print(
            f"Epoch [{epoch+1:02d}/{epochs}]  "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%  |  "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%",
            flush=True,
        )

        try:
            val_metrics = calculate_metrics(y_true_val, y_pred_val, y_prob_val)
            print(
                f"         F1: {val_metrics['f1_score']:.4f} | AUC: {val_metrics['roc_auc']:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"         (metrics error: {e})", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy":     val_acc,
                    "epoch":            epoch + 1,
                    "dataset":          "CIFAKE",
                    "threshold":        0.5,
                },
                best_ckpt_path,
            )
            print(
                f"  [SAVED] Best checkpoint → {best_ckpt_path}  "
                f"(Val Acc: {val_acc*100:.2f}%)",
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
    # Guard against --epochs 0 (loop never ran; nothing worth saving).
    if epochs > 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy":     val_acc,
                "epoch":            epoch + 1,
                "dataset":          "CIFAKE",
                "threshold":        0.5,
            },
            final_ckpt_path,
        )
        print(f"\n[SAVED] Final checkpoint → {final_ckpt_path}", flush=True)

    print("=" * 65, flush=True)
    print(
        f"Training complete. Best Val Acc: {best_val_acc*100:.2f}%", flush=True
    )
    print(f"Best checkpoint  : {best_ckpt_path}", flush=True)
    print(f"Final checkpoint : {final_ckpt_path}", flush=True)
    print("=" * 65, flush=True)

    # ── Post-training instructions ────────────────────────────────────────
    print(
        "\nTo use the new CIFAKE-trained checkpoint in the API:\n"
        "  Edit api/pipeline.py line IMAGE_CHECKPOINT to point to:\n"
        f"    {best_ckpt_path}\n"
        "  Or copy it over the existing checkpoint:\n"
        f"    copy \"{best_ckpt_path}\" \"{CKPT_DIR / 'efficientnet_b0_image_best.pth'}\"\n"
        "  Then restart the API server.\n"
        "  See train/CIFAKE_RETRAINING.md for full instructions.",
        flush=True,
    )


if __name__ == "__main__":
    main()
