import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.dataset import VideoFrameDataset
from models.video_model import VideoDeepfakeModel

NUM_FRAMES = 8
IMAGE_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 15
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("=" * 65)
    print("SYSTEMATIC EFFICIENTNET-B0 FINE-TUNING & DEEPFAKE RETRAINING")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch Size:", BATCH_SIZE)
    print("Train CSV: data/metadata/train_split.csv (220 samples)")
    print("Val CSV:   data/metadata/val_split.csv   (60 samples)")
    print("Test CSV:  data/metadata/test.csv        (60 samples - HELD OUT)")
    print("=" * 65)

    # 1. Datasets
    train_dataset = VideoFrameDataset(
        csv_file="data/metadata/train_split.csv",
        frames_root="data/processed/train/frames",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        training=True
    )

    val_dataset = VideoFrameDataset(
        csv_file="data/metadata/val_split.csv",
        frames_root="data/processed/train/frames",
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model Initialization
    model = VideoDeepfakeModel()
    
    # Load existing baseline if available
    ckpt_path = Path("checkpoints/video_deepfake/efficientnet_b0_best.pth")
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            print("Loaded baseline checkpoint for fine-tuning.")
        except Exception as e:
            print(f"Notice: Initial baseline load: {e}")

    model.to(DEVICE)

    # 3. Fine-tuning Unfreezing Strategy
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    # Unfreeze final 4 feature blocks of EfficientNet-B0 for deepfake artifact adaptation
    for param in model.backbone.features[-4:].parameters():
        param.requires_grad = True
        
    for param in model.temporal_gru.parameters():
        param.requires_grad = True
        
    for param in model.attention.parameters():
        param.requires_grad = True
        
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 4. Criterion & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.features[-4:].parameters(), "lr": BACKBONE_LR},
        {"params": model.temporal_gru.parameters(), "lr": HEAD_LR},
        {"params": model.attention.parameters(), "lr": HEAD_LR},
        {"params": model.classifier.parameters(), "lr": HEAD_LR}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    print("\nStarting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for frames, labels in train_loader:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(frames)
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

        # Validation phase on val_split.csv
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "epoch": epoch + 1
            }, ckpt_path)
            print(f"  [SAVED] Checkpoint saved with Val Acc: {val_acc*100:.2f}%", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.", flush=True)
                break

    print("=" * 65, flush=True)
    print(f"Training completed. Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    main()
