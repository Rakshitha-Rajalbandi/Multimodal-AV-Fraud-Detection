# import torch
# import torch.nn as nn

# from torch.utils.data import DataLoader

# from dataset import VideoFrameDataset
# from models.video_model import VideoDeepfakeModel

# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# import torch
# import torch.nn as nn

# from torch.utils.data import DataLoader

# from dataset import VideoFrameDataset
# from models.video_model import VideoDeepfakeModel


# # ============================================================
# # Configuration
# # ============================================================

# NUM_FRAMES = 8
# IMAGE_SIZE = 160

# BATCH_SIZE = 2

# EPOCHS = 10

# LEARNING_RATE = 1e-4

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("Device:", DEVICE)


# # ============================================================
# # Dataset
# # ============================================================

# # train_dataset = VideoFrameDataset(
# #     csv_file="data/processed/train.csv",
# #     frames_root="data/processed/frames",
# #     num_frames=NUM_FRAMES,
# #     image_size=IMAGE_SIZE,
# #     training=True
# # )

# # val_dataset = VideoFrameDataset(
# #     csv_file="data/processed/val.csv",
# #     frames_root="data/processed/frames",
# #     num_frames=NUM_FRAMES,
# #     image_size=IMAGE_SIZE,
# #     training=False
# # )

# train_dataset = VideoFrameDataset(
#     csv_file="data/metadata/train.csv",
#     frames_root="data/processed/train/frames",
#     num_frames=NUM_FRAMES,
#     image_size=IMAGE_SIZE,
#     training=True
# )

# val_dataset = VideoFrameDataset(
#     csv_file="data/metadata/test.csv",
#     frames_root="data/processed/test/frames",
#     num_frames=NUM_FRAMES,
#     image_size=IMAGE_SIZE,
#     training=False
# )


# print("Training samples:", len(train_dataset))
# print("Validation samples:", len(val_dataset))


# # ============================================================
# # DataLoaders
# # ============================================================

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0
# )


# # ============================================================
# # Model
# # ============================================================

# model = VideoDeepfakeModel()

# model.to(DEVICE)


# # ============================================================
# # Loss
# # ============================================================

# criterion = nn.CrossEntropyLoss()


# # ============================================================
# # Optimizer
# # ============================================================

# optimizer = torch.optim.AdamW(
#     filter(
#         lambda p: p.requires_grad,
#         model.parameters()
#     ),
#     lr=LEARNING_RATE,
#     weight_decay=1e-5
# )


# # ============================================================
# # Training function
# # ============================================================

# def train_one_epoch():

#     model.train()

#     total_loss = 0

#     correct = 0

#     total = 0

#     for frames, labels in train_loader:

#         frames = frames.to(DEVICE)

#         labels = labels.to(DEVICE)

#         optimizer.zero_grad()

#         outputs = model(frames)

#         loss = criterion(
#             outputs,
#             labels
#         )

#         loss.backward()

#         optimizer.step()

#         total_loss += loss.item()

#         predictions = torch.argmax(
#             outputs,
#             dim=1
#         )

#         correct += (
#             predictions == labels
#         ).sum().item()

#         total += labels.size(0)

#     accuracy = correct / total

#     return total_loss / len(train_loader), accuracy


# # ============================================================
# # Validation function
# # ============================================================

# def validate():

#     model.eval()

#     total_loss = 0

#     correct = 0

#     total = 0

#     with torch.no_grad():

#         for frames, labels in val_loader:

#             frames = frames.to(DEVICE)

#             labels = labels.to(DEVICE)

#             outputs = model(frames)

#             loss = criterion(
#                 outputs,
#                 labels
#             )

#             total_loss += loss.item()

#             predictions = torch.argmax(
#                 outputs,
#                 dim=1
#             )

#             correct += (
#                 predictions == labels
#             ).sum().item()

#             total += labels.size(0)

#     accuracy = correct / total

#     return total_loss / len(val_loader), accuracy


# # ============================================================
# # Training loop
# # ============================================================

# best_val_accuracy = 0.0


# for epoch in range(EPOCHS):

#     train_loss, train_accuracy = train_one_epoch()

#     val_loss, val_accuracy = validate()

#     print(
#         f"\nEpoch {epoch + 1}/{EPOCHS}"
#     )

#     print(
#         f"Train Loss: {train_loss:.4f}"
#     )

#     print(
#         f"Train Accuracy: "
#         f"{train_accuracy * 100:.2f}%"
#     )

#     print(
#         f"Val Loss: {val_loss:.4f}"
#     )

#     print(
#         f"Val Accuracy: "
#         f"{val_accuracy * 100:.2f}%"
#     )

#     # Save best model

#     if val_accuracy > best_val_accuracy:

#         best_val_accuracy = val_accuracy

#         torch.save(
#             {
#                 "model_state_dict":
#                     model.state_dict(),

#                 "val_accuracy":
#                     val_accuracy,

#                 "epoch":
#                     epoch + 1
#             },
#             "checkpoints/video_deepfake/"
#             "efficientnet_b0_best.pth"
#         )

#         print(
#             "✓ Best model saved"
#         )


# print("\nTraining completed.")

# print(
#     "Best validation accuracy:",
#     f"{best_val_accuracy * 100:.2f}%"
# )





# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parent.parent))

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader

# from train.dataset import VideoFrameDataset
# from models.video_model import VideoDeepfakeModel


# # ============================================================
# # CONFIGURATION
# # ============================================================

# NUM_FRAMES = 16
# IMAGE_SIZE = 160

# BATCH_SIZE = 2
# EPOCHS = 15

# LEARNING_RATE = 5e-5
# WEIGHT_DECAY = 1e-4

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("=" * 60)
# print("VIDEO DEEPFAKE MODEL TRAINING")
# print("=" * 60)
# print("Device:", DEVICE)
# print("Frames:", NUM_FRAMES)
# print("Image size:", IMAGE_SIZE)
# print("Batch size:", BATCH_SIZE)
# print("Epochs:", EPOCHS)
# print("Learning rate:", LEARNING_RATE)
# print("=" * 60)


# # ============================================================
# # DATASET
# # ============================================================

# train_dataset = VideoFrameDataset(
#     csv_file="data/metadata/train.csv",
#     frames_root="data/processed/train/frames",
#     num_frames=NUM_FRAMES,
#     image_size=IMAGE_SIZE,
#     training=True
# )

# val_dataset = VideoFrameDataset(
#     csv_file="data/metadata/test.csv",
#     frames_root="data/processed/test/frames",
#     num_frames=NUM_FRAMES,
#     image_size=IMAGE_SIZE,
#     training=False
# )

# print("Training samples:", len(train_dataset))
# print("Validation samples:", len(val_dataset))


# # ============================================================
# # DATALOADERS
# # ============================================================

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=0
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=0
# )


# # ============================================================
# # MODEL
# # ============================================================

# model = VideoDeepfakeModel()

# model.to(DEVICE)


# # ============================================================
# # LOSS
# # ============================================================

# criterion = nn.CrossEntropyLoss()


# # ============================================================
# # OPTIMIZER
# # ============================================================

# optimizer = torch.optim.AdamW(
#     filter(
#         lambda p: p.requires_grad,
#         model.parameters()
#     ),
#     lr=LEARNING_RATE,
#     weight_decay=WEIGHT_DECAY
# )


# # ============================================================
# # LEARNING RATE SCHEDULER
# # ============================================================

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="max",
#     factor=0.5,
#     patience=2
# )


# # ============================================================
# # TRAIN
# # ============================================================

# def train_one_epoch():

#     model.train()

#     total_loss = 0.0
#     correct = 0
#     total = 0

#     for frames, labels in train_loader:

#         frames = frames.to(DEVICE)
#         labels = labels.to(DEVICE)

#         optimizer.zero_grad()

#         outputs = model(frames)

#         loss = criterion(
#             outputs,
#             labels
#         )

#         loss.backward()

#         # Prevent exploding gradients
#         torch.nn.utils.clip_grad_norm_(
#             model.parameters(),
#             max_norm=1.0
#         )

#         optimizer.step()

#         total_loss += loss.item()

#         predictions = torch.argmax(
#             outputs,
#             dim=1
#         )

#         correct += (
#             predictions == labels
#         ).sum().item()

#         total += labels.size(0)

#     accuracy = correct / total

#     return (
#         total_loss / len(train_loader),
#         accuracy
#     )


# # ============================================================
# # VALIDATION
# # ============================================================

# def validate():

#     model.eval()

#     total_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():

#         for frames, labels in val_loader:

#             frames = frames.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(frames)

#             loss = criterion(
#                 outputs,
#                 labels
#             )

#             total_loss += loss.item()

#             predictions = torch.argmax(
#                 outputs,
#                 dim=1
#             )

#             correct += (
#                 predictions == labels
#             ).sum().item()

#             total += labels.size(0)

#     accuracy = correct / total

#     return (
#         total_loss / len(val_loader),
#         accuracy
#     )


# # ============================================================
# # TRAINING LOOP
# # ============================================================

# best_val_accuracy = 0.0

# for epoch in range(EPOCHS):

#     train_loss, train_accuracy = train_one_epoch()

#     val_loss, val_accuracy = validate()

#     scheduler.step(val_accuracy)

#     current_lr = optimizer.param_groups[0]["lr"]

#     print()
#     print("-" * 60)
#     print(
#         f"Epoch {epoch + 1}/{EPOCHS}"
#     )

#     print(
#         f"Train Loss: {train_loss:.4f}"
#     )

#     print(
#         f"Train Accuracy: "
#         f"{train_accuracy * 100:.2f}%"
#     )

#     print(
#         f"Val Loss: {val_loss:.4f}"
#     )

#     print(
#         f"Val Accuracy: "
#         f"{val_accuracy * 100:.2f}%"
#     )

#     print(
#         f"Learning Rate: {current_lr:.7f}"
#     )

#     # --------------------------------------------------------
#     # Save best model
#     # --------------------------------------------------------

#     if val_accuracy > best_val_accuracy:

#         best_val_accuracy = val_accuracy

#         checkpoint_dir = Path(
#             "checkpoints/video_deepfake"
#         )

#         checkpoint_dir.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         torch.save(
#             {
#                 "model_state_dict":
#                     model.state_dict(),

#                 "val_accuracy":
#                     val_accuracy,

#                 "epoch":
#                     epoch + 1
#             },
#             checkpoint_dir /
#             "efficientnet_b0_best.pth"
#         )

#         print("✓ Best model saved")


# # ============================================================
# # COMPLETE
# # ============================================================

# print()
# print("=" * 60)
# print("TRAINING COMPLETED")
# print("=" * 60)

# print(
#     "Best validation accuracy:",
#     f"{best_val_accuracy * 100:.2f}%"
# )

# print("=" * 60)



import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.dataset import VideoFrameDataset
from models.video_model import VideoDeepfakeModel


# ============================================================
# CONFIGURATION
# ============================================================

NUM_FRAMES = 16
IMAGE_SIZE = 160

BATCH_SIZE = 2

EPOCHS = 15

# Small LR for pretrained EfficientNet
BACKBONE_LR = 1e-5

# Larger LR for GRU + classifier
HEAD_LR = 5e-5

WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("VIDEO DEEPFAKE MODEL FINE-TUNING")
print("=" * 60)

print("Device:", DEVICE)
print("Frames:", NUM_FRAMES)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

train_dataset = VideoFrameDataset(
    csv_file="data/metadata/train.csv",
    frames_root="data/processed/train/frames",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    training=True
)

val_dataset = VideoFrameDataset(
    csv_file="data/metadata/test.csv",
    frames_root="data/processed/test/frames",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    training=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = VideoDeepfakeModel()

model.to(DEVICE)


# ============================================================
# FREEZE EVERYTHING FIRST
# ============================================================

for param in model.backbone.parameters():
    param.requires_grad = False


# ============================================================
# UNFREEZE LAST EFFICIENTNET BLOCKS
# ============================================================

print()
print("Unfreezing final EfficientNet blocks...")

for param in model.backbone.features[-2:].parameters():
    param.requires_grad = True


# GRU + classifier are always trainable

for param in model.temporal_gru.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True


# ============================================================
# SHOW TRAINABLE PARAMETERS
# ============================================================

trainable = 0
total = 0

for param in model.parameters():

    total += param.numel()

    if param.requires_grad:
        trainable += param.numel()

print()
print("Total parameters:", total)
print("Trainable parameters:", trainable)

print(
    "Trainable percentage:",
    f"{100 * trainable / total:.2f}%"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    [

        {
            "params":
                model.backbone.features[-2:].parameters(),

            "lr":
                BACKBONE_LR
        },

        {
            "params":
                model.temporal_gru.parameters(),

            "lr":
                HEAD_LR
        },

        {
            "params":
                model.classifier.parameters(),

            "lr":
                HEAD_LR
        }

    ],

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for frames, labels in train_loader:

        frames = frames.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    return (
        total_loss / len(train_loader),
        accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    with torch.no_grad():

        for frames, labels in val_loader:

            frames = frames.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(frames)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return (
        total_loss / len(val_loader),
        accuracy
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

checkpoint_dir = Path(
    "checkpoints/video_deepfake"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = (
    checkpoint_dir /
    "efficientnet_b0_finetuned_best.pth"
)


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print("-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Backbone LR: {current_lr:.7f}"
    )

    # --------------------------------------------------------
    # Save best
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1
            },
            checkpoint_path
        )

        print("✓ Best fine-tuned model saved")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("FINE-TUNING COMPLETED")
print("=" * 60)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Checkpoint:",
    checkpoint_path
)

print("=" * 60)