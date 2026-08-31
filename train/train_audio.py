import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train.audio_mel_dataset import AudioMelDataset
from models.audio_model import AudioDeepfakeModel


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

train_dataset = AudioMelDataset(
    csv_file="data/metadata/train.csv",
    audio_root="data/processed/train/audio"
)

test_dataset = AudioMelDataset(
    csv_file="data/metadata/test.csv",
    audio_root="data/processed/test/audio"
)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = AudioDeepfakeModel()

model.to(DEVICE)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)


# ============================================================
# Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for mel, labels in train_loader:

        mel = mel.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(mel)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

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
# Evaluation
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for mel, labels in test_loader:

            mel = mel.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(mel)

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
        total_loss / len(test_loader),
        accuracy
    )


# ============================================================
# Training loop
# ============================================================

best_accuracy = 0.0


for epoch in range(EPOCHS):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    test_loss, test_accuracy = (
        evaluate()
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "test_accuracy":
                    test_accuracy,

                "epoch":
                    epoch + 1
            },
            "checkpoints/audio_deepfake_best.pth"
        )

        print("✓ Best audio model saved")


print("\nAudio training completed.")

print(
    "Best audio accuracy:",
    f"{best_accuracy * 100:.2f}%"
)