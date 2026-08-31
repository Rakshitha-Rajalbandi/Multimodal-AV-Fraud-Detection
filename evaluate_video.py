import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from train.dataset import VideoFrameDataset
from models.video_model import VideoDeepfakeModel


# ============================================================
# CONFIG
# ============================================================

NUM_FRAMES = 16
IMAGE_SIZE = 160
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    Path("checkpoints")
    / "video_deepfake"
    / "efficientnet_b0_best.pth"
)

CSV_FILE = Path("data/metadata/test.csv")

FRAMES_ROOT = Path(
    "data/processed/test/frames"
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("VIDEO MODEL EVALUATION")
print("=" * 60)

print("Device:", DEVICE)
print("Frames:", NUM_FRAMES)
print("Image size:", IMAGE_SIZE)
print("Checkpoint:", CHECKPOINT)

print("=" * 60)


# ============================================================
# DATASET
# ============================================================

dataset = VideoFrameDataset(
    csv_file=CSV_FILE,
    frames_root=FRAMES_ROOT,
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    training=False
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(dataset))


# ============================================================
# MODEL
# ============================================================

model = VideoDeepfakeModel()

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_fake_probabilities = []


with torch.no_grad():

    for frames, labels in loader:

        frames = frames.to(DEVICE)

        outputs = model(frames)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        fake_probabilities = (
            probabilities[:, 0]
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_fake_probabilities.extend(
            fake_probabilities.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1]
)

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    "                Predicted"
)

print(
    "              FAKE   REAL"
)

print(
    f"Actual FAKE   {cm[0][0]:4d}   {cm[0][1]:4d}"
)

print(
    f"Actual REAL   {cm[1][0]:4d}   {cm[1][1]:4d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "FAKE",
            "REAL"
        ]
    )
)


# ============================================================
# MISCLASSIFIED VIDEOS
# ============================================================

metadata = pd.read_csv(
    CSV_FILE
)

print()
print("=" * 60)
print("MISCLASSIFIED VIDEOS")
print("=" * 60)

for i in range(len(all_labels)):

    actual = all_labels[i]

    predicted = all_predictions[i]

    if actual != predicted:

        video_id = metadata.iloc[i]["video_id"]

        fake_probability = (
            all_fake_probabilities[i]
        )

        actual_name = (
            "FAKE"
            if actual == 0
            else "REAL"
        )

        predicted_name = (
            "FAKE"
            if predicted == 0
            else "REAL"
        )

        print(
            f"{video_id} | "
            f"Actual: {actual_name} | "
            f"Predicted: {predicted_name} | "
            f"Fake probability: "
            f"{fake_probability * 100:.2f}%"
        )


print()
print("=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)




# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

import numpy as np

print()
print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

best_accuracy = 0
best_threshold = 0

for threshold in np.arange(0.30, 0.81, 0.05):

    threshold_predictions = [
        0 if p >= threshold else 1
        for p in all_fake_probabilities
    ]

    acc = accuracy_score(
        all_labels,
        threshold_predictions
    )

    precision = precision_score(
        all_labels,
        threshold_predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        threshold_predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        threshold_predictions,
        pos_label=0,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Accuracy: {acc * 100:.2f}% | "
        f"Precision: {precision * 100:.2f}% | "
        f"Recall: {recall * 100:.2f}% | "
        f"F1: {f1 * 100:.2f}%"
    )

    if acc > best_accuracy:

        best_accuracy = acc
        best_threshold = threshold


print()
print(
    f"Best threshold by accuracy: "
    f"{best_threshold:.2f}"
)

print(
    f"Best threshold accuracy: "
    f"{best_accuracy * 100:.2f}%"
)