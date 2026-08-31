import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from train.dataset import VideoFrameDataset
from models.video_model import VideoDeepfakeModel


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = "checkpoints/video_deepfake/efficientnet_b0_best.pth"

TEST_CSV = "data/metadata/test.csv"

FRAMES_ROOT = "data/processed/test/frames"

OUTPUT_DIR = Path("results/evaluation")

NUM_FRAMES = 8
IMAGE_SIZE = 160
BATCH_SIZE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Dataset
# ============================================================

test_dataset = VideoFrameDataset(
    csv_file=TEST_CSV,
    frames_root=FRAMES_ROOT,
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    training=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(test_dataset))


# ============================================================
# Load model
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
print(
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation accuracy:",
    checkpoint.get("val_accuracy")
)


# ============================================================
# Evaluation
# ============================================================

y_true = []
y_pred = []
y_fake_probability = []

prediction_rows = []


sample_index = 0


with torch.no_grad():

    for frames, labels in test_loader:

        frames = frames.to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = model(frames)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Your project mapping is:
        #
        # 0 = FAKE
        # 1 = REAL
        #
        # Therefore fake probability = probabilities[:, 0]
        # ----------------------------------------------------

        fake_probabilities = probabilities[:, 0]

        for i in range(len(labels)):

            true_label = int(
                labels[i].item()
            )

            predicted_label = int(
                predictions[i].item()
            )

            fake_probability = float(
                fake_probabilities[i].item()
            )

            video_id = test_dataset.samples[
                sample_index
            ][0]

            y_true.append(true_label)

            y_pred.append(predicted_label)

            y_fake_probability.append(
                fake_probability
            )

            prediction_rows.append(
                {
                    "video_id": video_id,
                    "true_label": (
                        "FAKE"
                        if true_label == 0
                        else "REAL"
                    ),
                    "predicted_label": (
                        "FAKE"
                        if predicted_label == 0
                        else "REAL"
                    ),
                    "fake_probability":
                        fake_probability
                }
            )

            sample_index += 1


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=0,
    zero_division=0
)


# ROC-AUC
#
# Since FAKE = class 0, ROC-AUC needs a score
# where larger value means more likely FAKE.
#
try:

    auc = roc_auc_score(
        [1 if y == 0 else 0 for y in y_true],
        y_fake_probability
    )

except ValueError:

    auc = float("nan")


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

# Matrix layout:
#
#              Predicted
#              FAKE REAL
#
# Actual FAKE
# Actual REAL


# ============================================================
# Print results
# ============================================================

print()
print("=" * 55)
print("TEST SET RESULTS")
print("=" * 55)

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

print(
    f"ROC-AUC  : {auc:.4f}"
)

print()
print("Confusion Matrix")
print(cm)

print()
print("Label mapping:")
print("0 = FAKE")
print("1 = REAL")

print("=" * 55)


# ============================================================
# Save predictions CSV
# ============================================================

predictions_file = (
    OUTPUT_DIR /
    "test_predictions.csv"
)

with open(
    predictions_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "true_label",
            "predicted_label",
            "fake_probability"
        ]
    )

    writer.writeheader()

    writer.writerows(
        prediction_rows
    )


# ============================================================
# Save metrics JSON
# ============================================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "roc_auc": auc,

    "confusion_matrix": cm.tolist(),

    "test_samples": len(test_dataset),

    "label_mapping": {
        "0": "FAKE",
        "1": "REAL"
    }
}


metrics_file = (
    OUTPUT_DIR /
    "metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


print()
print("Saved:")
print(predictions_file)
print(metrics_file)