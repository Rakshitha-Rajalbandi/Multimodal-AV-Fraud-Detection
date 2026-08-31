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

from train.audio_mel_dataset import AudioMelDataset
from models.audio_model import AudioDeepfakeModel


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = (
    "checkpoints/audio_deepfake_best.pth"
)

CSV_FILE = (
    "data/metadata/test.csv"
)

AUDIO_ROOT = (
    "data/processed/test/audio"
)

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

OUTPUT_DIR = Path(
    "results/audio_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("Device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

dataset = AudioMelDataset(
    csv_file=CSV_FILE,
    audio_root=AUDIO_ROOT
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test samples:", len(dataset))


# ============================================================
# Load model
# ============================================================

model = AudioDeepfakeModel()

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
    checkpoint["epoch"]
)

print(
    "Best test accuracy:",
    f"{checkpoint['test_accuracy'] * 100:.2f}%"
)


# ============================================================
# Evaluation
# ============================================================

y_true = []
y_pred = []
y_score = []

video_ids = [
    video_id
    for video_id, label
    in dataset.samples
]


with torch.no_grad():

    index = 0

    for mel, labels in loader:

        mel = mel.to(DEVICE)

        outputs = model(mel)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        # Our mapping:
        # 0 = FAKE
        # 1 = REAL
        #
        # Therefore probability of FAKE = probability index 0

        fake_probabilities = (
            probabilities[:, 0]
        )

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predictions.cpu().tolist()
        )

        y_score.extend(
            fake_probabilities.cpu().tolist()
        )


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

auc = roc_auc_score(
    [1 if y == 0 else 0 for y in y_true],
    y_score
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


# ============================================================
# Print results
# ============================================================

print()
print("=" * 55)
print("AUDIO MODEL TEST RESULTS")
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
# Save predictions
# ============================================================

prediction_file = (
    OUTPUT_DIR /
    "audio_predictions.csv"
)

with open(
    prediction_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "video_id",
        "true_label",
        "predicted_label",
        "fake_probability"
    ])

    for i in range(len(y_true)):

        true_name = (
            "FAKE"
            if y_true[i] == 0
            else "REAL"
        )

        predicted_name = (
            "FAKE"
            if y_pred[i] == 0
            else "REAL"
        )

        writer.writerow([
            video_ids[i],
            true_name,
            predicted_name,
            y_score[i]
        ])


# ============================================================
# Save metrics
# ============================================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "roc_auc": auc,

    "confusion_matrix": cm.tolist(),

    "label_mapping": {
        "0": "FAKE",
        "1": "REAL"
    }
}


with open(
    OUTPUT_DIR / "metrics.json",
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

print(
    prediction_file
)

print(
    OUTPUT_DIR / "metrics.json"
)