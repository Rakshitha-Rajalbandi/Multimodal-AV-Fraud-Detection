import csv
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PREDICTIONS_FILE = Path(
    "results/evaluation/test_predictions.csv"
)

FRAMES_ROOT = Path(
    "data/processed/test/frames"
)

OUTPUT_DIR = Path(
    "results/evaluation/misclassified"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Read predictions
# ------------------------------------------------------------

with open(
    PREDICTIONS_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    rows = list(reader)


# ------------------------------------------------------------
# Find misclassified videos
# ------------------------------------------------------------

misclassified = []

for row in rows:

    true_label = row["true_label"].strip().upper()
    predicted_label = row["predicted_label"].strip().upper()

    if true_label != predicted_label:

        misclassified.append(
            {
                "video_id": row["video_id"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "fake_probability": row["fake_probability"]
            }
        )


# ------------------------------------------------------------
# Count errors
# ------------------------------------------------------------

false_negatives = sum(
    1
    for x in misclassified
    if x["true_label"] == "FAKE"
    and x["predicted_label"] == "REAL"
)

false_positives = sum(
    1
    for x in misclassified
    if x["true_label"] == "REAL"
    and x["predicted_label"] == "FAKE"
)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("MISCLASSIFIED VIDEO ANALYSIS")
print("=" * 60)

print(
    "Total test videos:",
    len(rows)
)

print(
    "Misclassified:",
    len(misclassified)
)

print(
    "FAKE -> REAL:",
    false_negatives
)

print(
    "REAL -> FAKE:",
    false_positives
)

print("=" * 60)


# ------------------------------------------------------------
# Save representative frames
# ------------------------------------------------------------

for item in misclassified:

    video_id = item["video_id"]

    frame_dir = (
        FRAMES_ROOT /
        video_id
    )

    if not frame_dir.exists():

        print(
            "Frames not found:",
            video_id
        )

        continue

    frames = sorted(
        frame_dir.glob("*.jpg")
    )

    if not frames:
        continue

    # First, middle and last frame
    indices = [
        0,
        len(frames) // 2,
        len(frames) - 1
    ]

    output_video_dir = (
        OUTPUT_DIR /
        f"{video_id}_{item['true_label']}_"
        f"pred_{item['predicted_label']}"
    )

    output_video_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index in indices:

        source = frames[index]

        destination = (
            output_video_dir /
            source.name
        )

        destination.write_bytes(
            source.read_bytes()
        )


# ------------------------------------------------------------
# Save misclassified CSV
# ------------------------------------------------------------

report_file = (
    OUTPUT_DIR /
    "misclassified.csv"
)

with open(
    report_file,
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
        misclassified
    )


# ------------------------------------------------------------
# Final output
# ------------------------------------------------------------

print()
print("Saved representative frames to:")

print(
    OUTPUT_DIR
)

print()
print("Saved misclassified report:")

print(
    report_file
)

print()
print("Misclassified video IDs:")

for item in misclassified:

    print(
        f"{item['video_id']} | "
        f"Actual: {item['true_label']} | "
        f"Predicted: {item['predicted_label']} | "
        f"Fake probability: "
        f"{float(item['fake_probability']):.4f}"
    )