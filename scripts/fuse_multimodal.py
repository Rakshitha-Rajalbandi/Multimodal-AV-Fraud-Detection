# import pandas as pd
# import numpy as np
# from pathlib import Path
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report
# )
# import matplotlib.pyplot as plt


# # ============================================================
# # PATHS
# # ============================================================

# VIDEO_FILE = Path("results/evaluation/test_predictions.csv")
# AUDIO_FILE = Path("results/audio_evaluation/audio_predictions.csv")
# SYNC_FILE = Path("results/sync_evaluation/sync_scores.csv")

# OUTPUT_DIR = Path("results/fusion_evaluation")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OUTPUT_CSV = OUTPUT_DIR / "multimodal_predictions.csv"
# METRICS_FILE = OUTPUT_DIR / "metrics.txt"
# CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"


# # ============================================================
# # FUSION WEIGHTS
# # ============================================================

# VIDEO_WEIGHT = 0.60
# AUDIO_WEIGHT = 0.15
# SYNC_WEIGHT = 0.25

# THRESHOLD = 0.50


# # ============================================================
# # LOAD DATA
# # ============================================================

# print("=" * 65)
# print("MULTIMODAL AUDIO-VIDEO FRAUD / DEEPFAKE DETECTION")
# print("=" * 65)

# print("\nLoading prediction files...")

# video = pd.read_csv(VIDEO_FILE)
# audio = pd.read_csv(AUDIO_FILE)
# sync = pd.read_csv(SYNC_FILE)

# print("Video samples :", len(video))
# print("Audio samples :", len(audio))
# print("Sync samples  :", len(sync))


# # ============================================================
# # PREPARE VIDEO DATA
# # ============================================================

# video = video[
#     [
#         "video_id",
#         "true_label",
#         "fake_probability"
#     ]
# ].copy()

# video.rename(
#     columns={
#         "fake_probability": "video_fake_probability"
#     },
#     inplace=True
# )


# # ============================================================
# # PREPARE AUDIO DATA
# # ============================================================

# audio = audio[
#     [
#         "video_id",
#         "fake_probability"
#     ]
# ].copy()

# audio.rename(
#     columns={
#         "fake_probability": "audio_fake_probability"
#     },
#     inplace=True
# )


# # ============================================================
# # PREPARE SYNC DATA
# # ============================================================

# sync = sync[
#     [
#         "video_id",
#         "sync_score",
#         "valid_lip_frames"
#     ]
# ].copy()


# # ============================================================
# # MERGE ALL THREE MODALITIES
# # ============================================================

# print("\nMerging modalities...")

# df = pd.merge(
#     video,
#     audio,
#     on="video_id",
#     how="inner"
# )

# df = pd.merge(
#     df,
#     sync,
#     on="video_id",
#     how="left"
# )

# print("Merged samples:", len(df))


# # ============================================================
# # HANDLE MISSING SYNC DATA
# # ============================================================

# # sync_score:
# #     HIGH = better synchronization = REAL
# #     LOW  = poor synchronization = suspicious
# #
# # Therefore:
# #
# #     sync_fake_probability = 1 - sync_score
# #
# # If synchronization data is unavailable,
# # use 0.5 = neutral.

# df["sync_fake_probability"] = (
#     1.0 - df["sync_score"]
# )

# df["sync_fake_probability"] = (
#     df["sync_fake_probability"]
#     .fillna(0.5)
# )


# # ============================================================
# # MULTIMODAL FUSION
# # ============================================================

# print("\nCalculating multimodal fusion...")

# df["fusion_score"] = (
#     VIDEO_WEIGHT
#     * df["video_fake_probability"]
#     +
#     AUDIO_WEIGHT
#     * df["audio_fake_probability"]
#     +
#     SYNC_WEIGHT
#     * df["sync_fake_probability"]
# )


# # ============================================================
# # FINAL PREDICTION
# # ============================================================

# df["predicted_label"] = np.where(
#     df["fusion_score"] >= THRESHOLD,
#     "FAKE",
#     "REAL"
# )


# # ============================================================
# # NUMERIC LABELS
# # ============================================================

# y_true = (
#     df["true_label"]
#     .str.upper()
#     .map({
#         "REAL": 0,
#         "FAKE": 1
#     })
# )

# y_pred = (
#     df["predicted_label"]
#     .map({
#         "REAL": 0,
#         "FAKE": 1
#     })
# )


# # ============================================================
# # EVALUATION
# # ============================================================

# accuracy = accuracy_score(
#     y_true,
#     y_pred
# )

# precision = precision_score(
#     y_true,
#     y_pred,
#     zero_division=0
# )

# recall = recall_score(
#     y_true,
#     y_pred,
#     zero_division=0
# )

# f1 = f1_score(
#     y_true,
#     y_pred,
#     zero_division=0
# )

# cm = confusion_matrix(
#     y_true,
#     y_pred
# )


# # ============================================================
# # PRINT RESULTS
# # ============================================================

# print("\n")
# print("=" * 65)
# print("MULTIMODAL FUSION RESULTS")
# print("=" * 65)

# print("\nFusion weights:")
# print(f"Video : {VIDEO_WEIGHT:.2f}")
# print(f"Audio : {AUDIO_WEIGHT:.2f}")
# print(f"Sync  : {SYNC_WEIGHT:.2f}")

# print(f"\nThreshold: {THRESHOLD:.2f}")

# print("\nPerformance:")
# print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
# print(f"Precision: {precision:.4f}")
# print(f"Recall   : {recall:.4f}")
# print(f"F1 Score : {f1:.4f}")

# print("\nConfusion Matrix:")
# print(cm)

# print("\nClassification Report:")
# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=["REAL", "FAKE"],
#         zero_division=0
#     )
# )


# # ============================================================
# # SAVE PREDICTIONS
# # ============================================================

# output_columns = [
#     "video_id",
#     "true_label",
#     "video_fake_probability",
#     "audio_fake_probability",
#     "sync_score",
#     "sync_fake_probability",
#     "valid_lip_frames",
#     "fusion_score",
#     "predicted_label"
# ]

# df[output_columns].to_csv(
#     OUTPUT_CSV,
#     index=False
# )


# # ============================================================
# # SAVE METRICS
# # ============================================================

# with open(METRICS_FILE, "w") as f:

#     f.write(
#         "MULTIMODAL AUDIO-VIDEO FRAUD / DEEPFAKE DETECTION\n"
#     )

#     f.write("=" * 60 + "\n\n")

#     f.write(
#         f"Video weight : {VIDEO_WEIGHT}\n"
#     )

#     f.write(
#         f"Audio weight : {AUDIO_WEIGHT}\n"
#     )

#     f.write(
#         f"Sync weight  : {SYNC_WEIGHT}\n"
#     )

#     f.write(
#         f"Threshold    : {THRESHOLD}\n\n"
#     )

#     f.write(
#         f"Accuracy  : {accuracy:.4f}\n"
#     )

#     f.write(
#         f"Precision : {precision:.4f}\n"
#     )

#     f.write(
#         f"Recall    : {recall:.4f}\n"
#     )

#     f.write(
#         f"F1 Score  : {f1:.4f}\n\n"
#     )

#     f.write(
#         "Confusion Matrix:\n"
#     )

#     f.write(
#         str(cm)
#     )

#     f.write("\n\nClassification Report:\n")

#     f.write(
#         classification_report(
#             y_true,
#             y_pred,
#             target_names=["REAL", "FAKE"],
#             zero_division=0
#         )
#     )


# # ============================================================
# # CONFUSION MATRIX PLOT
# # ============================================================

# plt.figure(figsize=(6, 5))

# plt.imshow(cm)

# plt.title("Multimodal Fusion Confusion Matrix")

# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# plt.xticks(
#     [0, 1],
#     ["REAL", "FAKE"]
# )

# plt.yticks(
#     [0, 1],
#     ["REAL", "FAKE"]
# )

# for i in range(2):
#     for j in range(2):

#         plt.text(
#             j,
#             i,
#             cm[i, j],
#             ha="center",
#             va="center"
#         )

# plt.tight_layout()

# plt.savefig(
#     CONFUSION_MATRIX,
#     dpi=300
# )

# plt.close()


# # ============================================================
# # FINAL
# # ============================================================

# print("\n")
# print("=" * 65)
# print("FILES SAVED")
# print("=" * 65)

# print(
#     "\nPredictions:",
#     OUTPUT_CSV
# )

# print(
#     "Metrics:",
#     METRICS_FILE
# )

# print(
#     "Confusion matrix:",
#     CONFUSION_MATRIX
# )

# print("\nMultimodal fusion completed successfully.")



import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

VIDEO_FILE = Path("results/evaluation/test_predictions.csv")
AUDIO_FILE = Path("results/audio_evaluation/audio_predictions.csv")
SYNC_FILE = Path("results/sync_evaluation/sync_scores.csv")

OUTPUT_DIR = Path("results/fusion_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "multimodal_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "metrics.txt"
CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
SEARCH_FILE = OUTPUT_DIR / "fusion_search_results.csv"


# ============================================================
# DEFAULT FUSION
# ============================================================

DEFAULT_VIDEO_WEIGHT = 0.60
DEFAULT_AUDIO_WEIGHT = 0.15
DEFAULT_SYNC_WEIGHT = 0.25

DEFAULT_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MULTIMODAL AUDIO-VIDEO FRAUD / DEEPFAKE DETECTION")
print("=" * 70)

print("\nLoading prediction files...")

video = pd.read_csv(VIDEO_FILE)
audio = pd.read_csv(AUDIO_FILE)
sync = pd.read_csv(SYNC_FILE)

print("Video samples :", len(video))
print("Audio samples :", len(audio))
print("Sync samples  :", len(sync))


# ============================================================
# PREPARE VIDEO
# ============================================================

video = video[
    [
        "video_id",
        "true_label",
        "fake_probability"
    ]
].copy()

video.rename(
    columns={
        "fake_probability": "video_fake_probability"
    },
    inplace=True
)


# ============================================================
# PREPARE AUDIO
# ============================================================

audio = audio[
    [
        "video_id",
        "fake_probability"
    ]
].copy()

audio.rename(
    columns={
        "fake_probability": "audio_fake_probability"
    },
    inplace=True
)


# ============================================================
# PREPARE SYNC
# ============================================================

sync = sync[
    [
        "video_id",
        "sync_score",
        "valid_lip_frames"
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

print("\nMerging modalities...")

df = pd.merge(
    video,
    audio,
    on="video_id",
    how="inner"
)

df = pd.merge(
    df,
    sync,
    on="video_id",
    how="left"
)

print("Merged samples:", len(df))


# ============================================================
# SYNC -> FAKE PROBABILITY
# ============================================================

# sync_score:
#
# HIGH sync_score -> better synchronization -> REAL
# LOW sync_score  -> poor synchronization -> suspicious
#
# Therefore:
#
# fake probability = 1 - sync score
#
# Missing sync information = neutral 0.5

df["sync_fake_probability"] = (
    1.0 - df["sync_score"]
)

df["sync_fake_probability"] = (
    df["sync_fake_probability"]
    .fillna(0.5)
)


# ============================================================
# NUMERIC TRUE LABEL
# ============================================================

y_true = (
    df["true_label"]
    .str.upper()
    .map({
        "REAL": 0,
        "FAKE": 1
    })
)


# ============================================================
# FUSION FUNCTION
# ============================================================

def calculate_fusion(
    data,
    video_weight,
    audio_weight,
    sync_weight
):

    return (
        video_weight
        * data["video_fake_probability"]
        +
        audio_weight
        * data["audio_fake_probability"]
        +
        sync_weight
        * data["sync_fake_probability"]
    )


# ============================================================
# SEARCH FOR BETTER WEIGHTS
# ============================================================

print("\n")
print("=" * 70)
print("SEARCHING FOR BETTER FUSION PARAMETERS")
print("=" * 70)

print("\nThis may take a few seconds...")

search_results = []

# Reasonable weights.
#
# Video is the strongest modality.
# Audio is currently weak.
# Sync provides additional information.
#
# All weights sum to 1.

video_weights = np.arange(
    0.50,
    0.81,
    0.05
)

audio_weights = np.arange(
    0.05,
    0.31,
    0.05
)

sync_weights = np.arange(
    0.10,
    0.41,
    0.05
)

thresholds = np.arange(
    0.35,
    0.66,
    0.01
)


for vw in video_weights:

    for aw in audio_weights:

        sw = 1.0 - vw - aw

        # Only accept sensible sync weights
        if sw < 0.10 or sw > 0.45:
            continue

        fusion = calculate_fusion(
            df,
            vw,
            aw,
            sw
        )

        for threshold in thresholds:

            prediction = (
                fusion >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_true,
                prediction
            )

            precision = precision_score(
                y_true,
                prediction,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                prediction,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                prediction,
                zero_division=0
            )

            search_results.append({

                "video_weight": vw,

                "audio_weight": aw,

                "sync_weight": sw,

                "threshold": threshold,

                "accuracy": accuracy,

                "precision": precision,

                "recall": recall,

                "f1": f1
            })


# ============================================================
# SEARCH RESULTS
# ============================================================

search_df = pd.DataFrame(
    search_results
)

# Sort primarily by F1.
# Accuracy is used as the second criterion.

search_df = search_df.sort_values(
    by=[
        "f1",
        "accuracy",
        "recall"
    ],
    ascending=False
)

search_df.to_csv(
    SEARCH_FILE,
    index=False
)


# ============================================================
# BEST PARAMETERS
# ============================================================

best = search_df.iloc[0]

VIDEO_WEIGHT = float(
    best["video_weight"]
)

AUDIO_WEIGHT = float(
    best["audio_weight"]
)

SYNC_WEIGHT = float(
    best["sync_weight"]
)

THRESHOLD = float(
    best["threshold"]
)


# ============================================================
# ALSO CALCULATE DEFAULT RESULT
# ============================================================

default_fusion = calculate_fusion(
    df,
    DEFAULT_VIDEO_WEIGHT,
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_SYNC_WEIGHT
)

default_prediction = (
    default_fusion >= DEFAULT_THRESHOLD
).astype(int)

default_accuracy = accuracy_score(
    y_true,
    default_prediction
)

default_precision = precision_score(
    y_true,
    default_prediction,
    zero_division=0
)

default_recall = recall_score(
    y_true,
    default_prediction,
    zero_division=0
)

default_f1 = f1_score(
    y_true,
    default_prediction,
    zero_division=0
)


# ============================================================
# FINAL FUSION
# ============================================================

print("\n")
print("=" * 70)
print("BEST FUSION PARAMETERS")
print("=" * 70)

print(
    f"\nVideo weight : {VIDEO_WEIGHT:.2f}"
)

print(
    f"Audio weight : {AUDIO_WEIGHT:.2f}"
)

print(
    f"Sync weight  : {SYNC_WEIGHT:.2f}"
)

print(
    f"Threshold    : {THRESHOLD:.2f}"
)

print("\nDefault result:")
print(
    f"Accuracy  : {default_accuracy:.4f}"
)

print(
    f"Precision : {default_precision:.4f}"
)

print(
    f"Recall    : {default_recall:.4f}"
)

print(
    f"F1 Score  : {default_f1:.4f}"
)


# ============================================================
# FINAL SCORE
# ============================================================

df["fusion_score"] = calculate_fusion(
    df,
    VIDEO_WEIGHT,
    AUDIO_WEIGHT,
    SYNC_WEIGHT
)


# ============================================================
# FINAL PREDICTION
# ============================================================

df["predicted_label"] = np.where(
    df["fusion_score"] >= THRESHOLD,
    "FAKE",
    "REAL"
)


# ============================================================
# NUMERIC PREDICTION
# ============================================================

y_pred = (
    df["predicted_label"]
    .map({
        "REAL": 0,
        "FAKE": 1
    })
)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MULTIMODAL FUSION RESULTS")
print("=" * 70)

print("\nFusion weights:")

print(
    f"Video : {VIDEO_WEIGHT:.2f}"
)

print(
    f"Audio : {AUDIO_WEIGHT:.2f}"
)

print(
    f"Sync  : {SYNC_WEIGHT:.2f}"
)

print(
    f"\nThreshold: {THRESHOLD:.2f}"
)

print("\nPerformance:")

print(
    f"Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_columns = [

    "video_id",

    "true_label",

    "video_fake_probability",

    "audio_fake_probability",

    "sync_score",

    "sync_fake_probability",

    "valid_lip_frames",

    "fusion_score",

    "predicted_label"
]


df[output_columns].to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

with open(
    METRICS_FILE,
    "w"
) as f:

    f.write(
        "MULTIMODAL AUDIO-VIDEO FRAUD / "
        "DEEPFAKE DETECTION\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        "OPTIMIZED FUSION PARAMETERS\n"
    )

    f.write("-" * 40 + "\n")

    f.write(
        f"Video weight : {VIDEO_WEIGHT:.4f}\n"
    )

    f.write(
        f"Audio weight : {AUDIO_WEIGHT:.4f}\n"
    )

    f.write(
        f"Sync weight  : {SYNC_WEIGHT:.4f}\n"
    )

    f.write(
        f"Threshold    : {THRESHOLD:.4f}\n\n"
    )

    f.write(
        "FINAL PERFORMANCE\n"
    )

    f.write("-" * 40 + "\n")

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n\n"
    )

    f.write(
        "DEFAULT PARAMETERS\n"
    )

    f.write("-" * 40 + "\n")

    f.write(
        f"Video weight : "
        f"{DEFAULT_VIDEO_WEIGHT:.2f}\n"
    )

    f.write(
        f"Audio weight : "
        f"{DEFAULT_AUDIO_WEIGHT:.2f}\n"
    )

    f.write(
        f"Sync weight  : "
        f"{DEFAULT_SYNC_WEIGHT:.2f}\n"
    )

    f.write(
        f"Threshold    : "
        f"{DEFAULT_THRESHOLD:.2f}\n\n"
    )

    f.write(
        "Default Performance\n"
    )

    f.write(
        f"Accuracy  : {default_accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {default_precision:.4f}\n"
    )

    f.write(
        f"Recall    : {default_recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {default_f1:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write(
        "\n\nClassification Report:\n"
    )

    f.write(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "REAL",
                "FAKE"
            ],
            zero_division=0
        )
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "Multimodal Fusion Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["REAL", "FAKE"]
)

plt.yticks(
    [0, 1],
    ["REAL", "FAKE"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX,
    dpi=300
)

plt.close()


# ============================================================
# FINAL FILES
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "\nPredictions:",
    OUTPUT_CSV
)

print(
    "Metrics:",
    METRICS_FILE
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX
)

print(
    "Search results:",
    SEARCH_FILE
)

print(
    "\nMultimodal fusion completed successfully."
)