import sys
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT PIPELINE
# ============================================================

from api.pipeline import (
    predict_video,
    predict_audio,
    predict_sync,
    fuse_predictions,
    extract_audio
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test"
)

LIPS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
    / "lips_highres"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "multimodal_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LABEL MAPPING
# ============================================================

# 0 = FAKE
# 1 = REAL


# ============================================================
# FIND TEST VIDEOS
# ============================================================

fake_videos = list(
    (TEST_ROOT / "FAKE").rglob("*.mp4")
)

real_videos = list(
    (TEST_ROOT / "REAL").rglob("*.mp4")
)

videos = []

for video in fake_videos:
    videos.append(
        (video, 0)
    )

for video in real_videos:
    videos.append(
        (video, 1)
    )


videos.sort(
    key=lambda x: str(x[0])
)


print("=" * 70)
print("MULTIMODAL DEEPFAKE FULL TEST EVALUATION")
print("=" * 70)

print(
    "Fake videos:",
    len(fake_videos)
)

print(
    "Real videos:",
    len(real_videos)
)

print(
    "Total videos:",
    len(videos)
)

print("=" * 70)


# ============================================================
# STORAGE
# ============================================================

y_true = []

video_pred = []
audio_pred = []
sync_pred = []
fusion_pred = []

video_score = []
audio_score = []
sync_score = []
fusion_score = []

rows = []


# ============================================================
# PROCESS VIDEOS
# ============================================================

for index, (video_path, true_label) in enumerate(
    videos,
    start=1
):

    video_id = video_path.stem

    print()
    print(
        f"[{index}/{len(videos)}] {video_id}"
    )

    print(
        "Path:",
        video_path
    )

    # --------------------------------------------------------
    # Temporary working directory
    # --------------------------------------------------------

    work_dir = (
        OUTPUT_DIR
        / "temp"
        / video_id
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ====================================================
        # VIDEO
        # ====================================================

        video_result = predict_video(
            video_path
        )

        # ====================================================
        # AUDIO
        # ====================================================

        audio_path = (
            work_dir
            / "audio.wav"
        )

        extract_audio(
            video_path,
            audio_path
        )

        audio_result = predict_audio(
            audio_path
        )

        # ====================================================
        # SYNC
        # ====================================================

        lips_path = (
            LIPS_ROOT
            / f"{video_id}.npy"
        )

        sync_result = predict_sync(
            audio_path,
            lips_path
        )

        # ====================================================
        # FUSION
        # ====================================================

        fusion_result = fuse_predictions(
            video_result,
            audio_result,
            sync_result
        )

        # ====================================================
        # LABELS
        # ====================================================

        vp = (
            0
            if video_result["prediction"] == "FAKE"
            else 1
        )

        ap = (
            0
            if audio_result["prediction"] == "FAKE"
            else 1
        )

        fp = (
            0
            if fusion_result["prediction"] == "FAKE"
            else 1
        )

        # ----------------------------------------------------
        # Sync prediction
        # ----------------------------------------------------
        #
        # sync_fake_probability >= 0.5
        # means sync signal considers it FAKE
        #

        sp = (
            0
            if sync_result["sync_fake_probability"] >= 0.5
            else 1
        )

        # ====================================================
        # SCORES
        # ====================================================

        vs = float(
            video_result["fake_probability"]
        )

        ass = float(
            audio_result["fake_probability"]
        )

        ss = float(
            sync_result["sync_fake_probability"]
        )

        fs = float(
            fusion_result["fraud_risk"]
        )

        # ====================================================
        # STORE
        # ====================================================

        y_true.append(
            true_label
        )

        video_pred.append(vp)
        audio_pred.append(ap)
        sync_pred.append(sp)
        fusion_pred.append(fp)

        video_score.append(vs)
        audio_score.append(ass)
        sync_score.append(ss)
        fusion_score.append(fs)

        rows.append(
            {
                "video_id": video_id,
                "true_label": (
                    "FAKE"
                    if true_label == 0
                    else "REAL"
                ),

                "video_prediction":
                    video_result["prediction"],

                "video_fake_probability":
                    vs,

                "audio_prediction":
                    audio_result["prediction"],

                "audio_fake_probability":
                    ass,

                "sync_available":
                    sync_result["available"],

                "sync_prediction":
                    (
                        "FAKE"
                        if sp == 0
                        else "REAL"
                    ),

                "sync_fake_probability":
                    ss,

                "sync_score":
                    sync_result["sync_score"],

                "correlation":
                    sync_result.get(
                        "correlation",
                        None
                    ),

                "best_lag":
                    sync_result.get(
                        "best_lag",
                        0
                    ),

                "valid_lip_frames":
                    sync_result[
                        "valid_lip_frames"
                    ],

                "total_lip_frames":
                    sync_result[
                        "total_lip_frames"
                    ],

                "fusion_prediction":
                    fusion_result["prediction"],

                "fraud_risk":
                    fs,

                "fraud_risk_percent":
                    fusion_result[
                        "fraud_risk_percent"
                    ]
            }
        )

        print(
            "  Video:",
            video_result["prediction"],
            f"({vs:.3f})"
        )

        print(
            "  Audio:",
            audio_result["prediction"],
            f"({ass:.3f})"
        )

        print(
            "  Sync:",
            (
                "AVAILABLE"
                if sync_result["available"]
                else "UNAVAILABLE"
            ),
            f"({ss:.3f})"
        )

        print(
            "  Fusion:",
            fusion_result["prediction"],
            f"({fs:.3f})"
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    true_values,
    predictions,
    scores
):

    accuracy = accuracy_score(
        true_values,
        predictions
    )

    precision = precision_score(
        true_values,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        true_values,
        predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        true_values,
        predictions,
        pos_label=0,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            [
                1 if y == 0 else 0
                for y in true_values
            ],
            scores
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        true_values,
        predictions,
        labels=[0, 1]
    )

    return {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),

        "roc_auc": float(
            auc
        ),

        "confusion_matrix":
            cm.tolist()
    }


# ============================================================
# CALCULATE ALL METRICS
# ============================================================

video_metrics = calculate_metrics(
    y_true,
    video_pred,
    video_score
)

audio_metrics = calculate_metrics(
    y_true,
    audio_pred,
    audio_score
)

sync_metrics = calculate_metrics(
    y_true,
    sync_pred,
    sync_score
)

fusion_metrics = calculate_metrics(
    y_true,
    fusion_pred,
    fusion_score
)


# ============================================================
# SYNC AVAILABILITY
# ============================================================

sync_available_count = sum(
    1
    for row in rows
    if row["sync_available"]
)

sync_unavailable_count = (
    len(rows)
    - sync_available_count
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print()
print("=" * 70)
print("FINAL MULTIMODAL EVALUATION RESULTS")
print("=" * 70)


def print_metrics(
    name,
    metrics
):

    print()
    print(name)

    print(
        "-" * 40
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )


print_metrics(
    "VIDEO MODEL",
    video_metrics
)

print_metrics(
    "AUDIO MODEL",
    audio_metrics
)

print_metrics(
    "SYNC SYSTEM",
    sync_metrics
)

print_metrics(
    "MULTIMODAL FUSION",
    fusion_metrics
)


print()
print("=" * 70)

print(
    "Sync available:",
    sync_available_count
)

print(
    "Sync unavailable:",
    sync_unavailable_count
)

print(
    "Successfully evaluated:",
    len(rows)
)

print("=" * 70)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    OUTPUT_DIR
    / "predictions.csv"
)

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = list(
        rows[0].keys()
    )

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(rows)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_output = {

    "test_samples":
        len(rows),

    "expected_samples":
        len(videos),

    "fake_videos":
        len(fake_videos),

    "real_videos":
        len(real_videos),

    "sync_available":
        sync_available_count,

    "sync_unavailable":
        sync_unavailable_count,

    "video": video_metrics,

    "audio": audio_metrics,

    "sync": sync_metrics,

    "fusion": fusion_metrics,

    "fusion_configuration": {

        "video_weight": 0.50,

        "audio_weight": 0.30,

        "sync_weight": 0.20,

        "threshold": 0.46
    },

    "label_mapping": {

        "0": "FAKE",

        "1": "REAL"
    }
}


json_file = (
    OUTPUT_DIR
    / "metrics.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4,
        allow_nan=True
    )


# ============================================================
# CLEAN TEMP DIRECTORY
# ============================================================

temp_dir = (
    OUTPUT_DIR
    / "temp"
)

if temp_dir.exists():

    import shutil

    shutil.rmtree(
        temp_dir
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("Saved:")
print(
    csv_file
)
print(
    json_file
)

print()
print(
    "Evaluation completed."
)