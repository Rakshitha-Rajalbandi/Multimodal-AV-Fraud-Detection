import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.metrics import calculate_metrics, print_metrics_summary, plot_confusion_matrix

RESULTS_DIR = PROJECT_ROOT / "results" / "multimodal_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ablation_study():
    """
    Evaluates system performance across single-modality baselines and full multimodal fusion.
    Compares:
      1. Video-Only Baseline
      2. Audio-Only Baseline
      3. Video + Audio (No Lip-Sync)
      4. Video + Audio + Lip-Sync Mismatch (Full Multimodal System)
    """
    print("=" * 65)
    print("MULTIMODAL DEEPFAKE DETECTION - ABLATION STUDY EVALUATION")
    print("=" * 65)

    video_csv = PROJECT_ROOT / "results" / "evaluation" / "test_predictions.csv"
    audio_csv = PROJECT_ROOT / "results" / "audio_evaluation" / "audio_predictions.csv"
    sync_csv = PROJECT_ROOT / "results" / "sync_evaluation" / "sync_scores.csv"

    if not (video_csv.exists() and audio_csv.exists() and sync_csv.exists()):
        print("Error: Missing prediction CSV files required for ablation study.")
        return

    df_v = pd.read_csv(video_csv)[["video_id", "true_label", "fake_probability"]].rename(
        columns={"fake_probability": "video_fake_prob"}
    )
    df_a = pd.read_csv(audio_csv)[["video_id", "fake_probability"]].rename(
        columns={"fake_probability": "audio_fake_prob"}
    )
    df_s = pd.read_csv(sync_csv)[["video_id", "sync_score"]]

    df = pd.merge(df_v, df_a, on="video_id", how="inner")
    df = pd.merge(df, df_s, on="video_id", how="inner")

    # Sync Mismatch Score: 1.0 - sync_score
    df["sync_mismatch_score"] = 1.0 - df["sync_score"]

    y_true = df["true_label"].str.upper().map({"REAL": 0, "FAKE": 1}).values

    # 1. Video-Only Baseline
    v_prob = df["video_fake_prob"].values
    v_pred = (v_prob >= 0.50).astype(int)
    metrics_video = calculate_metrics(y_true, v_pred, v_prob)

    # 2. Audio-Only Baseline
    a_prob = df["audio_fake_prob"].values
    a_pred = (a_prob >= 0.50).astype(int)
    metrics_audio = calculate_metrics(y_true, a_pred, a_prob)

    # 3. Video + Audio (No Lip-Sync)
    va_prob = 0.65 * v_prob + 0.35 * a_prob
    va_pred = (va_prob >= 0.50).astype(int)
    metrics_va = calculate_metrics(y_true, va_pred, va_prob)

    # 4. Full Video + Audio + Lip-Sync Mismatch Fusion
    s_prob = df["sync_mismatch_score"].values
    full_prob = 0.50 * v_prob + 0.30 * a_prob + 0.20 * s_prob
    full_pred = (full_prob >= 0.50).astype(int)
    metrics_full = calculate_metrics(y_true, full_pred, full_prob)

    # Summary Ablation Table
    ablation_summary = {
        "1. Video-Only (EfficientNet-B0)": metrics_video,
        "2. Audio-Only (2D-CNN Spectrogram)": metrics_audio,
        "3. Video + Audio (Dual Modality)": metrics_va,
        "4. Full System (Video + Audio + LipSync Mismatch)": metrics_full
    }

    print("\n" + "=" * 75)
    print(f" {'MODEL / FEATURE CONFIGURATION':<45} | {'ACC':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6} | {'AUC':<6}")
    print("=" * 75)
    for model_name, m in ablation_summary.items():
        print(f" {model_name:<45} | {m['accuracy']:<6.4f} | {m['precision']:<6.4f} | {m['recall']:<6.4f} | {m['f1_score']:<6.4f} | {m['roc_auc']:<6.4f}")
    print("=" * 75 + "\n")

    # Save metrics JSON & plot confusion matrix
    with open(RESULTS_DIR / "ablation_study_metrics.json", "w") as f:
        json.dump(ablation_summary, f, indent=2)

    plot_confusion_matrix(
        y_true, full_pred,
        save_path=RESULTS_DIR / "full_system_confusion_matrix.png",
        title="Full Multimodal System Confusion Matrix"
    )

    print(f"Ablation study evaluation completed. Results saved to: {RESULTS_DIR}")
    return ablation_summary


if __name__ == "__main__":
    run_ablation_study()