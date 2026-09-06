from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import MultimodalFusionModel
from train.metrics import calculate_metrics, print_metrics_summary, plot_confusion_matrix

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FUSION_CHECKPOINT = CHECKPOINT_DIR / "fusion_model.pt"

RESULTS_DIR = PROJECT_ROOT / "results" / "fusion_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_fusion_model(epochs=50, lr=0.005, batch_size=16):
    """
    Train PyTorch MultimodalFusionModel on extracted video, audio, and sync predictions.
    """
    print("=" * 60)
    print("TRAINING MULTIMODAL FUSION NETWORK")
    print("=" * 60)

    video_csv = PROJECT_ROOT / "results" / "evaluation" / "test_predictions.csv"
    audio_csv = PROJECT_ROOT / "results" / "audio_evaluation" / "audio_predictions.csv"
    sync_csv = PROJECT_ROOT / "results" / "sync_evaluation" / "sync_scores.csv"

    # Fallback dataset creation if CSVs don't exist
    if video_csv.exists() and audio_csv.exists():
        df_v = pd.read_csv(video_csv)[["video_id", "true_label", "fake_probability"]].rename(
            columns={"fake_probability": "video_fake_prob"}
        )
        df_a = pd.read_csv(audio_csv)[["video_id", "fake_probability"]].rename(
            columns={"fake_probability": "audio_fake_prob"}
        )
        df = pd.merge(df_v, df_a, on="video_id", how="inner")

        if sync_csv.exists():
            df_s = pd.read_csv(sync_csv)[["video_id", "sync_score"]]
            df = pd.merge(df, df_s, on="video_id", how="left")
            df["sync_mismatch_score"] = 1.0 - df["sync_score"]
            df["sync_mismatch_score"] = df["sync_mismatch_score"].fillna(0.5)
        else:
            df["sync_mismatch_score"] = 0.5

        y = df["true_label"].str.upper().map({"REAL": 0, "FAKE": 1}).values.astype(np.float32)
        X = df[["video_fake_prob", "audio_fake_prob", "sync_mismatch_score"]].values.astype(np.float32)
    else:
        print("Warning: Prediction CSVs not found. Creating synthetic training dataset...")
        np.random.seed(42)
        n_samples = 200
        y = np.random.randint(0, 2, size=n_samples).astype(np.float32)
        v_prob = y * 0.7 + np.random.normal(0, 0.2, n_samples)
        a_prob = y * 0.6 + np.random.normal(0, 0.25, n_samples)
        s_prob = y * 0.65 + np.random.normal(0, 0.2, n_samples)
        X = np.clip(np.stack([v_prob, a_prob, s_prob], axis=1), 0.0, 1.0).astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusionModel(input_dim=3, hidden_dim=32).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    print(f"Training on {len(dataset)} samples for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:2d}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 3,
        "hidden_dim": 32
    }
    torch.save(checkpoint, FUSION_CHECKPOINT)
    print(f"\nTrained fusion model saved to: {FUSION_CHECKPOINT}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        all_probs = model(X_tensor.to(device)).cpu().numpy().flatten()
        all_preds = (all_probs >= 0.50).astype(int)

    metrics = calculate_metrics(y, all_preds, all_probs)
    print_metrics_summary(metrics, title="MULTIMODAL FUSION METRICS")

    plot_confusion_matrix(
        y, all_preds,
        save_path=RESULTS_DIR / "fusion_confusion_matrix.png",
        title="Multimodal Fusion Confusion Matrix"
    )

    return metrics


if __name__ == "__main__":
    train_fusion_model()
