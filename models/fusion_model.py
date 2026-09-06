import torch
import torch.nn as nn
import numpy as np


class MultimodalFusionModel(nn.Module):
    """
    PyTorch Multimodal Fusion Layer.
    Fuses predictions from Video classifier, Audio classifier, and Lip-Sync Mismatch evaluator
    into a unified Fraud Risk Score (0.0 = Real/Legitimate, 1.0 = Fraud/Deepfake).
    """

    def __init__(self, input_dim=3, hidden_dim=32, num_classes=1):
        super().__init__()

        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Input: x of shape (batch, 3) representing [video_fake_prob, audio_fake_prob, sync_mismatch_score]
        Output: fraud_risk_score of shape (batch, 1) in range [0, 1]
        """
        return self.fusion_network(x)


def fuse_tri_modality(
    video_fake_prob,
    audio_fake_prob,
    sync_mismatch_score,
    weights=(0.50, 0.30, 0.20),
    model=None,
    device="cpu"
):
    """
    Combines video score, audio score, and lip-sync mismatch score into a high-accuracy fraud risk assessment.
    """
    v_score = float(np.clip(video_fake_prob, 0.0, 1.0))
    a_score = float(np.clip(audio_fake_prob, 0.0, 1.0))
    s_score = float(np.clip(sync_mismatch_score, 0.0, 1.0))

    # Weighted baseline risk score
    vw, aw, sw = weights
    total_w = vw + aw + sw
    vw, aw, sw = vw / total_w, aw / total_w, sw / total_w
    weighted_score = (vw * v_score) + (aw * a_score) + (sw * s_score)

    if model is not None and isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            inp_tensor = torch.tensor([[v_score, a_score, s_score]], dtype=torch.float32).to(device)
            learned_risk = float(model(inp_tensor).cpu().numpy()[0, 0])
        fusion_score = max(learned_risk, weighted_score)
    else:
        fusion_score = weighted_score

    # Forensic Decision Rules:
    # 1. Clean Real Video Guardrail: if all sub-scores are low (<= 0.38), guarantee LOW Fraud Risk (<= 22%)
    if v_score <= 0.38 and a_score <= 0.38 and s_score <= 0.38:
        fusion_score = min(fusion_score, 0.22)

    # 2. Strong Deepfake Trigger: if any sub-score shows clear deepfake evidence (>= 0.52), elevate to FAKE (>= 68%)
    elif max(v_score, a_score, s_score) >= 0.52:
        fusion_score = max(fusion_score, max(v_score, a_score, s_score) * 0.96, 0.68)

    # 3. Multi-Cue Accumulation: if 2+ sub-scores show elevated cues (>= 0.42), escalate to FAKE (>= 65%)
    elif sum([v_score >= 0.42, a_score >= 0.42, s_score >= 0.42]) >= 2:
        fusion_score = max(fusion_score, 0.65)

    fraud_risk_percent = round(fusion_score * 100, 2)
    prediction = "FAKE" if fusion_score >= 0.50 else "REAL"

    # Risk level classification for forensic workflow
    if fusion_score >= 0.75:
        risk_level = "CRITICAL"
    elif fusion_score >= 0.50:
        risk_level = "HIGH"
    elif fusion_score >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Identify primary triggering modality
    modalities = {
        "VIDEO": v_score,
        "AUDIO": a_score,
        "SYNC_MISMATCH": s_score
    }
    max_modality = max(modalities, key=modalities.get)
    max_val = modalities[max_modality]

    if max_val >= 0.45:
        if s_score >= 0.50 and v_score >= 0.45 and a_score < 0.45:
            trigger_modality = "SYNC_MISMATCH (Dubbed Voice on Swapped Face)"
        elif v_score >= 0.50 and a_score >= 0.50:
            trigger_modality = "MULTIMODAL (Face Swap + Voice Clone)"
        else:
            trigger_modality = f"{max_modality} ANOMALY"
    else:
        trigger_modality = "NONE (Consistent Real Video)"

    # Confidence calculation
    score_variance = float(np.std([v_score, a_score, s_score]))
    dist_from_thresh = abs(fusion_score - 0.50)
    if dist_from_thresh > 0.25 or (fusion_score > 0.68 and score_variance < 0.30):
        confidence = "HIGH"
    elif dist_from_thresh > 0.10:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "fraud_risk_score": round(fusion_score, 4),
        "fraud_risk_percent": fraud_risk_percent,
        "prediction": prediction,
        "risk_level": risk_level,
        "confidence": confidence,
        "trigger_modality": trigger_modality,
        "sub_scores": {
            "video_fake_prob": round(v_score, 4),
            "audio_fake_prob": round(a_score, 4),
            "sync_mismatch_score": round(s_score, 4)
        }
    }
