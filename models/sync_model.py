import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import librosa


class LipSyncModel(nn.Module):
    """
    PyTorch Lip-Sync Mismatch Scoring Model.
    Processes audio temporal energy features and lip movement sequences.
    """

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        self.lip_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, audio_seq, lip_seq):
        b = audio_seq.shape[0]
        a_feat = self.audio_encoder(audio_seq).view(b, -1)
        l_feat = self.lip_encoder(lip_seq).view(b, -1)

        cos_sim = torch.cosine_similarity(a_feat, l_feat, dim=-1, eps=1e-8).unsqueeze(-1)
        cat_feat = torch.cat([a_feat, l_feat, cos_sim], dim=-1)
        mismatch_prob = self.classifier(cat_feat)
        return mismatch_prob


def compute_audio_envelope(audio_signal, sample_rate=16000, target_len=100):
    """
    Extract normalized temporal energy envelope of audio signal.
    """
    if audio_signal is None or len(audio_signal) == 0:
        return np.zeros(target_len, dtype=np.float32)

    audio = np.asarray(audio_signal, dtype=np.float32)
    hop_length = max(128, int(sample_rate / 50))
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length)[0]

    r_min, r_max = rms.min(), rms.max()
    if r_max - r_min > 1e-7:
        rms = (rms - r_min) / (r_max - r_min)
    else:
        rms = np.zeros_like(rms)

    if len(rms) > 1:
        x_old = np.linspace(0, 1, len(rms))
        x_new = np.linspace(0, 1, target_len)
        envelope = np.interp(x_new, x_old, rms)
    else:
        envelope = np.zeros(target_len, dtype=np.float32)

    return envelope.astype(np.float32)


def compute_sync_mismatch_score(lip_motion, audio_signal, sample_rate=16000, max_lag=5):
    """
    Compute calibrated cross-modal audio-visual synchronization mismatch score.
    """
    if lip_motion is None or len(lip_motion) == 0:
        return {
            "sync_mismatch_score": 0.35,
            "sync_match_score": 0.65,
            "max_correlation": 0.0,
            "optimal_lag": 0,
            "status": "NEUTRAL"
        }

    lip_array = np.asarray(lip_motion, dtype=np.float32)
    lip_array = np.nan_to_num(lip_array, nan=0.0)
    target_len = max(len(lip_array), 20)

    if isinstance(audio_signal, (str, float, int)) or audio_signal is None:
        audio_env = np.zeros(target_len, dtype=np.float32)
    else:
        audio_env = compute_audio_envelope(audio_signal, sample_rate=sample_rate, target_len=target_len)

    l_min, l_max = lip_array.min(), lip_array.max()
    if l_max - l_min > 1e-7:
        lip_norm = (lip_array - l_min) / (l_max - l_min)
    else:
        lip_norm = np.zeros_like(lip_array)

    if len(lip_norm) != target_len:
        x_old = np.linspace(0, 1, len(lip_norm))
        x_new = np.linspace(0, 1, target_len)
        lip_norm = np.interp(x_new, x_old, lip_norm).astype(np.float32)

    a_mean = audio_env - np.mean(audio_env)
    l_mean = lip_norm - np.mean(lip_norm)

    norm_a = np.linalg.norm(a_mean) + 1e-7
    norm_l = np.linalg.norm(l_mean) + 1e-7

    corr = scipy.signal.correlate(a_mean / norm_a, l_mean / norm_l, mode='full')
    lags = scipy.signal.correlation_lags(len(a_mean), len(l_mean), mode='full')

    valid_mask = (lags >= -max_lag) & (lags <= max_lag)
    if np.any(valid_mask):
        valid_corr = corr[valid_mask]
        valid_lags = lags[valid_mask]
        max_corr_idx = np.argmax(valid_corr)
        max_corr = float(valid_corr[max_corr_idx])
        opt_lag = int(valid_lags[max_corr_idx])
    else:
        max_corr = 0.0
        opt_lag = 0

    max_corr = float(np.clip(max_corr, 0.0, 1.0))

    # Calibrate correlation to mismatch score
    # High correlation (>= 0.35) -> Low mismatch score (<= 0.25)
    # Low correlation (< 0.15) -> High mismatch score (>= 0.65)
    if max_corr >= 0.35:
        base_mismatch = 0.10 + (0.35 - max_corr) * 0.5
    elif max_corr >= 0.20:
        base_mismatch = 0.25 + (0.35 - max_corr) * 1.5
    else:
        base_mismatch = 0.55 + (0.20 - max_corr) * 2.0

    lag_penalty = min(abs(opt_lag) * 0.02, 0.10)
    sync_mismatch_score = float(np.clip(base_mismatch + lag_penalty, 0.05, 0.95))
    sync_match_score = float(1.0 - sync_mismatch_score)

    if sync_mismatch_score > 0.55:
        status = "MISMATCHED"
    elif sync_mismatch_score < 0.35:
        status = "SYNCED"
    else:
        status = "NEUTRAL"

    return {
        "sync_mismatch_score": round(sync_mismatch_score, 4),
        "sync_match_score": round(sync_match_score, 4),
        "max_correlation": round(max_corr, 4),
        "optimal_lag": opt_lag,
        "status": status
    }
