import unittest
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sync_model import compute_sync_mismatch_score, compute_audio_envelope
from models.fusion_model import fuse_tri_modality, MultimodalFusionModel


class TestPreprocessingAndModels(unittest.TestCase):

    def test_audio_envelope_computation(self):
        """Test audio envelope extraction and normalization."""
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2)
        audio_signal = np.sin(2 * np.pi * 440 * t)

        env = compute_audio_envelope(audio_signal, sample_rate=sr, target_len=50)
        self.assertIsInstance(env, np.ndarray)
        self.assertEqual(len(env), 50)
        self.assertLessEqual(env.max(), 1.0)
        self.assertGreaterEqual(env.min(), 0.0)

    def test_sync_mismatch_scoring_synced(self):
        """Test lip-sync calculation for matched signals."""
        t = np.linspace(0, 1.0, 50)
        lip_motion = np.sin(2 * np.pi * 2 * t) + 1.0
        audio_signal = np.sin(2 * np.pi * 2 * t)

        res = compute_sync_mismatch_score(lip_motion, audio_signal, sample_rate=16000)
        self.assertIn("sync_mismatch_score", res)
        self.assertTrue(0.0 <= res["sync_mismatch_score"] <= 1.0)
        self.assertIn(res["status"], ["SYNCED", "MISMATCHED", "NEUTRAL"])

    def test_tri_modality_fusion_logic(self):
        """Test weighted score fusion and risk level categorization."""
        res = fuse_tri_modality(
            video_fake_prob=0.85,
            audio_fake_prob=0.80,
            sync_mismatch_score=0.90
        )
        self.assertEqual(res["prediction"], "FAKE")
        self.assertIn(res["risk_level"], ["HIGH", "CRITICAL"])
        self.assertGreater(res["fraud_risk_percent"], 75.0)

    def test_multimodal_fusion_pytorch_module(self):
        """Test PyTorch MultimodalFusionModel tensor shape and output range."""
        model = MultimodalFusionModel(input_dim=3, hidden_dim=32)
        inp = torch.tensor([[0.7, 0.6, 0.8], [0.1, 0.2, 0.1]], dtype=torch.float32)
        out = model(inp)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
