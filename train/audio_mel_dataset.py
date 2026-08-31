import csv
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioMelDataset(Dataset):

    def __init__(
        self,
        csv_file,
        audio_root,
        sample_rate=16000,
        duration=4,
        n_mels=128,
        n_fft=1024,
        hop_length=256
    ):

        self.audio_root = Path(audio_root)

        self.sample_rate = sample_rate

        self.num_samples = (
            sample_rate * duration
        )

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.samples = []

        # --------------------------------------------------
        # Read metadata
        # --------------------------------------------------

        with open(
            csv_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                video_id = row["video_id"]

                # Your project mapping:
                # 0 = FAKE
                # 1 = REAL

                label = int(row["label"])

                self.samples.append(
                    (video_id, label)
                )

    # ------------------------------------------------------
    # Dataset length
    # ------------------------------------------------------

    def __len__(self):

        return len(self.samples)

    # ------------------------------------------------------
    # Load waveform
    # ------------------------------------------------------

    def _load_audio(self, video_id):

        audio_path = (
            self.audio_root /
            f"{video_id}.wav"
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, _ = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            mono=True
        )

        # --------------------------------------------------
        # Fix duration
        # --------------------------------------------------

        if len(waveform) < self.num_samples:

            waveform = librosa.util.fix_length(
                waveform,
                size=self.num_samples
            )

        else:

            waveform = waveform[
                :self.num_samples
            ]

        return waveform

    # ------------------------------------------------------
    # Convert waveform → Mel spectrogram
    # ------------------------------------------------------

    def _mel_spectrogram(self, waveform):

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        # Convert power → decibels

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Normalize

        mel_db = (
            mel_db - mel_db.mean()
        ) / (
            mel_db.std() + 1e-8
        )

        return mel_db

    # ------------------------------------------------------
    # Get item
    # ------------------------------------------------------

    def __getitem__(self, index):

        video_id, label = self.samples[index]

        waveform = self._load_audio(
            video_id
        )

        mel = self._mel_spectrogram(
            waveform
        )

        # NumPy → PyTorch

        mel = torch.tensor(
            mel,
            dtype=torch.float32
        )

        # Add channel dimension
        #
        # (mel_bins, time)
        #       ↓
        # (1, mel_bins, time)

        mel = mel.unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return mel, label