import csv
from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):

    def __init__(
        self,
        csv_file,
        audio_root,
        sample_rate=16000,
        duration=4
    ):
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate

        # 4 seconds × 16000 samples/second
        self.num_samples = sample_rate * duration

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
    # Load audio
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

        # Load as mono at 16 kHz
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

        # Convert NumPy array → PyTorch tensor
        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        # Add channel dimension
        # (64000,) → (1, 64000)

        waveform = waveform.unsqueeze(0)

        return waveform

    # ------------------------------------------------------
    # Get item
    # ------------------------------------------------------

    def __getitem__(self, index):

        video_id, label = self.samples[index]

        waveform = self._load_audio(
            video_id
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return waveform, label
        