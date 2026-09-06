import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class VideoFrameDataset(Dataset):
    """
    High-Precision Video Frame Dataset with Strong Data Augmentation
    for Fine-Tuning EfficientNet-B0 Deepfake Classifiers.
    """

    def __init__(
        self,
        csv_file,
        frames_root,
        num_frames=16,
        image_size=160,
        training=False
    ):
        self.frames_root = Path(frames_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training

        self.samples = []

        # Read CSV metadata
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_id = row["video_id"]
                label = int(row["label"])
                self.samples.append((video_id, label))

        # Strong Training Augmentations for Fine-Tuning EfficientNet
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.5, 2.0))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def _get_frame_paths(self, video_id):
        video_dir = self.frames_root / video_id
        if not video_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {video_dir}")

        frame_paths = sorted(video_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames found for {video_id}")
        return frame_paths

    def _select_frames(self, frame_paths):
        total = len(frame_paths)
        if total >= self.num_frames:
            indices = torch.linspace(0, total - 1, self.num_frames).long()
            return [frame_paths[i] for i in indices]

        selected = list(frame_paths)
        while len(selected) < self.num_frames:
            selected.append(frame_paths[-1])
        return selected

    def __getitem__(self, index):
        video_id, label = self.samples[index]
        frame_paths = self._get_frame_paths(video_id)
        frame_paths = self._select_frames(frame_paths)

        frames = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image = self.transform(image)
            frames.append(image)

        frames = torch.stack(frames)
        label = torch.tensor(label, dtype=torch.long)
        return frames, label