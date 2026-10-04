"""
CIFAKEDataset — PyTorch Dataset for the CIFAKE benchmark.

Folder structure expected (produced by download_cifake.py):
  <root>/
    train/
      REAL/   *.jpg  → label 1
      FAKE/   *.jpg  → label 0
    test/
      REAL/   *.jpg  → label 1
      FAKE/   *.jpg  → label 0

Label convention matches ImageDeepfakeModel (models/image_model.py):
  0 = FAKE (AI-generated / manipulated)
  1 = REAL (authentic photograph)

Transform values mirror train/image_dataset.py exactly:
  image_size = 160
  normalize mean = [0.485, 0.456, 0.406]
  normalize std  = [0.229, 0.224, 0.225]
"""
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Label map: folder name → integer label
# Must match ImageDeepfakeModel output head and api/pipeline.py:
#   Class 0 = FAKE (AI-generated / manipulated)
#   Class 1 = REAL (authentic photograph)
CIFAKE_LABEL_MAP = {
    "FAKE": 0,
    "REAL": 1,
}


class CIFAKEDataset(Dataset):
    """
    PyTorch Dataset for CIFAKE real/fake image classification.

    Args:
        root (str | Path): Path to the CIFAKE root folder (e.g. data/cifake/).
        split (str): "train" or "test".
        image_size (int): Resize target (must match IMAGE_SIZE in train scripts = 160).
        training (bool): If True, applies training augmentations; otherwise val/test transforms.
    """

    def __init__(self, root, split="train", image_size=160, training=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.training = training
        self.samples = []  # list of (Path, int) tuples

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"CIFAKE split directory not found: {split_dir}\n"
                f"Run train/download_cifake.py first."
            )

        for class_name, label in CIFAKE_LABEL_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No .jpg images found under {split_dir}. "
                f"Check that download_cifake.py completed successfully."
            )

        # Transforms mirror train/image_dataset.py exactly
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Print class balance on init
        dist = self.class_distribution()
        print(
            f"CIFAKEDataset [{split}] — "
            f"FAKE: {dist['FAKE']}, REAL: {dist['REAL']}, "
            f"Total: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)

    def class_distribution(self):
        """Returns dict of {class_name: count} for diagnostic logging."""
        counts = {"FAKE": 0, "REAL": 0}
        for _, label in self.samples:
            counts["FAKE" if label == 0 else "REAL"] += 1
        return counts
