"""
GenImage Dataset — PyTorch Dataset for the GenImage benchmark.

Folder structure expected (produced by download_genimage.py):
  <root>/
    imagenet_glide/           *.jpg / *.JPEG  → label 1  (REAL)
    imagenet_ai_0419_biggan/  *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0419_vqdm/    *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0508_adm/     *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0508_glide/   *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0419_midjourney/ *.jpg        → label 0  (FAKE)
    imagenet_ai_0419_sdv4/    *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0419_sdv5/    *.jpg / *.JPEG  → label 0  (FAKE)
    imagenet_ai_0424_wukong/  *.jpg / *.JPEG  → label 0  (FAKE)

Label convention matches ImageDeepfakeModel (models/image_model.py):
  0 = FAKE (AI-generated / manipulated)
  1 = REAL (authentic photograph)

Transform values mirror cifake_dataset.py exactly:
  image_size = 160
  normalize mean = [0.485, 0.456, 0.406]
  normalize std  = [0.229, 0.224, 0.225]

Split is deterministic (no random shuffle):
  train : first 80% of sorted file list per class
  val   : next  10%
  test  : last  10%
"""
from pathlib import Path
from typing import List, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Label convention — must match ImageDeepfakeModel and api/pipeline.py
FAKE_LABEL = 0
REAL_LABEL = 1

# Common image extensions to scan
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG")

# The folder name (substring match) that identifies REAL images
REAL_FOLDER_PATTERN = "imagenet_glide"

# Deterministic split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (implicit remainder)


class GenImageDataset(Dataset):
    """
    PyTorch Dataset for GenImage real/fake image classification.

    Args:
        root (str | Path): Path to the GenImage root folder (e.g. data/genimage/).
                           Defaults to <project_root>/data/genimage/.
        split (str): One of "train", "val", "test".
        image_size (int): Resize target — must match IMAGE_SIZE = 160 in train scripts.
        training (bool): If True, applies training augmentations; otherwise val/test only.
    """

    def __init__(
        self,
        root=None,
        split: str = "train",
        image_size: int = 160,
        training: bool = False,
    ):
        if root is None:
            root = PROJECT_ROOT / "data" / "genimage"
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.training = training
        self.samples: List[Tuple[Path, int]] = []

        if not self.root.exists():
            raise FileNotFoundError(
                f"GenImage data directory not found: {self.root}\n"
                f"Run train/download_genimage.py first."
            )

        # Collect all images, classifying by folder name
        all_real: List[Path] = []
        all_fake: List[Path] = []

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            imgs = sorted(
                p for p in subdir.rglob("*")
                if p.suffix.lower() in {ext.lower() for ext in IMAGE_EXTENSIONS}
            )
            if REAL_FOLDER_PATTERN in subdir.name:
                all_real.extend(imgs)
            else:
                all_fake.extend(imgs)

        # Deterministic split for each class
        self.samples = (
            _split_list(all_real, split, REAL_LABEL) +
            _split_list(all_fake, split, FAKE_LABEL)
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found for split '{split}' under {self.root}. "
                f"Check that download_genimage.py completed successfully."
            )

        # Transforms — mirror cifake_dataset.py exactly
        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # Print class balance on init
        dist = self.class_distribution()
        print(
            f"GenImageDataset [{split}] — "
            f"FAKE: {dist['FAKE']}, REAL: {dist['REAL']}, "
            f"Total: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            print(f"WARNING: Skipping corrupt image {img_path}: {exc}")
            return self.__getitem__((index + 1) % len(self.samples))
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)

    def class_distribution(self):
        """Returns dict of {class_name: count} for diagnostic logging."""
        counts = {"FAKE": 0, "REAL": 0}
        for _, label in self.samples:
            counts["FAKE" if label == FAKE_LABEL else "REAL"] += 1
        return counts


def _split_list(
    items: List[Path], split: str, label: int
) -> List[Tuple[Path, int]]:
    """Slice a sorted list into train/val/test using index-based deterministic split."""
    n = len(items)
    train_end = int(n * TRAIN_RATIO)
    val_end   = train_end + int(n * VAL_RATIO)

    if split == "train":
        chosen = items[:train_end]
    elif split == "val":
        chosen = items[train_end:val_end]
    elif split == "test":
        chosen = items[val_end:]
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

    return [(p, label) for p in chosen]
