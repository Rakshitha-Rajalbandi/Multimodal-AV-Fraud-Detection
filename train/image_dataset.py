import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFrameDataset(Dataset):
    """
    Dataset for single frame image deepfake classification.
    Iterates over all frame images (.jpg) in video directories.
    """
    def __init__(self, csv_file, frames_root, image_size=160, training=False):
        self.frames_root = Path(frames_root)
        self.image_size = image_size
        self.training = training
        self.samples = []

        # Read CSV
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid_id = row["video_id"]
                label = int(row["label"])
                vid_dir = self.frames_root / vid_id
                if vid_dir.exists():
                    img_paths = sorted(list(vid_dir.glob("*.jpg")))
                    for img_p in img_paths:
                        self.samples.append((img_p, label))

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)
