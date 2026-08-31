# import csv
# from pathlib import Path

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# from torchvision import transforms


# class VideoFrameDataset(Dataset):

#     def __init__(
#         self,
#         metadata_file,
#         frames_root,
#         num_frames=8,
#         image_size=160,
#         training=False
#     ):
#         """
#         metadata_file:
#             CSV containing:
#                 video_id,label

#             Example:
#                 sample,0
#                 fake001,1

#         frames_root:
#             Root directory containing:

#                 frames_root/
#                     sample/
#                         0000.jpg
#                         0001.jpg
#                         ...
#                     fake001/
#                         0000.jpg
#                         ...

#         Labels:
#             0 = Real
#             1 = Fake
#         """

#         self.frames_root = Path(frames_root)
#         self.num_frames = num_frames
#         self.image_size = image_size
#         self.training = training

#         self.samples = []

#         # --------------------------------------------------
#         # Read metadata
#         # --------------------------------------------------

#         with open(
#             metadata_file,
#             "r",
#             newline="",
#             encoding="utf-8"
#         ) as f:

#             reader = csv.DictReader(f)

#             for row in reader:

#                 video_id = row["video_id"]
#                 label = int(row["label"])

#                 self.samples.append(
#                     (video_id, label)
#                 )

#         # --------------------------------------------------
#         # Image preprocessing
#         # --------------------------------------------------

#         if training:

#             self.transform = transforms.Compose([
#                 transforms.Resize(
#                     (image_size, image_size)
#                 ),

#                 transforms.RandomHorizontalFlip(
#                     p=0.5
#                 ),

#                 transforms.ColorJitter(
#                     brightness=0.15,
#                     contrast=0.15,
#                     saturation=0.15
#                 ),

#                 transforms.ToTensor(),

#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])

#         else:

#             self.transform = transforms.Compose([
#                 transforms.Resize(
#                     (image_size, image_size)
#                 ),

#                 transforms.ToTensor(),

#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 )
#             ])

#     def __len__(self):
#         return len(self.samples)

#     def _get_frame_paths(self, video_id):

#         video_dir = self.frames_root / video_id

#         if not video_dir.exists():
#             raise FileNotFoundError(
#                 f"Frame directory not found: {video_dir}"
#             )

#         frame_paths = sorted(
#             video_dir.glob("*.jpg")
#         )

#         if len(frame_paths) == 0:
#             raise RuntimeError(
#                 f"No frames found for video: {video_id}"
#             )

#         return frame_paths

#     def _select_frames(self, frame_paths):

#         total = len(frame_paths)

#         # ----------------------------------------------
#         # If enough frames exist, sample evenly
#         # ----------------------------------------------

#         if total >= self.num_frames:

#             indices = torch.linspace(
#                 0,
#                 total - 1,
#                 self.num_frames
#             ).long()

#             return [
#                 frame_paths[i]
#                 for i in indices
#             ]

#         # ----------------------------------------------
#         # If fewer frames exist, repeat last frame
#         # ----------------------------------------------

#         selected = list(frame_paths)

#         while len(selected) < self.num_frames:
#             selected.append(frame_paths[-1])

#         return selected

#     def __getitem__(self, index):

#         video_id, label = self.samples[index]

#         frame_paths = self._get_frame_paths(
#             video_id
#         )

#         frame_paths = self._select_frames(
#             frame_paths
#         )

#         frames = []

#         for frame_path in frame_paths:

#             image = Image.open(
#                 frame_path
#             ).convert("RGB")

#             image = self.transform(image)

#             frames.append(image)

#         # --------------------------------------------------
#         # Stack frames
#         # --------------------------------------------------

#         frames = torch.stack(frames)

#         label = torch.tensor(
#             label,
#             dtype=torch.long
#         )

#         return frames, label




import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class VideoFrameDataset(Dataset):

    def __init__(
        self,
        csv_file,
        frames_root,
        num_frames=8,
        image_size=160,
        training=False
    ):
        self.frames_root = Path(frames_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.training = training

        self.samples = []

        # Read CSV
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

        # Training augmentation
        if training:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Validation / testing
        else:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (image_size, image_size)
                ),

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

            raise FileNotFoundError(
                f"Frame directory not found: {video_dir}"
            )

        frame_paths = sorted(
            video_dir.glob("*.jpg")
        )

        if not frame_paths:

            raise RuntimeError(
                f"No frames found for {video_id}"
            )

        return frame_paths

    def _select_frames(self, frame_paths):

        total = len(frame_paths)

        # Enough frames
        if total >= self.num_frames:

            indices = torch.linspace(
                0,
                total - 1,
                self.num_frames
            ).long()

            return [
                frame_paths[i]
                for i in indices
            ]

        # Not enough frames → repeat last
        selected = list(frame_paths)

        while len(selected) < self.num_frames:

            selected.append(
                frame_paths[-1]
            )

        return selected

    def __getitem__(self, index):

        video_id, label = self.samples[index]

        frame_paths = self._get_frame_paths(
            video_id
        )

        frame_paths = self._select_frames(
            frame_paths
        )

        frames = []

        for frame_path in frame_paths:

            image = Image.open(
                frame_path
            ).convert("RGB")

            image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return frames, label