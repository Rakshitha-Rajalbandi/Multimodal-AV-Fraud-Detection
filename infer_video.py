# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms

# from models.video_model import VideoDeepfakeModel


# # ============================================================
# # Configuration
# # ============================================================

# NUM_FRAMES = 8
# IMAGE_SIZE = 160

# CHECKPOINT = (
#     "checkpoints/video_deepfake/"
#     "efficientnet_b0_best.pth"
# )

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # ============================================================
# # Extract frames from video
# # ============================================================

# def extract_frames(video_path, num_frames=8):

#     cap = cv2.VideoCapture(str(video_path))

#     if not cap.isOpened():
#         raise RuntimeError(
#             f"Could not open video: {video_path}"
#         )

#     total_frames = int(
#         cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     )

#     if total_frames <= 0:
#         cap.release()

#         raise RuntimeError(
#             f"No frames found in: {video_path}"
#         )

#     indices = np.linspace(
#         0,
#         total_frames - 1,
#         num_frames
#     ).astype(int)

#     frames = []

#     for index in indices:

#         cap.set(
#             cv2.CAP_PROP_POS_FRAMES,
#             int(index)
#         )

#         success, frame = cap.read()

#         if not success:
#             continue

#         frame = cv2.cvtColor(
#             frame,
#             cv2.COLOR_BGR2RGB
#         )

#         image = Image.fromarray(frame)

#         frames.append(image)

#     cap.release()

#     if len(frames) == 0:
#         raise RuntimeError(
#             "Could not extract any frames."
#         )

#     # If some frames failed, repeat the last frame
#     while len(frames) < num_frames:
#         frames.append(frames[-1].copy())

#     return frames


# # ============================================================
# # Image transformation
# # ============================================================

# transform = transforms.Compose([

#     transforms.Resize(
#         (IMAGE_SIZE, IMAGE_SIZE)
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])


# # ============================================================
# # Load trained model
# # ============================================================

# def load_model():

#     print("Loading model...")
#     print("Device:", DEVICE)

#     model = VideoDeepfakeModel()

#     checkpoint = torch.load(
#         CHECKPOINT,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     if "model_state_dict" in checkpoint:

#         state_dict = checkpoint[
#             "model_state_dict"
#         ]

#     else:

#         state_dict = checkpoint

#     model.load_state_dict(
#         state_dict,
#         strict=True
#     )

#     model.to(DEVICE)

#     model.eval()

#     print("Checkpoint loaded successfully.")

#     if "epoch" in checkpoint:

#         print(
#             "Checkpoint epoch:",
#             checkpoint["epoch"]
#         )

#     if "val_accuracy" in checkpoint:

#         print(
#             "Validation accuracy:",
#             f"{checkpoint['val_accuracy'] * 100:.2f}%"
#         )

#     return model


# # ============================================================
# # Predict video
# # ============================================================

# def predict_video(model, video_path):

#     print()
#     print("Extracting frames...")

#     frames = extract_frames(
#         video_path,
#         NUM_FRAMES
#     )

#     print(
#         f"Frames extracted: {len(frames)}"
#     )

#     tensors = []

#     for frame in frames:

#         tensor = transform(frame)

#         tensors.append(tensor)

#     # Shape:
#     # (frames, channels, height, width)

#     frames_tensor = torch.stack(
#         tensors
#     )

#     # Add batch dimension:
#     # (1, frames, channels, height, width)

#     frames_tensor = frames_tensor.unsqueeze(
#         0
#     )

#     frames_tensor = frames_tensor.to(
#         DEVICE
#     )

#     # --------------------------------------------------------
#     # Model prediction
#     # --------------------------------------------------------

#     with torch.no_grad():

#         logits = model(
#             frames_tensor
#         )

#         probabilities = torch.softmax(
#             logits,
#             dim=1
#         )[0]

#     # IMPORTANT:
#     # Your project mapping is:
#     # 0 = FAKE
#     # 1 = REAL

#     fake_probability = float(
#         probabilities[0]
#     )

#     real_probability = float(
#         probabilities[1]
#     )

#     if fake_probability > real_probability:

#         prediction = "FAKE"

#         fraud_score = fake_probability * 100

#     else:

#         prediction = "REAL"

#         fraud_score = fake_probability * 100

#     return (
#         prediction,
#         fake_probability,
#         real_probability,
#         fraud_score
#     )


# # ============================================================
# # Main
# # ============================================================

# def main():

#     parser = argparse.ArgumentParser(
#         description=(
#             "Video Deepfake Detection "
#             "Inference"
#         )
#     )

#     parser.add_argument(
#         "--video",
#         required=True,
#         help="Path to input video"
#     )

#     args = parser.parse_args()

#     video_path = Path(
#         args.video
#     )

#     if not video_path.exists():

#         raise FileNotFoundError(
#             f"Video not found: {video_path}"
#         )

#     print()
#     print("=" * 60)
#     print("VIDEO DEEPFAKE DETECTION")
#     print("=" * 60)

#     print(
#         "Video:",
#         video_path
#     )

#     model = load_model()

#     (
#         prediction,
#         fake_probability,
#         real_probability,
#         fraud_score
#     ) = predict_video(
#         model,
#         video_path
#     )

#     print()
#     print("=" * 60)
#     print("RESULT")
#     print("=" * 60)

#     print(
#         f"Fake Probability : "
#         f"{fake_probability * 100:.2f}%"
#     )

#     print(
#         f"Real Probability : "
#         f"{real_probability * 100:.2f}%"
#     )

#     print(
#         f"Fraud Risk Score : "
#         f"{fraud_score:.2f}%"
#     )

#     print(
#         f"Prediction       : "
#         f"{prediction}"
#     )

#     print("=" * 60)


# if __name__ == "__main__":
#     main()




import argparse
import torch
import cv2
import numpy as np

from PIL import Image
from torchvision import transforms

from models.video_model import VideoDeepfakeModel


# ============================================================
# Configuration
# ============================================================

NUM_FRAMES = 8
IMAGE_SIZE = 160

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    "checkpoints/video_deepfake/"
    "efficientnet_b0_best.pth"
)


# ============================================================
# Extract frames from video
# ============================================================

def extract_frames(video_path, num_frames=8):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video frame count."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int)

    frames = []

    for index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        frames.append(image)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            "No frames could be extracted."
        )

    # If some frames failed, repeat last frame
    while len(frames) < num_frames:

        frames.append(frames[-1])

    return frames


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load model
# ============================================================

def load_model():

    print("Device:", DEVICE)

    print("Loading model...")

    model = VideoDeepfakeModel()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("Checkpoint loaded successfully.")

    if "epoch" in checkpoint:
        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

    if "val_accuracy" in checkpoint:
        print(
            "Validation accuracy:",
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    return model


# ============================================================
# Predict
# ============================================================

def predict(video_path):

    model = load_model()

    print()
    print("=" * 60)
    print("VIDEO DEEPFAKE DETECTION")
    print("=" * 60)

    print("Video:", video_path)

    # --------------------------------------------------------
    # Extract frames
    # --------------------------------------------------------

    frames = extract_frames(
        video_path,
        NUM_FRAMES
    )

    print(
        "Frames extracted:",
        len(frames)
    )

    # --------------------------------------------------------
    # Transform frames
    # --------------------------------------------------------

    frame_tensors = []

    for frame in frames:

        tensor = transform(frame)

        frame_tensors.append(tensor)

    frames_tensor = torch.stack(
        frame_tensors
    )

    # Shape:
    # (frames, channels, height, width)

    frames_tensor = frames_tensor.unsqueeze(0)

    # Shape:
    # (1, frames, channels, height, width)

    frames_tensor = frames_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            frames_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # IMPORTANT:
    # Your actual trained mapping is:
    #
    # 0 = FAKE
    # 1 = REAL
    #
    # This matches your metadata.

    fake_probability = float(
        probabilities[0, 0]
    )

    real_probability = float(
        probabilities[0, 1]
    )

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    if fake_probability > real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("-" * 60)

    print(
        f"FAKE probability : "
        f"{fake_probability * 100:.2f}%"
    )

    print(
        f"REAL probability : "
        f"{real_probability * 100:.2f}%"
    )

    print()

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Fraud Risk Score: "
        f"{fake_probability * 100:.2f}%"
    )

    print("-" * 60)

    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Video Deepfake Detection"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video"
    )

    args = parser.parse_args()

    predict(
        args.video
    )