import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor

from models.video_model import load_video_model, calculate_fraud_score


CHECKPOINT = "checkpoints/video_deepfake/pytorch_model.bin"
NUM_FRAMES = 16
IMAGE_SIZE = 224


def load_frames(frames_dir, processor):
    frames_dir = Path(frames_dir)

    frame_files = sorted(frames_dir.glob("*.jpg"))

    if not frame_files:
        raise RuntimeError(
            f"No JPG frames found in: {frames_dir}"
        )

    # Use exactly 16 frames
    if len(frame_files) >= NUM_FRAMES:
        frame_files = frame_files[:NUM_FRAMES]

    images = []

    for frame_file in frame_files:
        image = Image.open(frame_file).convert("RGB")
        images.append(image)

    processed = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = processed["pixel_values"]

    return pixel_values


def predict(video_frames_dir):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load CLIP processor
    processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # Load trained model
    model = load_video_model(
        CHECKPOINT,
        device=device
    )

    # Load frames
    frames = load_frames(
        video_frames_dir,
        processor
    )

    print("Frames tensor:", frames.shape)

    # Model expects:
    # (batch, frames, channels, height, width)

    frames = frames.unsqueeze(0)

    frames = frames.to(device)

    print("Model input:", frames.shape)

    # Inference
    with torch.no_grad():
        logits = model(frames)

    print("Logits:", logits)

    result = calculate_fraud_score(logits.cpu())

    print()
    print("=" * 40)
    print("VIDEO DEEPFAKE RESULT")
    print("=" * 40)

    print(
        f"Real probability : "
        f"{result['real_probability']:.2f}%"
    )

    print(
        f"Fake probability : "
        f"{result['fake_probability']:.2f}%"
    )

    print(
        f"Fraud score      : "
        f"{result['fraud_score']:.2f}%"
    )

    if result["fake_probability"] >= 50:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    print(f"Prediction       : {prediction}")

    print("=" * 40)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python video_inference.py "
            "<frames_directory>"
        )
        sys.exit(1)

    frames_dir = sys.argv[1]

    predict(frames_dir)