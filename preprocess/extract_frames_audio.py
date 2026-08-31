# import cv2
# import subprocess
# import argparse
# from pathlib import Path


# def extract_frames(video_path, output_dir, frames_per_video=16):
#     video = cv2.VideoCapture(str(video_path))

#     if not video.isOpened():
#         raise ValueError(f"Could not open video: {video_path}")

#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames <= 0:
#         video.release()
#         raise ValueError(f"No frames found in: {video_path}")

#     # Select evenly spaced frames
#     frame_indices = [
#         int(i * (total_frames - 1) / (frames_per_video - 1))
#         for i in range(frames_per_video)
#     ]

#     output_dir.mkdir(parents=True, exist_ok=True)

#     for i, frame_index in enumerate(frame_indices):
#         video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

#         success, frame = video.read()

#         if success:
#             output_path = output_dir / f"frame_{i:03d}.jpg"
#             cv2.imwrite(str(output_path), frame)

#     video.release()

#     print(f"Extracted {frames_per_video} frames")


# def extract_audio(video_path, output_audio):
#     output_audio.parent.mkdir(parents=True, exist_ok=True)

#     command = [
#         "ffmpeg",
#         "-y",
#         "-i",
#         str(video_path),
#         "-vn",
#         "-ac",
#         "1",
#         "-ar",
#         "16000",
#         str(output_audio)
#     ]

#     result = subprocess.run(
#         command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )

#     if result.returncode != 0:
#         raise RuntimeError(
#             f"FFmpeg audio extraction failed:\n{result.stderr}"
#         )

#     print(f"Audio saved to: {output_audio}")


# def process_video(video_path, output_dir, frames_per_video):
#     video_id = video_path.stem

#     frames_dir = output_dir / "frames" / video_id
#     audio_dir = output_dir / "audio"

#     audio_path = audio_dir / f"{video_id}.wav"

#     print(f"\nProcessing: {video_path.name}")

#     extract_frames(
#         video_path,
#         frames_dir,
#         frames_per_video
#     )

#     extract_audio(
#         video_path,
#         audio_path
#     )

#     print("Processing completed.")


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--input-dir",
#         required=True
#     )

#     parser.add_argument(
#         "--out-dir",
#         required=True
#     )

#     parser.add_argument(
#         "--frames-per-video",
#         type=int,
#         default=16
#     )

#     args = parser.parse_args()

#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.out_dir)

#     videos = list(input_dir.glob("*.mp4"))

#     if not videos:
#         print(f"No MP4 videos found in {input_dir}")
#         return

#     print(f"Found {len(videos)} video(s)")

#     for video in videos:
#         process_video(
#             video,
#             output_dir,
#             args.frames_per_video
#         )


# if __name__ == "__main__":
#     main()





# """
# Extract frames and audio from videos. Sample N frames per video.
# Outputs:
#   - frames saved under out_dir/frames/<video_id>/*.jpg
#   - audio saved as out_dir/audio/<video_id>.wav
# """
# import argparse
# from pathlib import Path
# import cv2
# import numpy as np
# from utils import run_ffmpeg_extract_audio, list_videos
# import os

# def sample_frames(video_path, out_dir, num_frames=16):
#     cap = cv2.VideoCapture(str(video_path))
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total <= 0:
#         cap.release()
#         raise RuntimeError(f"Could not read frames from {video_path}")
#     indices = np.linspace(0, total-1, num_frames).astype(int)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     for i, idx in enumerate(indices):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
#         ret, frame = cap.read()
#         if not ret:
#             # write a black image placeholder
#             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
#             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
#             frame = np.zeros((h, w, 3), dtype=np.uint8)
#         out_path = out_dir / f"{i:04d}.jpg"
#         cv2.imwrite(str(out_path), frame)
#     cap.release()

# def process_one(video_path, out_root, frames_per_video=16):
#     video_path = Path(video_path)
#     vid_id = video_path.stem
#     frames_out = Path(out_root) / "frames" / vid_id
#     audio_out = Path(out_root) / "audio" / f"{vid_id}.wav"
#     if frames_out.exists() and audio_out.exists():
#         print(f"Skipping existing: {vid_id}")
#         return
#     print(f"Processing {vid_id} ...")
#     frames_out.mkdir(parents=True, exist_ok=True)
#     sample_frames(video_path, frames_out, num_frames=frames_per_video)
#     run_ffmpeg_extract_audio(video_path, audio_out)
#     print("Done:", vid_id)

# def main(args):
#     input_dir = Path(args.input_dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     vids = list(list_videos(input_dir))
#     if not vids:
#         print("No videos found in", input_dir)
#         return
#     for v in vids:
#         try:
#             process_one(v, out_dir, frames_per_video=args.frames_per_video)
#         except Exception as e:
#             print("Failed:", v, e)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--input-dir", required=True, help="Directory with raw videos")
#     p.add_argument("--out-dir", required=True, help="Destination processed dir")
#     p.add_argument("--frames-per-video", type=int, default=16)
#     args = p.parse_args()
#     main(args)





"""
Extract frames and audio from videos.

Outputs:
    out_dir/
        frames/
            <video_id>/
                0000.jpg
                0001.jpg
                ...
        audio/
            <video_id>.wav
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from utils import run_ffmpeg_extract_audio, list_videos


def sample_frames(video_path, out_dir, num_frames=16, image_size=224):
    """
    Extract evenly spaced frames efficiently.

    Frames are resized to 224x224 to reduce storage and processing time.
    """

    video_path = Path(video_path)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames found in: {video_path}")

    # Select frame positions evenly across the video
    indices = np.linspace(
        0,
        total - 1,
        min(num_frames, total)
    ).astype(int)

    indices_set = set(indices)

    frame_number = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number in indices_set:

            # Resize to 224x224
            frame = cv2.resize(
                frame,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            )

            out_path = out_dir / f"{saved:04d}.jpg"

            cv2.imwrite(
                str(out_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            saved += 1

            if saved >= len(indices):
                break

        frame_number += 1

    cap.release()

    if saved == 0:
        raise RuntimeError(
            f"Could not extract frames from: {video_path}"
        )

    print(f"  Frames: {saved}/{len(indices)}")


def process_one(video_path, out_root, frames_per_video=16):
    """
    Process one video:
      1. Extract frames
      2. Extract audio
    """

    video_path = Path(video_path)

    vid_id = video_path.stem

    out_root = Path(out_root)

    frames_out = out_root / "frames" / vid_id
    audio_out = out_root / "audio" / f"{vid_id}.wav"

    # Skip only if BOTH outputs already exist
    existing_frames = list(frames_out.glob("*.jpg")) if frames_out.exists() else []

    if len(existing_frames) >= frames_per_video and audio_out.exists():
        print(f"Skipping existing: {vid_id}")
        return

    print(f"\nProcessing {vid_id} ...")

    # Create directories
    frames_out.mkdir(parents=True, exist_ok=True)
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    # Extract frames
    if len(existing_frames) < frames_per_video:
        sample_frames(
            video_path,
            frames_out,
            num_frames=frames_per_video,
            image_size=224
        )
    else:
        print("  Frames already exist")

    # Extract audio
    if not audio_out.exists():
        run_ffmpeg_extract_audio(
            video_path,
            audio_out
        )
        print("  Audio extracted")
    else:
        print("  Audio already exists")

    print(f"Done: {vid_id}")


def main(args):

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    videos = list(list_videos(input_dir))

    if not videos:
        print("No videos found in:", input_dir)
        return

    print(f"Found {len(videos)} video(s)")

    for i, video in enumerate(videos, start=1):

        print(f"\n[{i}/{len(videos)}]")

        try:

            process_one(
                video,
                out_dir,
                frames_per_video=args.frames_per_video
            )

        except Exception as e:

            print(f"Failed: {video}")
            print("Reason:", e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing raw videos"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Destination processed directory"
    )

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=16,
        help="Number of frames to extract per video"
    )

    args = parser.parse_args()

    main(args)