# """
# Compute a lip-opening time series per video:
# - Use MediaPipe FaceMesh to detect mouth keypoints.
# - For each frame, compute vertical lip opening distance normalized by face size.
# - Save as numpy sequence to out_dir/lips/<video_id>.npy
# """
# from pathlib import Path
# import cv2
# import numpy as np
# import mediapipe as mp
# import argparse

# mp_face = mp.solutions.face_mesh

# # Landmarks chosen are approximate; MediaPipe landmarks are indexed 0..468
# # We'll use a small set of mouth landmarks for robust estimate
# TOP_LIP_IDX = [13, 14, 15]    # approximate upper lip region
# BOT_LIP_IDX = [17, 18, 19]    # approximate lower lip region
# FACE_TOP_IDX = 10
# FACE_BOTTOM_IDX = 152

# def lip_opening_from_frame(image, face_mesh):
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     if not results.multi_face_landmarks:
#         return None
#     lm = results.multi_face_landmarks[0]
#     h, w = image.shape[:2]
#     # convert some landmark indices to (x,y)
#     def idx_to_xy(i):
#         p = lm.landmark[i]
#         return np.array([p.x * w, p.y * h])
#     top = np.mean([idx_to_xy(i) for i in TOP_LIP_IDX], axis=0)
#     bot = np.mean([idx_to_xy(i) for i in BOT_LIP_IDX], axis=0)
#     face_top = idx_to_xy(FACE_TOP_IDX)[1]
#     face_bottom = idx_to_xy(FACE_BOTTOM_IDX)[1]
#     face_h = abs(face_bottom - face_top) + 1e-9
#     return (bot[1] - top[1]) / face_h

# def process_frames_dir(frames_dir, out_path):
#     frames = sorted(list(Path(frames_dir).glob("*.jpg")))
#     lip_ts = []
#     with mp_face.FaceMesh(static_image_mode=True) as fm:
#         for f in frames:
#             img = cv2.imread(str(f))
#             if img is None:
#                 lip_ts.append(0.0)
#                 continue
#             val = lip_opening_from_frame(img, fm)
#             lip_ts.append(float(val) if val is not None else 0.0)
#     np.save(out_path, np.array(lip_ts, dtype=np.float32))

# def main(args):
#     frames_root = Path(args.frames_dir)
#     out_root = Path(args.out_dir) / "lips"
#     out_root.mkdir(parents=True, exist_ok=True)
#     for vid in sorted(frames_root.iterdir()):
#         if not vid.is_dir():
#             continue
#         out_path = out_root / f"{vid.name}.npy"
#         try:
#             process_frames_dir(vid, out_path)
#             print("lips saved", vid.name)
#         except Exception as e:
#             print("failed lips for", vid.name, e)

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--frames-dir", required=True, help="data/processed/frames")
#     p.add_argument("--out-dir", required=True, help="base output directory (will write ./lips/)")
#     args = p.parse_args()
#     main(args)



from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# MediaPipe mouth landmarks
UPPER_LIP = [13, 14]
LOWER_LIP = [17, 18]

FACE_TOP = 10
FACE_BOTTOM = 152


def calculate_lip_opening(landmarks, width, height):
    """
    Calculate normalized distance between upper and lower lip.

    Larger value -> mouth more open.
    Smaller value -> mouth more closed.
    """

    def point(index):
        p = landmarks[index]
        return np.array([
            p.x * width,
            p.y * height
        ])

    upper = np.mean(
        [point(i) for i in UPPER_LIP],
        axis=0
    )

    lower = np.mean(
        [point(i) for i in LOWER_LIP],
        axis=0
    )

    face_top = point(FACE_TOP)[1]
    face_bottom = point(FACE_BOTTOM)[1]

    face_height = abs(face_bottom - face_top) + 1e-6

    lip_distance = abs(lower[1] - upper[1])

    return lip_distance / face_height


def process_video_frames(frames_dir, output_file, model_path):

    frames = sorted(
        Path(frames_dir).glob("*.jpg")
    )

    if not frames:
        print("No frames found:", frames_dir)
        return

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    lip_motion = []

    with FaceLandmarker.create_from_options(options) as landmarker:

        for frame_path in frames:

            image = cv2.imread(str(frame_path))

            if image is None:
                lip_motion.append(0.0)
                continue

            height, width = image.shape[:2]

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if not result.face_landmarks:
                lip_motion.append(0.0)
                continue

            landmarks = result.face_landmarks[0]

            value = calculate_lip_opening(
                landmarks,
                width,
                height
            )

            lip_motion.append(float(value))

    np.save(
        output_file,
        np.array(lip_motion, dtype=np.float32)
    )


def main(args):

    frames_root = Path(args.frames_dir)

    output_root = Path(args.out_dir) / "lips"

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = Path(args.model)

    if not model_path.exists():
        print("ERROR: MediaPipe model not found:")
        print(model_path)
        return

    for video_dir in sorted(frames_root.iterdir()):

        if not video_dir.is_dir():
            continue

        output_file = (
            output_root /
            f"{video_dir.name}.npy"
        )

        try:

            process_video_frames(
                video_dir,
                output_file,
                str(model_path)
            )

            print(
                "Lip motion saved:",
                video_dir.name
            )

        except Exception as e:

            print(
                "Failed:",
                video_dir.name,
                e
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frames-dir",
        required=True
    )

    parser.add_argument(
        "--out-dir",
        required=True
    )

    parser.add_argument(
        "--model",
        default="models/mediapipe/face_landmarker.task"
    )

    args = parser.parse_args()

    main(args)