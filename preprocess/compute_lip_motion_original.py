from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


UPPER_LIP = [13, 14]
LOWER_LIP = [17, 18]

FACE_TOP = 10
FACE_BOTTOM = 152


def calculate_lip_opening(landmarks):

    def point(index):
        p = landmarks[index]
        return np.array([p.x, p.y])

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


def process_video(video_path, output_file, model_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path)
        return False

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return False

    # Sample 32 frames instead of 16
    num_samples = min(32, total_frames)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_samples
    ).astype(int)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    lip_motion = []
    detected = 0

    with FaceLandmarker.create_from_options(options) as landmarker:

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                lip_motion.append(np.nan)
                continue

            # Resize large frames for faster processing
            height, width = frame.shape[:2]

            scale = 640 / max(width, height)

            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (
                        int(width * scale),
                        int(height * scale)
                    )
                )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if not result.face_landmarks:

                # NaN = face not detected
                lip_motion.append(np.nan)

                continue

            landmarks = result.face_landmarks[0]

            value = calculate_lip_opening(
                landmarks
            )

            lip_motion.append(
                float(value)
            )

            detected += 1

    cap.release()

    values = np.array(
        lip_motion,
        dtype=np.float32
    )

    np.save(
        output_file,
        values
    )

    video_id = Path(video_path).stem

    print(
        f"{video_id}: "
        f"{detected}/{num_samples} faces detected"
    )

    return True


def main(args):

    videos_root = Path(args.videos_dir)

    output_root = Path(args.out_dir) / "lips"

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = Path(args.model)

    if not model_path.exists():

        print(
            "ERROR: MediaPipe model not found:"
        )

        print(model_path)

        return

    # Process FAKE and REAL folders
    for label_dir in sorted(videos_root.iterdir()):

        if not label_dir.is_dir():
            continue

        for video_path in sorted(
            label_dir.glob("*.mp4")
        ):

            output_file = (
                output_root /
                f"{video_path.stem}.npy"
            )

            try:

                process_video(
                    video_path,
                    output_file,
                    str(model_path)
                )

            except Exception as e:

                print(
                    "Failed:",
                    video_path.name,
                    e
                )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--videos-dir",
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