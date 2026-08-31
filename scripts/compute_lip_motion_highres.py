# import cv2
# import mediapipe as mp
# import numpy as np
# from pathlib import Path


# # ============================================================
# # CONFIGURATION
# # ============================================================

# VIDEO_DIR = Path("data/raw/test")
# OUTPUT_DIR = Path("data/processed/test/lips_highres")

# MODEL_PATH = "models/mediapipe/face_landmarker.task"

# SAMPLES_PER_VIDEO = 32

# # Minimum confidence
# MIN_FACE_DETECTION_CONFIDENCE = 0.1
# MIN_FACE_PRESENCE_CONFIDENCE = 0.1

# # Face crop margin
# FACE_MARGIN = 0.25


# # ============================================================
# # MEDIAPIPE
# # ============================================================

# def create_landmarker():

#     BaseOptions = mp.tasks.BaseOptions
#     FaceLandmarker = mp.tasks.vision.FaceLandmarker
#     Options = mp.tasks.vision.FaceLandmarkerOptions
#     RunningMode = mp.tasks.vision.RunningMode

#     options = Options(
#         base_options=BaseOptions(
#             model_asset_path=MODEL_PATH
#         ),
#         running_mode=RunningMode.IMAGE,
#         num_faces=1,
#         min_face_detection_confidence=MIN_FACE_DETECTION_CONFIDENCE,
#         min_face_presence_confidence=MIN_FACE_PRESENCE_CONFIDENCE,
#         min_tracking_confidence=0.1
#     )

#     return FaceLandmarker.create_from_options(options)


# # ============================================================
# # FACE BOUNDING BOX FROM LANDMARKS
# # ============================================================

# def get_face_box(face_landmarks, width, height):

#     xs = np.array([
#         landmark.x for landmark in face_landmarks
#     ])

#     ys = np.array([
#         landmark.y for landmark in face_landmarks
#     ])

#     x1 = int(np.min(xs) * width)
#     y1 = int(np.min(ys) * height)

#     x2 = int(np.max(xs) * width)
#     y2 = int(np.max(ys) * height)

#     return x1, y1, x2, y2


# # ============================================================
# # ADD MARGIN
# # ============================================================

# def expand_box(box, width, height):

#     x1, y1, x2, y2 = box

#     w = x2 - x1
#     h = y2 - y1

#     mx = int(w * FACE_MARGIN)
#     my = int(h * FACE_MARGIN)

#     x1 = max(0, x1 - mx)
#     y1 = max(0, y1 - my)

#     x2 = min(width, x2 + mx)
#     y2 = min(height, y2 + my)

#     return x1, y1, x2, y2


# # ============================================================
# # LIP MOTION
# # ============================================================

# # MediaPipe Face Mesh / Face Landmarker lip landmark indices
# UPPER_LIP = [
#     13,
#     82,
#     312,
#     311,
#     308
# ]

# LOWER_LIP = [
#     14,
#     87,
#     317,
#     402,
#     378
# ]


# def calculate_lip_opening(face_landmarks):

#     upper_y = np.mean([
#         face_landmarks[i].y
#         for i in UPPER_LIP
#     ])

#     lower_y = np.mean([
#         face_landmarks[i].y
#         for i in LOWER_LIP
#     ])

#     return abs(lower_y - upper_y)


# # ============================================================
# # PROCESS ONE VIDEO
# # ============================================================

# def process_video(video_path, landmarker):

#     video_id = video_path.stem

#     cap = cv2.VideoCapture(str(video_path))

#     if not cap.isOpened():

#         print(video_id, ": could not open video")

#         return None

#     total_frames = int(
#         cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     )

#     if total_frames <= 0:

#         cap.release()

#         print(video_id, ": invalid frame count")

#         return None

#     # Select evenly distributed frames
#     frame_indices = np.linspace(
#         0,
#         total_frames - 1,
#         SAMPLES_PER_VIDEO
#     ).astype(int)

#     lip_values = []

#     detected = 0

#     for frame_index in frame_indices:

#         cap.set(
#             cv2.CAP_PROP_POS_FRAMES,
#             int(frame_index)
#         )

#         ok, frame = cap.read()

#         if not ok:

#             lip_values.append(np.nan)

#             continue

#         height, width = frame.shape[:2]

#         # ----------------------------------------------------
#         # FIRST TRY: FULL ORIGINAL FRAME
#         # ----------------------------------------------------

#         rgb = cv2.cvtColor(
#             frame,
#             cv2.COLOR_BGR2RGB
#         )

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=rgb
#         )

#         result = landmarker.detect(mp_image)

#         face = None

#         if result.face_landmarks:

#             face = result.face_landmarks[0]

#         # ----------------------------------------------------
#         # SECOND TRY: CENTER CROP
#         # ----------------------------------------------------

#         # if face is None:

#         #     crop = frame[
#         #         :,
#         #         width // 4:
#         #         3 * width // 4
#         #     ]

#         #     crop_rgb = cv2.cvtColor(
#         #         crop,
#         #         cv2.COLOR_BGR2RGB
#         #     )

#         #     crop_image = mp.Image(
#         #         image_format=mp.ImageFormat.SRGB,
#         #         data=crop_rgb
#         #     )

#         #     crop_result = landmarker.detect(
#         #         crop_image
#         #     )

#         #     if crop_result.face_landmarks:

#         #         face = crop_result.face_landmarks[0]

#         # # ----------------------------------------------------
#         # # THIRD TRY: RIGHT HALF
#         # # ----------------------------------------------------

#         # if face is None:

#         #     crop = frame[
#         #         :,
#         #         width // 2:
#         #     ]

#         #     crop_rgb = cv2.cvtColor(
#         #         crop,
#         #         cv2.COLOR_BGR2RGB
#         #     )

#         #     crop_image = mp.Image(
#         #         image_format=mp.ImageFormat.SRGB,
#         #         data=crop_rgb
#         #     )

#         #     crop_result = landmarker.detect(
#         #         crop_image
#         #     )

#         #     if crop_result.face_landmarks:

#         #         face = crop_result.face_landmarks[0]

#         # ----------------------------------------------------
#         # NO FACE
#         # ----------------------------------------------------


#         # ----------------------------------------------------
# # FALLBACK: OVERLAPPING CROPS
# # ----------------------------------------------------

# if face is None:

#     crop_regions = [
#         ("left", 0.0, 0.60),
#         ("center", 0.20, 0.80),
#         ("right", 0.40, 1.0)
#     ]

#     for name, start_ratio, end_ratio in crop_regions:

#         x1 = int(width * start_ratio)
#         x2 = int(width * end_ratio)

#         crop = frame[:, x1:x2]

#         # Try original crop
#         crop_rgb = cv2.cvtColor(
#             crop,
#             cv2.COLOR_BGR2RGB
#         )

#         crop_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=crop_rgb
#         )

#         crop_result = landmarker.detect(
#             crop_image
#         )

#         if crop_result.face_landmarks:

#             face = crop_result.face_landmarks[0]
#             break

#         # Try 1.5× upscaled crop
#         up = cv2.resize(
#             crop,
#             None,
#             fx=1.5,
#             fy=1.5,
#             interpolation=cv2.INTER_CUBIC
#         )

#         up_rgb = cv2.cvtColor(
#             up,
#             cv2.COLOR_BGR2RGB
#         )

#         up_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=up_rgb
#         )

#         up_result = landmarker.detect(
#             up_image
#         )

#         if up_result.face_landmarks:

#             face = up_result.face_landmarks[0]
#             break

#         if face is None:

#             lip_values.append(np.nan)

#             continue

#         # ----------------------------------------------------
#         # LIP MOTION
#         # ----------------------------------------------------

#         lip_opening = calculate_lip_opening(
#             face
#         )

#         lip_values.append(
#             float(lip_opening)
#         )

#         detected += 1

#     cap.release()

#     return np.array(
#         lip_values,
#         dtype=np.float32
#     ), detected


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     videos = list(
#         VIDEO_DIR.rglob("*.mp4")
#     )

# #     videos = [
# #     VIDEO_DIR / "FAKE" / "ckwdmymgfn.mp4"
# # ]

#     print("=" * 60)
#     print("HIGH-RESOLUTION LIP MOTION EXTRACTION")
#     print("=" * 60)

#     print(
#         "Videos found:",
#         len(videos)
#     )

#     landmarker = create_landmarker()

#     total = 0
#     successful = 0

#     try:

#         for video_path in videos:

#             result = process_video(
#                 video_path,
#                 landmarker
#             )

#             if result is None:

#                 continue

#             lip_values, detected = result

#             output_file = (
#                 OUTPUT_DIR /
#                 f"{video_path.stem}.npy"
#             )

#             np.save(
#                 output_file,
#                 lip_values
#             )

#             total += 1

#             if detected >= 16:
#                 successful += 1

#             print(
#                 f"{video_path.stem}: "
#                 f"{detected}/{SAMPLES_PER_VIDEO} "
#                 f"faces detected"
#             )

#     finally:

#         landmarker.close()

#     print()
#     print("=" * 60)
#     print("COMPLETED")
#     print("=" * 60)

#     print(
#         "Videos processed:",
#         total
#     )

#     print(
#         "Videos with >=16 detections:",
#         successful
#     )

#     print(
#         "Output:",
#         OUTPUT_DIR
#     )


# if __name__ == "__main__":
#     main()






import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_DIR = Path("data/raw/test")
OUTPUT_DIR = Path("data/processed/test/lips_highres")

MODEL_PATH = "models/mediapipe/face_landmarker.task"

SAMPLES_PER_VIDEO = 32

# MediaPipe confidence
MIN_FACE_DETECTION_CONFIDENCE = 0.1
MIN_FACE_PRESENCE_CONFIDENCE = 0.1
MIN_TRACKING_CONFIDENCE = 0.1

# ============================================================
# MEDIAPIPE
# ============================================================

def create_landmarker():

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    Options = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = Options(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1,

        min_face_detection_confidence=(
            MIN_FACE_DETECTION_CONFIDENCE
        ),

        min_face_presence_confidence=(
            MIN_FACE_PRESENCE_CONFIDENCE
        ),

        min_tracking_confidence=(
            MIN_TRACKING_CONFIDENCE
        )
    )

    return FaceLandmarker.create_from_options(options)


# ============================================================
# DETECT FACE
# ============================================================

def detect_face(landmarker, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.face_landmarks:

        return result.face_landmarks[0]

    return None


# ============================================================
# ROBUST FACE DETECTION
# ============================================================

def detect_face_robust(frame, landmarker):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # METHOD 1: FULL FRAME
    # --------------------------------------------------------

    face = detect_face(
        landmarker,
        frame
    )

    if face is not None:

        return face, "full"


    # --------------------------------------------------------
    # METHOD 2: OVERLAPPING CROPS
    # --------------------------------------------------------

    crop_regions = [

        ("left", 0.00, 0.60),

        ("center", 0.20, 0.80),

        ("right", 0.40, 1.00)

    ]

    for name, start_ratio, end_ratio in crop_regions:

        x1 = int(width * start_ratio)
        x2 = int(width * end_ratio)

        crop = frame[:, x1:x2]

        if crop.size == 0:
            continue


        # ----------------------------------------------------
        # ORIGINAL CROP
        # ----------------------------------------------------

        face = detect_face(
            landmarker,
            crop
        )

        if face is not None:

            return face, f"{name}_crop"


        # ----------------------------------------------------
        # UPSCALED CROP
        # ----------------------------------------------------

        upscaled = cv2.resize(
            crop,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        face = detect_face(
            landmarker,
            upscaled
        )

        if face is not None:

            return face, f"{name}_upscaled"


    # --------------------------------------------------------
    # METHOD 3: FULL FRAME UPSCALED
    # --------------------------------------------------------

    upscaled = cv2.resize(
        frame,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    face = detect_face(
        landmarker,
        upscaled
    )

    if face is not None:

        return face, "full_upscaled"


    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    return None, "none"


# ============================================================
# LIP LANDMARK INDICES
# ============================================================

UPPER_LIP = [
    13,
    82,
    312,
    311,
    308
]

LOWER_LIP = [
    14,
    87,
    317,
    402,
    378
]


# ============================================================
# CALCULATE LIP OPENING
# ============================================================

def calculate_lip_opening(face_landmarks):

    upper_y = np.mean([
        face_landmarks[i].y
        for i in UPPER_LIP
    ])

    lower_y = np.mean([
        face_landmarks[i].y
        for i in LOWER_LIP
    ])

    return abs(
        lower_y - upper_y
    )


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, landmarker):

    video_id = video_path.stem

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            video_id,
            ": could not open video"
        )

        return None


    # --------------------------------------------------------
    # TOTAL FRAMES
    # --------------------------------------------------------

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        print(
            video_id,
            ": invalid frame count"
        )

        return None


    # --------------------------------------------------------
    # SELECT 32 EVENLY DISTRIBUTED FRAMES
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SAMPLES_PER_VIDEO
    ).astype(int)


    lip_values = []

    detected = 0

    detection_methods = {}


    # --------------------------------------------------------
    # PROCESS FRAMES
    # --------------------------------------------------------

    for frame_index in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        ok, frame = cap.read()

        if not ok:

            lip_values.append(
                np.nan
            )

            continue


        # ----------------------------------------------------
        # ROBUST FACE DETECTION
        # ----------------------------------------------------

        face, method = detect_face_robust(
            frame,
            landmarker
        )


        # Count detection method

        detection_methods[method] = (
            detection_methods.get(method, 0) + 1
        )


        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if face is None:

            lip_values.append(
                np.nan
            )

            continue


        # ----------------------------------------------------
        # CALCULATE LIP OPENING
        # ----------------------------------------------------

        lip_opening = calculate_lip_opening(
            face
        )


        lip_values.append(
            float(lip_opening)
        )

        detected += 1


    cap.release()


    # --------------------------------------------------------
    # PRINT METHOD STATISTICS
    # --------------------------------------------------------

    method_text = ", ".join(
        f"{k}={v}"
        for k, v in detection_methods.items()
    )


    print(
        f"{video_id}: "
        f"{detected}/{SAMPLES_PER_VIDEO} "
        f"faces detected"
    )

    print(
        f"    Methods: {method_text}"
    )


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return (
        np.array(
            lip_values,
            dtype=np.float32
        ),
        detected
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # FIND ALL VIDEOS
    # --------------------------------------------------------

    videos = list(
        VIDEO_DIR.rglob("*.mp4")
    )


    # --------------------------------------------------------
    # OPTIONAL: TEST ONLY ONE VIDEO
    # --------------------------------------------------------

    # Uncomment this if you want to test one video first.
    #
    # videos = [
    #     VIDEO_DIR / "FAKE" / "ckwdmymgfn.mp4"
    # ]


    print("=" * 60)
    print(
        "HIGH-RESOLUTION LIP MOTION EXTRACTION"
    )
    print("=" * 60)

    print(
        "Videos found:",
        len(videos)
    )


    # --------------------------------------------------------
    # CREATE MEDIAPIPE LANDMARKER
    # --------------------------------------------------------

    landmarker = create_landmarker()


    total = 0
    successful = 0


    try:

        for video_path in videos:

            result = process_video(
                video_path,
                landmarker
            )

            if result is None:

                continue


            lip_values, detected = result


            # ------------------------------------------------
            # OUTPUT FILE
            # ------------------------------------------------

            output_file = (
                OUTPUT_DIR /
                f"{video_path.stem}.npy"
            )


            np.save(
                output_file,
                lip_values
            )


            total += 1


            # ------------------------------------------------
            # SUCCESS CRITERIA
            # ------------------------------------------------

            if detected >= 16:

                successful += 1


    finally:

        landmarker.close()


    # ========================================================
    # SUMMARY
    # ========================================================

    print()

    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)

    print(
        "Videos processed:",
        total
    )

    print(
        "Videos with >=16 detections:",
        successful
    )

    print(
        "Output:",
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()