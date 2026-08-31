# import sys
# from pathlib import Path

# import cv2
# import librosa
# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms

# # Make project root importable
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from models.video_model import VideoDeepfakeModel
# from models.audio_model import AudioDeepfakeModel


# # ============================================================
# # CONFIGURATION
# # ============================================================

# DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# # ------------------------------------------------------------
# # Video
# # ------------------------------------------------------------

# VIDEO_CHECKPOINT = (
#     PROJECT_ROOT
#     / "checkpoints"
#     / "video_deepfake"
#     / "efficientnet_b0_best.pth"
# )

# VIDEO_NUM_FRAMES = 8
# VIDEO_IMAGE_SIZE = 160

# # ------------------------------------------------------------
# # Audio
# # ------------------------------------------------------------

# AUDIO_CHECKPOINT = (
#     PROJECT_ROOT
#     / "checkpoints"
#     / "audio_deepfake_best.pth"
# )

# AUDIO_SAMPLE_RATE = 16000
# AUDIO_DURATION = 4

# AUDIO_NUM_SAMPLES = (
#     AUDIO_SAMPLE_RATE
#     * AUDIO_DURATION
# )

# N_MELS = 128
# N_FFT = 1024
# HOP_LENGTH = 256

# # ------------------------------------------------------------
# # Synchronization
# # ------------------------------------------------------------

# SYNC_MAX_LAG = 5
# SYNC_MIN_VALID_FRAMES = 8


# # ------------------------------------------------------------
# # Fusion
# # ------------------------------------------------------------

# VIDEO_WEIGHT = 0.50
# AUDIO_WEIGHT = 0.30
# SYNC_WEIGHT = 0.20

# FUSION_THRESHOLD = 0.46


# # ============================================================
# # DEVICE
# # ============================================================

# print("=" * 60)
# print("MULTIMODAL DEEPFAKE INFERENCE PIPELINE")
# print("=" * 60)

# print("Device:", DEVICE)


# # ============================================================
# # VIDEO TRANSFORM
# # ============================================================

# video_transform = transforms.Compose([

#     transforms.Resize(
#         (
#             VIDEO_IMAGE_SIZE,
#             VIDEO_IMAGE_SIZE
#         )
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         mean=[
#             0.485,
#             0.456,
#             0.406
#         ],
#         std=[
#             0.229,
#             0.224,
#             0.225
#         ]
#     )
# ])


# # ============================================================
# # LOAD VIDEO MODEL
# # ============================================================

# def load_video_model():

#     print("Loading video model...")

#     model = VideoDeepfakeModel()

#     checkpoint = torch.load(
#         VIDEO_CHECKPOINT,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     model.load_state_dict(
#         checkpoint["model_state_dict"]
#     )

#     model.to(DEVICE)

#     model.eval()

#     print("Video model loaded.")

#     return model


# # ============================================================
# # LOAD AUDIO MODEL
# # ============================================================

# def load_audio_model():

#     print("Loading audio model...")

#     model = AudioDeepfakeModel()

#     checkpoint = torch.load(
#         AUDIO_CHECKPOINT,
#         map_location=DEVICE,
#         weights_only=False
#     )

#     model.load_state_dict(
#         checkpoint["model_state_dict"]
#     )

#     model.to(DEVICE)

#     model.eval()

#     print("Audio model loaded.")

#     return model


# # ============================================================
# # LOAD MODELS ONCE
# # ============================================================

# VIDEO_MODEL = load_video_model()

# AUDIO_MODEL = load_audio_model()


# # ============================================================
# # VIDEO FRAME EXTRACTION
# # ============================================================

# def extract_video_frames(
#     video_path,
#     num_frames=VIDEO_NUM_FRAMES
# ):

#     cap = cv2.VideoCapture(
#         str(video_path)
#     )

#     if not cap.isOpened():

#         raise RuntimeError(
#             f"Could not open video: {video_path}"
#         )

#     total_frames = int(
#         cap.get(
#             cv2.CAP_PROP_FRAME_COUNT
#         )
#     )

#     if total_frames <= 0:

#         cap.release()

#         raise RuntimeError(
#             "Could not determine video frame count."
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

#         image = Image.fromarray(
#             frame
#         )

#         frames.append(image)

#     cap.release()

#     if len(frames) == 0:

#         raise RuntimeError(
#             "No frames could be extracted."
#         )

#     while len(frames) < num_frames:

#         frames.append(
#             frames[-1].copy()
#         )

#     return frames


# # ============================================================
# # VIDEO PREDICTION
# # ============================================================

# def predict_video(video_path):

#     frames = extract_video_frames(
#         video_path
#     )

#     tensors = []

#     for frame in frames:

#         tensor = video_transform(
#             frame
#         )

#         tensors.append(tensor)

#     frames_tensor = torch.stack(
#         tensors
#     )

#     # (frames, channels, height, width)
#     #
#     # Add batch dimension
#     #
#     # (1, frames, channels, height, width)

#     frames_tensor = (
#         frames_tensor
#         .unsqueeze(0)
#         .to(DEVICE)
#     )

#     with torch.no_grad():

#         outputs = VIDEO_MODEL(
#             frames_tensor
#         )

#         probabilities = torch.softmax(
#             outputs,
#             dim=1
#         )

#     # Project mapping:
#     #
#     # 0 = FAKE
#     # 1 = REAL

#     fake_probability = float(
#         probabilities[0, 0].item()
#     )

#     real_probability = float(
#         probabilities[0, 1].item()
#     )

#     if fake_probability > real_probability:

#         prediction = "FAKE"

#     else:

#         prediction = "REAL"

#     return {
#         "prediction": prediction,
#         "fake_probability": fake_probability,
#         "real_probability": real_probability,
#         "frames_used": len(frames)
#     }


# # ============================================================
# # AUDIO EXTRACTION
# # ============================================================

# def extract_audio(
#     video_path,
#     output_audio_path
# ):

#     import subprocess

#     command = [

#         "ffmpeg",

#         "-y",

#         "-i",
#         str(video_path),

#         "-vn",

#         "-ac",
#         "1",

#         "-ar",
#         str(AUDIO_SAMPLE_RATE),

#         str(output_audio_path)
#     ]

#     result = subprocess.run(
#         command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True
#     )

#     if result.returncode != 0:

#         raise RuntimeError(
#             "FFmpeg audio extraction failed:\n"
#             + result.stderr
#         )

#     return output_audio_path


# # ============================================================
# # AUDIO → MEL SPECTROGRAM
# # ============================================================

# def create_mel_spectrogram(
#     audio_path
# ):

#     waveform, _ = librosa.load(
#         str(audio_path),
#         sr=AUDIO_SAMPLE_RATE,
#         mono=True
#     )

#     # --------------------------------------------------------
#     # Match AudioMelDataset exactly
#     # --------------------------------------------------------

#     if len(waveform) < AUDIO_NUM_SAMPLES:

#         waveform = librosa.util.fix_length(
#             waveform,
#             size=AUDIO_NUM_SAMPLES
#         )

#     else:

#         waveform = waveform[
#             :AUDIO_NUM_SAMPLES
#         ]

#     mel = librosa.feature.melspectrogram(
#         y=waveform,
#         sr=AUDIO_SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     mel_db = librosa.power_to_db(
#         mel,
#         ref=np.max
#     )

#     mel_db = (
#         mel_db - mel_db.mean()
#     ) / (
#         mel_db.std() + 1e-8
#     )

#     mel = torch.tensor(
#         mel_db,
#         dtype=torch.float32
#     )

#     # (128, time)
#     #
#     # Add channel
#     #
#     # (1, 128, time)
#     #
#     # Add batch
#     #
#     # (1, 1, 128, time)

#     mel = mel.unsqueeze(0)
#     mel = mel.unsqueeze(0)

#     return mel


# # ============================================================
# # AUDIO PREDICTION
# # ============================================================

# def predict_audio(
#     audio_path
# ):

#     mel = create_mel_spectrogram(
#         audio_path
#     )

#     mel = mel.to(DEVICE)

#     with torch.no_grad():

#         outputs = AUDIO_MODEL(
#             mel
#         )

#         probabilities = torch.softmax(
#             outputs,
#             dim=1
#         )

#     # Project mapping:
#     #
#     # 0 = FAKE
#     # 1 = REAL

#     fake_probability = float(
#         probabilities[0, 0].item()
#     )

#     real_probability = float(
#         probabilities[0, 1].item()
#     )

#     if fake_probability > real_probability:

#         prediction = "FAKE"

#     else:

#         prediction = "REAL"

#     return {
#         "prediction": prediction,
#         "fake_probability": fake_probability,
#         "real_probability": real_probability
#     }


# # ============================================================
# # AUDIO ACTIVITY
# # ============================================================

# def calculate_audio_activity(
#     audio_path,
#     target_length
# ):

#     audio, _ = librosa.load(
#         str(audio_path),
#         sr=AUDIO_SAMPLE_RATE,
#         mono=True
#     )

#     if len(audio) == 0:

#         return np.zeros(
#             target_length,
#             dtype=np.float32
#         )

#     rms = librosa.feature.rms(
#         y=audio,
#         frame_length=1024,
#         hop_length=512
#     )[0]

#     rms_min = rms.min()
#     rms_max = rms.max()

#     if rms_max - rms_min > 1e-8:

#         rms = (
#             rms - rms_min
#         ) / (
#             rms_max - rms_min
#         )

#     else:

#         rms = np.zeros_like(rms)

#     x_old = np.linspace(
#         0,
#         1,
#         len(rms)
#     )

#     x_new = np.linspace(
#         0,
#         1,
#         target_length
#     )

#     activity = np.interp(
#         x_new,
#         x_old,
#         rms
#     )

#     return activity.astype(
#         np.float32
#     )


# # ============================================================
# # LIP MOTION
# # ============================================================

# def load_lip_motion(
#     lip_path
# ):

#     lip = np.load(
#         lip_path
#     ).astype(
#         np.float32
#     )

#     if len(lip) == 0:

#         return None

#     lip = np.where(
#         np.isfinite(lip),
#         lip,
#         np.nan
#     )

#     return lip


# # ============================================================
# # FILL MISSING LIP VALUES
# # ============================================================

# def fill_missing_lip_values(
#     lip
# ):

#     lip = np.asarray(
#         lip,
#         dtype=np.float32
#     )

#     valid = (
#         np.isfinite(lip)
#         &
#         (lip > 0)
#     )

#     valid_indices = np.where(
#         valid
#     )[0]

#     if len(valid_indices) < SYNC_MIN_VALID_FRAMES:

#         return None

#     if len(valid_indices) == len(lip):

#         return lip

#     all_indices = np.arange(
#         len(lip)
#     )

#     filled = np.interp(
#         all_indices,
#         valid_indices,
#         lip[valid_indices]
#     )

#     return filled.astype(
#         np.float32
#     )


# # ============================================================
# # NORMALIZATION
# # ============================================================

# def normalize(x):

#     x = np.asarray(
#         x,
#         dtype=np.float32
#     )

#     minimum = np.min(x)
#     maximum = np.max(x)

#     if maximum - minimum < 1e-8:

#         return np.zeros_like(x)

#     return (
#         x - minimum
#     ) / (
#         maximum - minimum
#     )


# # ============================================================
# # LAGGED CORRELATION
# # ============================================================

# def calculate_lagged_correlation(
#     lip,
#     audio,
#     max_lag=SYNC_MAX_LAG
# ):

#     lip = normalize(lip)
#     audio = normalize(audio)

#     best_correlation = -1.0
#     best_lag = 0

#     max_lag = min(
#         max_lag,
#         len(lip) - 2
#     )

#     for lag in range(
#         -max_lag,
#         max_lag + 1
#     ):

#         if lag > 0:

#             lip_part = lip[lag:]
#             audio_part = audio[:-lag]

#         elif lag < 0:

#             lip_part = lip[:lag]
#             audio_part = audio[-lag:]

#         else:

#             lip_part = lip
#             audio_part = audio

#         if len(lip_part) < 3:

#             continue

#         if (
#             np.std(lip_part) < 1e-8
#             or
#             np.std(audio_part) < 1e-8
#         ):

#             continue

#         correlation = np.corrcoef(
#             lip_part,
#             audio_part
#         )[0, 1]

#         if np.isnan(correlation):

#             continue

#         if correlation > best_correlation:

#             best_correlation = float(
#                 correlation
#             )

#             best_lag = lag

#     if best_correlation == -1.0:

#         return 0.0, 0

#     return (
#         best_correlation,
#         best_lag
#     )


# # ============================================================
# # SYNCHRONIZATION PREDICTION
# # ============================================================

# def predict_sync(
#     audio_path,
#     lips_path
# ):

#     if not lips_path.exists():

#         return {
#             "available": False,
#             "sync_score": 0.5,
#             "sync_fake_probability": 0.5,
#             "valid_lip_frames": 0,
#             "total_lip_frames": 0,
#             "best_lag": 0
#         }

#     lip = load_lip_motion(
#         lips_path
#     )

#     if lip is None:

#         return {
#             "available": False,
#             "sync_score": 0.5,
#             "sync_fake_probability": 0.5,
#             "valid_lip_frames": 0,
#             "total_lip_frames": 0,
#             "best_lag": 0
#         }

#     valid_mask = (
#         np.isfinite(lip)
#         &
#         (lip > 0)
#     )

#     valid_count = int(
#         np.sum(valid_mask)
#     )

#     total_count = len(lip)

#     if valid_count < SYNC_MIN_VALID_FRAMES:

#         return {
#             "available": False,
#             "sync_score": 0.5,
#             "sync_fake_probability": 0.5,
#             "valid_lip_frames": valid_count,
#             "total_lip_frames": total_count,
#             "best_lag": 0
#         }

#     lip_filled = fill_missing_lip_values(
#         lip
#     )

#     if lip_filled is None:

#         return {
#             "available": False,
#             "sync_score": 0.5,
#             "sync_fake_probability": 0.5,
#             "valid_lip_frames": valid_count,
#             "total_lip_frames": total_count,
#             "best_lag": 0
#         }

#     audio_activity = (
#         calculate_audio_activity(
#             audio_path,
#             len(lip_filled)
#         )
#     )

#     correlation, best_lag = (
#         calculate_lagged_correlation(
#             lip_filled,
#             audio_activity
#         )
#     )

#     sync_score = (
#         correlation + 1.0
#     ) / 2.0

#     sync_fake_probability = (
#         1.0 - sync_score
#     )

#     return {
#         "available": True,
#         "sync_score": float(sync_score),
#         "sync_fake_probability": float(
#             sync_fake_probability
#         ),
#         "correlation": float(
#             correlation
#         ),
#         "best_lag": int(
#             best_lag
#         ),
#         "valid_lip_frames": valid_count,
#         "total_lip_frames": total_count
#     }


# # ============================================================
# # MULTIMODAL FUSION
# # ============================================================

# def fuse_predictions(
#     video_result,
#     audio_result,
#     sync_result
# ):

#     video_score = (
#         video_result[
#             "fake_probability"
#         ]
#     )

#     audio_score = (
#         audio_result[
#             "fake_probability"
#         ]
#     )

#     sync_score = (
#         sync_result[
#             "sync_fake_probability"
#         ]
#     )

#     fusion_score = (

#         VIDEO_WEIGHT
#         * video_score

#         +

#         AUDIO_WEIGHT
#         * audio_score

#         +

#         SYNC_WEIGHT
#         * sync_score
#     )

#     if fusion_score >= FUSION_THRESHOLD:

#         prediction = "FAKE"

#     else:

#         prediction = "REAL"

#     return {
#         "prediction": prediction,
#         "fraud_risk": float(
#             fusion_score
#         ),
#         "fraud_risk_percent": float(
#             fusion_score * 100
#         )
#     }


# # ============================================================
# # COMPLETE PIPELINE
# # ============================================================

# def analyze_video(
#     video_path,
#     work_dir
# ):

#     video_path = Path(
#         video_path
#     )

#     work_dir = Path(
#         work_dir
#     )

#     work_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     # --------------------------------------------------------
#     # Video
#     # --------------------------------------------------------

#     video_result = predict_video(
#         video_path
#     )

#     # --------------------------------------------------------
#     # Audio
#     # --------------------------------------------------------

#     audio_path = (
#         work_dir /
#         "audio.wav"
#     )

#     extract_audio(
#         video_path,
#         audio_path
#     )

#     audio_result = predict_audio(
#         audio_path
#     )

#     # --------------------------------------------------------
#     # Sync
#     # --------------------------------------------------------
#     #
#     # The current sync system uses precomputed
#     # lips_highres .npy files.
#     #
#     # Therefore the API checks whether a matching
#     # lip-motion file already exists.
#     #
#     # Later we can add real-time MediaPipe
#     # lip extraction here.

#     lips_path = (
#         PROJECT_ROOT
#         / "data"
#         / "processed"
#         / "test"
#         / "lips_highres"
#         / f"{video_path.stem}.npy"
#     )

#     sync_result = predict_sync(
#         audio_path,
#         lips_path
#     )

#     # --------------------------------------------------------
#     # Fusion
#     # --------------------------------------------------------

#     fusion_result = fuse_predictions(
#         video_result,
#         audio_result,
#         sync_result
#     )

#     return {
#         "video": video_result,
#         "audio": audio_result,
#         "sync": sync_result,
#         "fusion": fusion_result
#     }








import sys
import subprocess
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.video_model import VideoDeepfakeModel
from models.audio_model import AudioDeepfakeModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VIDEO CONFIGURATION
# ============================================================

VIDEO_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "video_deepfake"
    / "efficientnet_b0_best.pth"
)

VIDEO_NUM_FRAMES = 16
VIDEO_IMAGE_SIZE = 160


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

AUDIO_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "audio_deepfake_best.pth"
)

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 4

AUDIO_NUM_SAMPLES = (
    AUDIO_SAMPLE_RATE
    * AUDIO_DURATION
)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# MULTIMODAL FUSION CONFIGURATION
# ============================================================

VIDEO_WEIGHT = 0.50
AUDIO_WEIGHT = 0.50

FUSION_THRESHOLD = 0.50


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("=" * 60)
print("MULTIMODAL AUDIO-VIDEO DEEPFAKE DETECTION")
print("=" * 60)

print("Device:", DEVICE)
print("Video weight:", VIDEO_WEIGHT)
print("Audio weight:", AUDIO_WEIGHT)
print("Fusion threshold:", FUSION_THRESHOLD)

print("=" * 60)


# ============================================================
# VIDEO TRANSFORM
# ============================================================

video_transform = transforms.Compose([

    transforms.Resize(
        (
            VIDEO_IMAGE_SIZE,
            VIDEO_IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD VIDEO MODEL
# ============================================================

def load_video_model():

    print("Loading video model...")

    if not VIDEO_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Video checkpoint not found:\n"
            f"{VIDEO_CHECKPOINT}"
        )

    model = VideoDeepfakeModel()

    checkpoint = torch.load(
        VIDEO_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("Video model loaded successfully.")

    return model


# ============================================================
# LOAD AUDIO MODEL
# ============================================================

def load_audio_model():

    print("Loading audio model...")

    if not AUDIO_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Audio checkpoint not found:\n"
            f"{AUDIO_CHECKPOINT}"
        )

    model = AudioDeepfakeModel()

    checkpoint = torch.load(
        AUDIO_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("Audio model loaded successfully.")

    return model


# ============================================================
# LOAD MODELS ONCE
# ============================================================

VIDEO_MODEL = load_video_model()

AUDIO_MODEL = load_audio_model()


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(
    video_path,
    num_frames=VIDEO_NUM_FRAMES
):

    video_path = Path(video_path)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine video frame count."
        )

    # --------------------------------------------------------
    # Select frames uniformly throughout the video
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # OpenCV BGR -> RGB
        # ----------------------------------------------------

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            frame
        )

        frames.append(image)

    cap.release()

    if len(frames) == 0:

        raise RuntimeError(
            "No frames could be extracted from video."
        )

    # --------------------------------------------------------
    # If fewer frames were extracted, duplicate last frame
    # --------------------------------------------------------

    while len(frames) < num_frames:

        frames.append(
            frames[-1].copy()
        )

    return frames


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(video_path):

    print()
    print("Extracting video frames...")

    frames = extract_video_frames(
        video_path
    )

    print(
        "Frames used:",
        len(frames)
    )

    tensors = []

    for frame in frames:

        tensor = video_transform(
            frame
        )

        tensors.append(
            tensor
        )

    # --------------------------------------------------------
    # Stack frames
    #
    # Before batch:
    #
    # (frames, channels, height, width)
    #
    # After unsqueeze:
    #
    # (1, frames, channels, height, width)
    # --------------------------------------------------------

    frames_tensor = torch.stack(
        tensors
    )

    frames_tensor = (
        frames_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = VIDEO_MODEL(
            frames_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # Project class mapping
    #
    # 0 = FAKE
    # 1 = REAL
    # --------------------------------------------------------

    fake_probability = float(
        probabilities[0, 0].item()
    )

    real_probability = float(
        probabilities[0, 1].item()
    )

    if fake_probability > real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    result = {

        "prediction":
            prediction,

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "frames_used":
            len(frames)
    }

    print(
        "Video prediction:",
        prediction
    )

    print(
        "Video fake probability:",
        f"{fake_probability * 100:.2f}%"
    )

    print(
        "Video real probability:",
        f"{real_probability * 100:.2f}%"
    )

    return result


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path,
    output_audio_path
):

    print()
    print("Extracting audio using FFmpeg...")

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(video_path),

        "-vn",

        "-ac",
        "1",

        "-ar",
        str(AUDIO_SAMPLE_RATE),

        str(output_audio_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg audio extraction failed:\n"
            + result.stderr
        )

    print(
        "Audio extracted:",
        output_audio_path
    )

    return output_audio_path


# ============================================================
# CREATE MEL SPECTROGRAM
# ============================================================

def create_mel_spectrogram(
    audio_path
):

    print("Creating Mel spectrogram...")

    waveform, _ = librosa.load(
        str(audio_path),
        sr=AUDIO_SAMPLE_RATE,
        mono=True
    )

    if len(waveform) == 0:

        raise RuntimeError(
            "Extracted audio contains no samples."
        )

    # --------------------------------------------------------
    # Match AudioMelDataset preprocessing
    # --------------------------------------------------------

    if len(waveform) < AUDIO_NUM_SAMPLES:

        waveform = librosa.util.fix_length(
            waveform,
            size=AUDIO_NUM_SAMPLES
        )

    else:

        waveform = waveform[
            :AUDIO_NUM_SAMPLES
        ]

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=AUDIO_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # --------------------------------------------------------
    # Convert power spectrogram to dB
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mel_db = (
        mel_db - mel_db.mean()
    ) / (
        mel_db.std() + 1e-8
    )

    mel_tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Shape:
    #
    # Original:
    # (128, time)
    #
    # Add channel:
    # (1, 128, time)
    #
    # Add batch:
    # (1, 1, 128, time)
    # --------------------------------------------------------

    mel_tensor = mel_tensor.unsqueeze(0)

    mel_tensor = mel_tensor.unsqueeze(0)

    print(
        "Mel spectrogram shape:",
        tuple(mel_tensor.shape)
    )

    return mel_tensor


# ============================================================
# AUDIO PREDICTION
# ============================================================

def predict_audio(
    audio_path
):

    print()
    print("Running audio model...")

    mel = create_mel_spectrogram(
        audio_path
    )

    mel = mel.to(DEVICE)

    with torch.no_grad():

        outputs = AUDIO_MODEL(
            mel
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    # --------------------------------------------------------
    # Project class mapping
    #
    # 0 = FAKE
    # 1 = REAL
    # --------------------------------------------------------

    fake_probability = float(
        probabilities[0, 0].item()
    )

    real_probability = float(
        probabilities[0, 1].item()
    )

    if fake_probability > real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    result = {

        "prediction":
            prediction,

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability
    }

    print(
        "Audio prediction:",
        prediction
    )

    print(
        "Audio fake probability:",
        f"{fake_probability * 100:.2f}%"
    )

    print(
        "Audio real probability:",
        f"{real_probability * 100:.2f}%"
    )

    return result


# ============================================================
# MULTIMODAL FUSION
# ============================================================

def fuse_predictions(
    video_result,
    audio_result
):

    # --------------------------------------------------------
    # Get fake probabilities from both modalities
    # --------------------------------------------------------

    video_score = float(
        video_result[
            "fake_probability"
        ]
    )

    audio_score = float(
        audio_result[
            "fake_probability"
        ]
    )

    # --------------------------------------------------------
    # Weighted late fusion
    # --------------------------------------------------------

    video_contribution = (
        VIDEO_WEIGHT
        * video_score
    )

    audio_contribution = (
        AUDIO_WEIGHT
        * audio_score
    )

    fusion_score = (
        video_contribution
        +
        audio_contribution
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    if fusion_score >= FUSION_THRESHOLD:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    result = {

        "prediction":
            prediction,

        "fraud_risk":
            float(fusion_score),

        "fraud_risk_percent":
            float(fusion_score * 100),

        "video_weight":
            VIDEO_WEIGHT,

        "audio_weight":
            AUDIO_WEIGHT,

        "video_contribution":
            float(video_contribution),

        "audio_contribution":
            float(audio_contribution),

        "threshold":
            FUSION_THRESHOLD
    }

    print()
    print("=" * 60)
    print("MULTIMODAL FUSION")
    print("=" * 60)

    print(
        "Video contribution:",
        f"{video_contribution * 100:.2f}%"
    )

    print(
        "Audio contribution:",
        f"{audio_contribution * 100:.2f}%"
    )

    print(
        "Final fraud risk:",
        f"{fusion_score * 100:.2f}%"
    )

    print(
        "Final prediction:",
        prediction
    )

    print("=" * 60)

    return result


# ============================================================
# COMPLETE TWO-MODALITY PIPELINE
# ============================================================

def analyze_video(
    video_path,
    work_dir
):

    video_path = Path(
        video_path
    )

    work_dir = Path(
        work_dir
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("STARTING MULTIMODAL VIDEO ANALYSIS")
    print("=" * 60)

    print(
        "Input video:",
        video_path
    )

    # ========================================================
    # 1. VIDEO MODALITY
    # ========================================================

    print()
    print("=" * 60)
    print("1. VIDEO MODALITY")
    print("=" * 60)

    video_result = predict_video(
        video_path
    )

    # ========================================================
    # 2. AUDIO MODALITY
    # ========================================================

    print()
    print("=" * 60)
    print("2. AUDIO MODALITY")
    print("=" * 60)

    audio_path = (
        work_dir /
        "audio.wav"
    )

    extract_audio(
        video_path,
        audio_path
    )

    audio_result = predict_audio(
        audio_path
    )

    # ========================================================
    # 3. MULTIMODAL FUSION
    # ========================================================

    fusion_result = fuse_predictions(
        video_result,
        audio_result
    )

    # ========================================================
    # 4. FINAL RESULT
    # ========================================================

    final_result = {

        "video":
            video_result,

        "audio":
            audio_result,

        "fusion":
            fusion_result
    }

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "Final prediction:",
        fusion_result["prediction"]
    )

    print(
        "Fraud risk:",
        f"{fusion_result['fraud_risk_percent']:.2f}%"
    )

    print("=" * 60)

    return final_result

