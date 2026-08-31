# import numpy as np
# import librosa
# import pandas as pd
# from pathlib import Path


# # ============================================================
# # CONFIGURATION
# # ============================================================

# METADATA = Path("data/metadata/test.csv")
# AUDIO_DIR = Path("data/processed/test/audio")
# LIPS_DIR = Path("data/processed/test/lips")

# OUTPUT_DIR = Path("results/sync_evaluation")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SAMPLE_RATE = 16000


# # ============================================================
# # AUDIO ACTIVITY
# # ============================================================

# def calculate_audio_activity(audio_path, target_length):
#     """
#     Convert audio into a simple temporal energy/activity sequence.

#     The sequence is resized to match the number of lip frames.
#     """

#     audio, sr = librosa.load(
#         audio_path,
#         sr=SAMPLE_RATE,
#         mono=True
#     )

#     # Calculate short-time RMS energy
#     rms = librosa.feature.rms(
#         y=audio,
#         frame_length=1024,
#         hop_length=512
#     )[0]

#     # Normalize
#     rms_min = rms.min()
#     rms_max = rms.max()

#     if rms_max - rms_min > 1e-8:
#         rms = (rms - rms_min) / (rms_max - rms_min)
#     else:
#         rms = np.zeros_like(rms)

#     # Resize audio activity to number of lip frames
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

#     return activity


# # ============================================================
# # LIP MOTION
# # ============================================================

# def load_lip_motion(lip_path):

#     lip = np.load(lip_path).astype(np.float32)

#     # Replace invalid values
#     lip = np.nan_to_num(
#         lip,
#         nan=0.0,
#         posinf=0.0,
#         neginf=0.0
#     )

#     if len(lip) == 0:
#         return None

#     return lip


# # ============================================================
# # NORMALIZATION
# # ============================================================

# def normalize(x):

#     x = np.asarray(x, dtype=np.float32)

#     minimum = x.min()
#     maximum = x.max()

#     if maximum - minimum < 1e-8:
#         return np.zeros_like(x)

#     return (x - minimum) / (
#         maximum - minimum
#     )


# # ============================================================
# # CORRELATION
# # ============================================================

# def calculate_correlation(lip, audio):

#     lip = normalize(lip)
#     audio = normalize(audio)

#     if len(lip) != len(audio):

#         x_old = np.linspace(
#             0,
#             1,
#             len(audio)
#         )

#         x_new = np.linspace(
#             0,
#             1,
#             len(lip)
#         )

#         audio = np.interp(
#             x_new,
#             x_old,
#             audio
#         )

#     # Remove constant sequences
#     if (
#         np.std(lip) < 1e-8
#         or
#         np.std(audio) < 1e-8
#     ):
#         return 0.0

#     correlation = np.corrcoef(
#         lip,
#         audio
#     )[0, 1]

#     if np.isnan(correlation):
#         return 0.0

#     return float(correlation)


# # ============================================================
# # MAIN
# # ============================================================

# def main():

#     metadata = pd.read_csv(METADATA)

#     results = []

#     print("=" * 60)
#     print("AUDIO-VIDEO SYNCHRONIZATION ANALYSIS")
#     print("=" * 60)

#     for _, row in metadata.iterrows():

#         video_id = str(row["video_id"])

#         audio_path = (
#             AUDIO_DIR /
#             f"{video_id}.wav"
#         )

#         lip_path = (
#             LIPS_DIR /
#             f"{video_id}.npy"
#         )

#         if not audio_path.exists():
#             print(
#                 video_id,
#                 ": audio missing"
#             )
#             continue

#         if not lip_path.exists():
#             print(
#                 video_id,
#                 ": lip data missing"
#             )
#             continue

#         lip = load_lip_motion(
#             lip_path
#         )

#         if lip is None:
#             continue

#         if np.count_nonzero(lip) < 8:
#             print(
#                 video_id,
#                 ": insufficient lip data"
#             )
#             continue

#         try:

#             audio_activity = (
#                 calculate_audio_activity(
#                     audio_path,
#                     len(lip)
#                 )
#             )

#             correlation = (
#                 calculate_correlation(
#                     lip,
#                     audio_activity
#                 )
#             )

#             # Convert correlation from [-1,1]
#             # into a simple [0,1] synchronization score.
#             sync_score = (
#                 correlation + 1
#             ) / 2

#             label = str(
#                 row["label"]
#             ).upper()

#             results.append({
#                 "video_id": video_id,
#                 "label": label,
#                 "correlation": correlation,
#                 "sync_score": sync_score
#             })

#             print(
#                 f"{video_id}: "
#                 f"correlation={correlation:.3f}, "
#                 f"sync={sync_score:.3f}"
#             )

#         except Exception as e:

#             print(
#                 video_id,
#                 ": ERROR",
#                 e
#             )

#     # ========================================================
#     # SAVE RESULTS
#     # ========================================================

#     df = pd.DataFrame(results)

#     output_file = (
#         OUTPUT_DIR /
#         "sync_scores.csv"
#     )

#     df.to_csv(
#         output_file,
#         index=False
#     )

#     print()
#     print("=" * 60)
#     print("SYNCHRONIZATION ANALYSIS COMPLETED")
#     print("=" * 60)

#     print(
#         "Videos analyzed:",
#         len(df)
#     )

#     if len(df) > 0:

#         print(
#             "Average correlation:",
#             round(
#                 df["correlation"].mean(),
#                 4
#             )
#         )

#         print(
#             "Average sync score:",
#             round(
#                 df["sync_score"].mean(),
#                 4
#             )
#         )

#     print()
#     print(
#         "Saved:",
#         output_file
#     )


# if __name__ == "__main__":
#     main()




import numpy as np
import librosa
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

METADATA = Path("data/metadata/test.csv")
AUDIO_DIR = Path("data/processed/test/audio")
# LIPS_DIR = Path("data/processed/test/lips")
LIPS_DIR = Path("data/processed/test/lips_highres")

OUTPUT_DIR = Path("results/sync_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

# Number of frames used for each video
EXPECTED_FRAMES = 32

# Minimum number of valid lip measurements required
MIN_VALID_FRAMES = 8

# Maximum temporal shift allowed when finding synchronization
MAX_LAG = 5


# ============================================================
# AUDIO ACTIVITY
# ============================================================

def calculate_audio_activity(audio_path, target_length):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float32)

    # RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=1024,
        hop_length=512
    )[0]

    # Normalize audio activity
    rms_min = rms.min()
    rms_max = rms.max()

    if rms_max - rms_min > 1e-8:

        rms = (
            rms - rms_min
        ) / (
            rms_max - rms_min
        )

    else:

        rms = np.zeros_like(rms)

    # Resize audio activity to lip sequence length
    x_old = np.linspace(
        0,
        1,
        len(rms)
    )

    x_new = np.linspace(
        0,
        1,
        target_length
    )

    activity = np.interp(
        x_new,
        x_old,
        rms
    )

    return activity.astype(np.float32)


# ============================================================
# LIP MOTION
# ============================================================

def load_lip_motion(lip_path):

    lip = np.load(lip_path).astype(np.float32)

    if len(lip) == 0:
        return None

    # Invalid values become NaN
    lip = np.where(
        np.isfinite(lip),
        lip,
        np.nan
    )

    return lip


# ============================================================
# INTERPOLATE MISSING LIP VALUES
# ============================================================

def fill_missing_lip_values(lip):

    lip = np.asarray(
        lip,
        dtype=np.float32
    )

    valid = np.isfinite(lip) & (lip > 0)

    valid_indices = np.where(valid)[0]

    if len(valid_indices) < MIN_VALID_FRAMES:
        return None

    # If all values are valid
    if len(valid_indices) == len(lip):
        return lip

    # Interpolate missing values
    all_indices = np.arange(len(lip))

    filled = np.interp(
        all_indices,
        valid_indices,
        lip[valid_indices]
    )

    return filled.astype(np.float32)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    minimum = np.min(x)
    maximum = np.max(x)

    if maximum - minimum < 1e-8:
        return np.zeros_like(x)

    return (
        x - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# CORRELATION WITH TEMPORAL LAG
# ============================================================

def calculate_lagged_correlation(
    lip,
    audio,
    max_lag=MAX_LAG
):

    lip = normalize(lip)
    audio = normalize(audio)

    best_correlation = -1.0
    best_lag = 0

    max_lag = min(
        max_lag,
        len(lip) - 2
    )

    for lag in range(
        -max_lag,
        max_lag + 1
    ):

        if lag > 0:

            lip_part = lip[lag:]
            audio_part = audio[:-lag]

        elif lag < 0:

            lip_part = lip[:lag]
            audio_part = audio[-lag:]

        else:

            lip_part = lip
            audio_part = audio

        if len(lip_part) < 3:
            continue

        if (
            np.std(lip_part) < 1e-8
            or
            np.std(audio_part) < 1e-8
        ):
            continue

        correlation = np.corrcoef(
            lip_part,
            audio_part
        )[0, 1]

        if np.isnan(correlation):
            continue

        if correlation > best_correlation:

            best_correlation = float(
                correlation
            )

            best_lag = lag

    if best_correlation == -1.0:
        return 0.0, 0

    return best_correlation, best_lag


# ============================================================
# MAIN
# ============================================================

def main():

    metadata = pd.read_csv(
        METADATA
    )

    results = []

    print("=" * 60)
    print("AUDIO-VIDEO SYNCHRONIZATION ANALYSIS")
    print("=" * 60)

    for _, row in metadata.iterrows():

        video_id = str(
            row["video_id"]
        )

        audio_path = (
            AUDIO_DIR /
            f"{video_id}.wav"
        )

        lip_path = (
            LIPS_DIR /
            f"{video_id}.npy"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not audio_path.exists():

            print(
                video_id,
                ": audio missing"
            )

            continue

        if not lip_path.exists():

            print(
                video_id,
                ": lip data missing"
            )

            continue

        # ----------------------------------------------------
        # Load lip data
        # ----------------------------------------------------

        lip = load_lip_motion(
            lip_path
        )

        if lip is None:

            print(
                video_id,
                ": empty lip data"
            )

            continue

        # ----------------------------------------------------
        # Count valid lip frames
        # ----------------------------------------------------

        valid_mask = (
            np.isfinite(lip)
            &
            (lip > 0)
        )

        valid_count = int(
            np.sum(valid_mask)
        )

        # ----------------------------------------------------
        # Minimum data requirement
        # ----------------------------------------------------

        if valid_count < MIN_VALID_FRAMES:

            print(
                f"{video_id}: "
                f"insufficient lip data "
                f"({valid_count}/{len(lip)})"
            )

            continue

        # ----------------------------------------------------
        # Fill missing values
        # ----------------------------------------------------

        lip_filled = (
            fill_missing_lip_values(
                lip
            )
        )

        if lip_filled is None:

            print(
                video_id,
                ": unable to interpolate lip data"
            )

            continue

        try:

            # ------------------------------------------------
            # Audio activity
            # ------------------------------------------------

            audio_activity = (
                calculate_audio_activity(
                    audio_path,
                    len(lip_filled)
                )
            )

            # ------------------------------------------------
            # Lagged correlation
            # ------------------------------------------------

            correlation, best_lag = (
                calculate_lagged_correlation(
                    lip_filled,
                    audio_activity
                )
            )

            # ------------------------------------------------
            # Convert correlation [-1,1]
            # into synchronization score [0,1]
            # ------------------------------------------------

            sync_score = (
                correlation + 1.0
            ) / 2.0

            label = str(
                row["label"]
            ).upper()

            results.append({

                "video_id":
                    video_id,

                "label":
                    label,

                "valid_lip_frames":
                    valid_count,

                "total_lip_frames":
                    len(lip),

                "correlation":
                    correlation,

                "best_lag":
                    best_lag,

                "sync_score":
                    sync_score
            })

            print(
                f"{video_id}: "
                f"valid={valid_count}/{len(lip)}, "
                f"correlation={correlation:.3f}, "
                f"lag={best_lag}, "
                f"sync={sync_score:.3f}"
            )

        except Exception as e:

            print(
                video_id,
                ": ERROR",
                e
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    df = pd.DataFrame(
        results
    )

    output_file = (
        OUTPUT_DIR /
        "sync_scores.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print(
        "SYNCHRONIZATION ANALYSIS COMPLETED"
    )
    print("=" * 60)

    print(
        "Videos analyzed:",
        len(df)
    )

    if len(df) > 0:

        print(
            "Average correlation:",
            round(
                df["correlation"].mean(),
                4
            )
        )

        print(
            "Average sync score:",
            round(
                df["sync_score"].mean(),
                4
            )
        )

        print(
            "Average valid lip frames:",
            round(
                df["valid_lip_frames"].mean(),
                2
            )
        )

    print()
    print(
        "Saved:",
        output_file
    )


if __name__ == "__main__":

    main()

