# import argparse
# from pathlib import Path

# import librosa
# import numpy as np


# def make_melspectrogram(audio_path, output_path):
#     # Load audio at 16 kHz
#     audio, sample_rate = librosa.load(
#         audio_path,
#         sr=16000,
#         mono=True
#     )

#     # Create Mel-spectrogram
#     mel = librosa.feature.melspectrogram(
#         y=audio,
#         sr=sample_rate,
#         n_fft=1024,
#         hop_length=256,
#         n_mels=128
#     )

#     # Convert power to decibels
#     mel_db = librosa.power_to_db(
#         mel,
#         ref=np.max
#     )

#     # Save as NumPy compressed file
#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     np.savez_compressed(
#         output_path,
#         mel=mel_db
#     )

#     print(f"Saved: {output_path}")
#     print(f"Shape: {mel_db.shape}")


# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--input-audio-dir",
#         required=True
#     )

#     parser.add_argument(
#         "--out-dir",
#         required=True
#     )

#     args = parser.parse_args()

#     input_dir = Path(args.input_audio_dir)
#     output_dir = Path(args.out_dir)

#     audio_files = list(input_dir.glob("*.wav"))

#     if not audio_files:
#         print(f"No WAV files found in {input_dir}")
#         return

#     for audio_file in audio_files:

#         output_file = (
#             output_dir /
#             f"{audio_file.stem}.npz"
#         )

#         make_melspectrogram(
#             audio_file,
#             output_file
#         )


# if __name__ == "__main__":
#     main()




"""
Compute log-mel spectrograms for each audio file (saved as numpy .npz)
"""
import argparse
from pathlib import Path
import librosa
import numpy as np

def compute_melspec(in_wav, out_npz, sr=16000, n_mels=128, hop_length=256, n_fft=1024):
    y, _ = librosa.load(in_wav, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    np.savez_compressed(out_npz, mel=log_mel.astype(np.float32))

def main(args):
    input_dir = Path(args.input_audio_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted(input_dir.glob("*.wav")):
        vid_id = wav.stem
        out_npz = out_dir / f"{vid_id}.npz"
        try:
            compute_melspec(str(wav), str(out_npz), sr=args.sr, n_mels=args.n_mels, hop_length=args.hop_length, n_fft=args.n_fft)
            print("Made melspec", vid_id)
        except Exception as e:
            print("Failed melspec for", vid_id, e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-audio-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--n-fft", type=int, default=1024)
    args = p.parse_args()
    main(args)