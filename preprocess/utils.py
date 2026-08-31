"""
Utility helpers for preprocessing: ffmpeg wrapper and video listing.
"""
import subprocess
from pathlib import Path
import shutil

def check_ffmpeg():
    """Ensure ffmpeg is available on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")

def run_ffmpeg_extract_audio(video_path, out_wav, sr=16000):
    """
    Extract single-channel audio at sr using ffmpeg.
    Overwrites out_wav if exists.
    """
    check_ffmpeg()
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), str(out_wav)
    ]
    # run and raise on failure
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")

def list_videos(input_dir, exts=(".mp4", ".mov", ".mkv", ".avi")):
    """Yield Path objects for video files in input_dir (recursively)."""
    p = Path(input_dir)
    for ext in exts:
        for f in p.rglob(f"*{ext}"):
            yield f