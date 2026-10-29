"""
GenImage Dataset Downloader
Downloads the GenImage dataset from Hugging Face:
  Repository: zhwang/GenImage
  URL: https://huggingface.co/datasets/zhwang/GenImage

Structure after download (data/genimage/):
  imagenet_glide/                   -> REAL  (label 1)
  imagenet_ai_0419_biggan/          -> FAKE  (label 0)
  imagenet_ai_0419_vqdm/            -> FAKE  (label 0)
  imagenet_ai_0508_adm/             -> FAKE  (label 0)
  imagenet_ai_0508_glide/           -> FAKE  (label 0)
  imagenet_ai_0419_midjourney/      -> FAKE  (label 0)
  imagenet_ai_0419_sdv4/            -> FAKE  (label 0)
  imagenet_ai_0419_sdv5/            -> FAKE  (label 0)
  imagenet_ai_0424_wukong/          -> FAKE  (label 0)

Prerequisites:
  pip install huggingface_hub
  Optionally: huggingface-cli login  (for gated datasets)
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "genimage"

HF_REPO_ID = "zhwang/GenImage"

# Subdirectory name that contains REAL images
REAL_FOLDER_PATTERN = "imagenet_glide"

# Known AI-generator subdirectory names (FAKE)
FAKE_FOLDERS = [
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0419_vqdm",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0508_glide",
    "imagenet_ai_0419_midjourney",
    "imagenet_ai_0419_sdv4",
    "imagenet_ai_0419_sdv5",
    "imagenet_ai_0424_wukong",
]


def check_huggingface_hub():
    """Check that huggingface_hub is installed; print install instructions if not."""
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("  Fix: pip install huggingface_hub")
        print("  Then optionally run: huggingface-cli login")
        print("  (Required if the dataset is gated / requires acceptance of terms.)")
        sys.exit(1)


def print_counts():
    """Print file counts per generator directory and overall summary."""
    print("\nGenImage dataset file counts:")
    if not DATA_DIR.exists():
        print(f"  WARNING: data directory not found: {DATA_DIR}")
        return

    extensions = (".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG")
    real_count = 0
    fake_count = 0

    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        count = sum(
            len(list(subdir.rglob(f"*{ext}")))
            for ext in extensions
        )
        if REAL_FOLDER_PATTERN in subdir.name:
            real_count += count
            print(f"  {subdir.name}/  (REAL): {count} images")
        else:
            fake_count += count
            print(f"  {subdir.name}/  (FAKE): {count} images")

    print(f"\nSummary:")
    print(f"  Total REAL : {real_count}")
    print(f"  Total FAKE : {fake_count}")
    print(f"  Grand total: {real_count + fake_count}")


def download_genimage():
    """Download GenImage from Hugging Face into DATA_DIR."""
    check_huggingface_hub()

    from huggingface_hub import snapshot_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GenImage dataset from Hugging Face...")
    print(f"  Repository: {HF_REPO_ID}")
    print(f"  Target    : {DATA_DIR}")
    print(f"  Note      : GenImage is ~1.3M images; this may take a long time.")
    print(f"  If download is interrupted, re-run — snapshot_download resumes.")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        ignore_patterns=["*.parquet", "*.arrow", "*.json"],
    )

    print(f"\nDownload complete.")
    print_counts()
    print(f"\nGenImage download complete.")
    print(f"Root: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the GenImage dataset from Hugging Face."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download and only print file counts for existing data.",
    )
    args = parser.parse_args()

    if args.skip_download:
        print(f"--skip-download flag set. Skipping download.")
        print_counts()
    else:
        download_genimage()


if __name__ == "__main__":
    main()
