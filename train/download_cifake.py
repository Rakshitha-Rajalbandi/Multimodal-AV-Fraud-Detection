"""
CIFAKE Dataset Downloader
Downloads the CIFAKE dataset from Kaggle:
  Dataset: birdy654/cifake-real-and-ai-generated-synthetic-images
  Classes: REAL (label=1) and FAKE (label=0)
  Structure after download:
    data/cifake/
      train/
        REAL/   (60,000 images)
        FAKE/   (60,000 images)
      test/
        REAL/   (10,000 images)
        FAKE/   (10,000 images)

Prerequisites:
  pip install kaggle
  Set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
  OR place kaggle.json at ~/.kaggle/kaggle.json
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cifake"

DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def check_kaggle_cli():
    """Check that the kaggle CLI is installed and on PATH."""
    result = subprocess.run(
        ["kaggle", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("ERROR: kaggle CLI not found.")
        print("  Fix: pip install kaggle")
        print("  Then set up ~/.kaggle/kaggle.json with your API key.")
        print("  Get your key at: https://www.kaggle.com/settings/account")
        sys.exit(1)


def check_kaggle_credentials():
    """Verify Kaggle credentials exist (env vars OR kaggle.json)."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not kaggle_json.exists() and not (username and key):
        print("ERROR: Kaggle credentials not found.")
        print("  Option 1: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("  Option 2: Place kaggle.json at ~/.kaggle/kaggle.json")
        print("  Get your API key at: https://www.kaggle.com/settings/account")
        sys.exit(1)


def print_counts():
    """Print file counts per split/class directory."""
    print("\nCIFAKE dataset file counts:")
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            folder = DATA_DIR / split / cls
            if folder.exists():
                # Count common image extensions
                count = sum(
                    len(list(folder.glob(f"*.{ext}")))
                    for ext in ("jpg", "jpeg", "png")
                )
                print(f"  {split}/{cls}: {count} images")
            else:
                print(f"  WARNING: {split}/{cls} folder not found at {folder}")


def download_cifake():
    """Download and unzip CIFAKE from Kaggle into DATA_DIR."""
    check_kaggle_cli()
    check_kaggle_credentials()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAKE dataset from Kaggle...")
    print(f"  Dataset : {DATASET_SLUG}")
    print(f"  Target  : {DATA_DIR}")

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        # Stream output directly to terminal so the user sees progress
        stdout=None,
        stderr=None,
    )

    if result.returncode != 0:
        print(f"\nERROR: Kaggle download failed (exit code {result.returncode}).")
        sys.exit(result.returncode)

    print("\nVerifying downloaded structure...")
    missing = False
    for split in ["train", "test"]:
        for cls in ["REAL", "FAKE"]:
            folder = DATA_DIR / split / cls
            if not folder.exists():
                print(f"  WARNING: Expected folder not found: {folder}")
                missing = True

    if missing:
        print(
            "\nWARNING: Some expected folders are missing.\n"
            "The Kaggle dataset may have a nested zip or a different directory layout.\n"
            "Check the contents of: " + str(DATA_DIR)
        )

    print_counts()
    print(f"\nCIFAKE download complete.")
    print(f"Root: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the CIFAKE dataset from Kaggle."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Kaggle download and only print file counts for existing data.",
    )
    args = parser.parse_args()

    if args.skip_download:
        print(f"--skip-download flag set. Skipping download.")
        print_counts()
    else:
        download_cifake()


if __name__ == "__main__":
    main()
