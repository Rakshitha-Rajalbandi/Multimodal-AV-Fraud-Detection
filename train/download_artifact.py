"""
ArtiFact Dataset Downloader
Downloads the ArtiFact dataset from Kaggle:
  Dataset: awsaf49/artifact-dataset
  Structure after download:
    data/artifact/
      real/           -> label 1 (REAL)
      midjourney/     -> label 0 (FAKE)
      dalle/          -> label 0 (FAKE)
      stablediffusion/ -> label 0 (FAKE)
      (other AI generator subdirs) -> label 0 (FAKE)

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
DATA_DIR = PROJECT_ROOT / "data" / "artifact"

DATASET_SLUG = "awsaf49/artifact-dataset"


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
    """Print file counts per class directory and overall summary."""
    print("\nArtiFact dataset file counts:")
    if not DATA_DIR.exists():
        print(f"  WARNING: data directory not found: {DATA_DIR}")
        return

    real_count = 0
    fake_count = 0
    extensions = ("jpg", "jpeg", "png", "webp")

    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        count = sum(
            len(list(subdir.rglob(f"*.{ext}")))
            for ext in extensions
        )
        if subdir.name.lower() == "real":
            real_count += count
            print(f"  {subdir.name}/  (REAL): {count} images")
        else:
            fake_count += count
            print(f"  {subdir.name}/  (FAKE): {count} images")

    print(f"\nSummary:")
    print(f"  Total REAL : {real_count}")
    print(f"  Total FAKE : {fake_count}")
    print(f"  Grand total: {real_count + fake_count}")


def download_artifact():
    """Download and unzip ArtiFact from Kaggle into DATA_DIR."""
    check_kaggle_cli()
    check_kaggle_credentials()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ArtiFact dataset from Kaggle...")
    print(f"  Dataset : {DATASET_SLUG}")
    print(f"  Target  : {DATA_DIR}")
    print(f"  Note    : ArtiFact is ~2.5M images; this may take a while.")

    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(DATA_DIR),
            "--unzip",
        ],
        stdout=None,
        stderr=None,
    )

    if result.returncode != 0:
        print(f"\nERROR: Kaggle download failed (exit code {result.returncode}).")
        sys.exit(result.returncode)

    print("\nVerifying downloaded structure...")
    real_dir = DATA_DIR / "real"
    if not real_dir.exists():
        # Search one level deeper in case of nested extraction
        candidates = list(DATA_DIR.rglob("real"))
        if candidates:
            print(f"  Found real/ at: {candidates[0]}")
            print(f"  NOTE: Data may be nested. Expected: {real_dir}")
        else:
            print(f"  WARNING: Expected 'real/' folder not found under {DATA_DIR}")
            print(f"  Check the contents of: {DATA_DIR}")

    print_counts()
    print(f"\nArtiFact download complete.")
    print(f"Root: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the ArtiFact dataset from Kaggle."
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
        download_artifact()


if __name__ == "__main__":
    main()
