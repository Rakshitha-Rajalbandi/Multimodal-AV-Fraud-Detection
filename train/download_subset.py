# import os
# import subprocess
# import pandas as pd

# DATASET = "pranay22077/dfdc-10"

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SPLITS = {
#     "train": os.path.join(PROJECT_ROOT, "data", "subset", "train.csv"),
#     "val": os.path.join(PROJECT_ROOT, "data", "subset", "val.csv"),
#     "test": os.path.join(PROJECT_ROOT, "data", "subset", "test.csv"),
# }

# RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


# def download_video(kaggle_path, label, split):
#     output_dir = os.path.join(RAW_DIR, split, label)
#     os.makedirs(output_dir, exist_ok=True)

#     filename = os.path.basename(kaggle_path)
#     output_file = os.path.join(output_dir, filename)

#     # Skip if already downloaded
#     if os.path.exists(output_file):
#         print(f"[SKIP] {filename}")
#         return True

#     print(f"[DOWNLOAD] {split}/{label}/{filename}")

#     command = [
#         "kaggle",
#         "datasets",
#         "download",
#         "-d",
#         DATASET,
#         "-f",
#         kaggle_path,
#         "-p",
#         output_dir,
#     ]

#     result = subprocess.run(command)

#     if result.returncode != 0:
#         print(f"[FAILED] {filename}")
#         return False

#     # Kaggle may create a ZIP file when downloading a single file.
#     zip_file = os.path.join(output_dir, filename + ".zip")

#     if os.path.exists(zip_file):
#         import zipfile

#         with zipfile.ZipFile(zip_file, "r") as z:
#             z.extractall(output_dir)

#         os.remove(zip_file)

#     return os.path.exists(output_file)


# def main():

#     print("=" * 60)
#     print("DFDC SUBSET DOWNLOADER")
#     print("=" * 60)

#     total = 0
#     success = 0
#     failed = 0

#     for split, csv_path in SPLITS.items():

#         df = pd.read_csv(csv_path)

#         print(f"\n{split.upper()}: {len(df)} videos")

#         for _, row in df.iterrows():

#             total += 1

#             if download_video(
#                 row["kaggle_path"],
#                 row["label"],
#                 split
#             ):
#                 success += 1
#             else:
#                 failed += 1

#     print("\n" + "=" * 60)
#     print("DOWNLOAD COMPLETE")
#     print("=" * 60)

#     print(f"Total   : {total}")
#     print(f"Success : {success}")
#     print(f"Failed  : {failed}")


# if __name__ == "__main__":
#     main()



import os
import subprocess
import pandas as pd
import time
import zipfile

DATASET = "pranay22077/dfdc-10"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLITS = {
    "train": os.path.join(PROJECT_ROOT, "data", "subset", "train.csv"),
    "val": os.path.join(PROJECT_ROOT, "data", "subset", "val.csv"),
    "test": os.path.join(PROJECT_ROOT, "data", "subset", "test.csv"),
}

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

MAX_RETRIES = 3
RETRY_DELAY = 5


def download_video(kaggle_path, label, split):

    output_dir = os.path.join(RAW_DIR, split, label)
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(kaggle_path)
    output_file = os.path.join(output_dir, filename)

    # ---------------------------------------------------------
    # 1. Skip already downloaded videos
    # ---------------------------------------------------------
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"[SKIP] {split}/{label}/{filename}")
        return True

    # ---------------------------------------------------------
    # 2. Try downloading multiple times
    # ---------------------------------------------------------
    for attempt in range(1, MAX_RETRIES + 1):

        print(
            f"[DOWNLOAD] {split}/{label}/{filename} "
            f"(attempt {attempt}/{MAX_RETRIES})"
        )

        command = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-f",
            kaggle_path,
            "-p",
            output_dir,
        ]

        try:
            result = subprocess.run(command)

        except KeyboardInterrupt:
            print("\n[STOPPED] Download interrupted by user.")
            raise

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            result = None

        # -----------------------------------------------------
        # 3. Check whether Kaggle succeeded
        # -----------------------------------------------------
        if result is not None and result.returncode == 0:

            zip_file = os.path.join(
                output_dir,
                filename + ".zip"
            )

            # Kaggle may return a ZIP file
            if os.path.exists(zip_file):

                try:
                    with zipfile.ZipFile(zip_file, "r") as z:
                        z.extractall(output_dir)

                    os.remove(zip_file)

                except Exception as e:
                    print(f"[ERROR] Extracting {filename}: {e}")
                    return False

            # Verify final file
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"[SUCCESS] {filename}")
                return True

        # -----------------------------------------------------
        # 4. Download failed
        # -----------------------------------------------------
        print(f"[RETRY] {filename} failed.")

        if attempt < MAX_RETRIES:
            print(f"Waiting {RETRY_DELAY} seconds before retry...")
            time.sleep(RETRY_DELAY)

    print(f"[FAILED] {filename} after {MAX_RETRIES} attempts")
    return False


def main():

    print("=" * 60)
    print("DFDC SUBSET DOWNLOADER")
    print("=" * 60)

    total = 0
    success = 0
    failed = 0

    for split, csv_path in SPLITS.items():

        df = pd.read_csv(csv_path)

        print(f"\n{split.upper()}: {len(df)} videos")

        for _, row in df.iterrows():

            total += 1

            result = download_video(
                row["kaggle_path"],
                row["label"],
                split
            )

            if result:
                success += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"Total   : {total}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")


if __name__ == "__main__":
    main()

