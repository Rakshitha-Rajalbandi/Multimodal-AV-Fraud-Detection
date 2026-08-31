import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
METADATA = r".\metadata\extracted\metadata.csv"
VIDEO_PATHS = r".\all_video_paths.txt"

# -----------------------------
# Load metadata
# -----------------------------
df = pd.read_csv(METADATA)

# Normalize filename
df["filename"] = df["filename"].astype(str).str.strip()

# -----------------------------
# Load Kaggle video paths
# -----------------------------
with open(VIDEO_PATHS, "r", encoding="utf-8") as f:
    paths = [line.strip() for line in f if line.strip()]

video_df = pd.DataFrame({
    "kaggle_path": paths
})

# Extract filename from path
video_df["filename"] = video_df["kaggle_path"].apply(
    os.path.basename
)

# Remove duplicate filenames
video_df = video_df.drop_duplicates("filename")

# -----------------------------
# Match with metadata
# -----------------------------
matched = video_df.merge(
    df,
    on="filename",
    how="inner"
)

# -----------------------------
# Create source_id
# -----------------------------
matched["source_id"] = matched.apply(
    lambda row:
        row["original"]
        if pd.notna(row["original"]) and row["original"] != "NAN"
        else row["filename"],
    axis=1
)

# -----------------------------
# Print results
# -----------------------------
print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)

print(f"Paths collected       : {len(paths)}")
print(f"Unique video filenames: {len(video_df)}")
print(f"Matched videos        : {len(matched)}")

print("\n===== LABEL DISTRIBUTION =====")
print(matched["label"].value_counts())

print("\n===== UNIQUE SOURCE IDs =====")
print(
    matched.groupby("label")["source_id"]
    .nunique()
)

print("\n===== SOURCES WITH MULTIPLE VIDEOS =====")

source_counts = (
    matched.groupby(["label", "source_id"])
    .size()
    .reset_index(name="count")
)

print(
    source_counts
    .sort_values("count", ascending=False)
    .head(20)
    .to_string(index=False)
)

# -----------------------------
# Save analysis
# -----------------------------
matched.to_csv(
    "matched_5197.csv",
    index=False
)

print("\nSaved: matched_5197.csv")