# import pandas as pd
# import random
# from pathlib import Path

# # ============================================================
# # CONFIGURATION
# # ============================================================

# METADATA = Path(r".\metadata\extracted\metadata.csv")
# MATCHED = Path(r".\matched_5197.csv")

# RANDOM_SEED = 42
# VIDEOS_PER_CLASS = 300

# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15


# # ============================================================
# # LOAD DATA
# # ============================================================

# print("=" * 60)
# print("CREATING LEAKAGE-SAFE SUBSET")
# print("=" * 60)

# metadata = pd.read_csv(METADATA)
# matched = pd.read_csv(MATCHED)

# print(f"Metadata rows       : {len(metadata)}")
# print(f"Matched videos      : {len(matched)}")


# # ============================================================
# # CREATE SOURCE ID
# # ============================================================

# # metadata["source_id"] = metadata["original"].fillna(
# #     metadata["filename"]
# # )



# # Treat both actual NaN values and the string "NAN"
# # as missing originals.

# metadata["original"] = metadata["original"].replace(
#     ["NAN", "nan", "NaN", ""],
#     pd.NA
# )

# metadata["source_id"] = metadata["original"].fillna(
#     metadata["filename"]
# )

# # ============================================================
# # KEEP ONLY REQUIRED METADATA COLUMNS
# # ============================================================

# metadata_small = metadata[
#     [
#         "filename",
#         "label",
#         "audio_label",
#         "video_label",
#         "original",
#         "source_id"
#     ]
# ].copy()


# # ============================================================
# # REMOVE DUPLICATE COLUMNS BEFORE MERGING
# # ============================================================

# # matched_5197.csv already has label/audio/video/original/source_id
# # so we only need filename from it for matching.

# matched_small = matched[
#     [
#         "kaggle_path",
#         "filename"
#     ]
# ].copy()


# # ============================================================
# # MERGE
# # ============================================================

# df = matched_small.merge(
#     metadata_small,
#     on="filename",
#     how="inner"
# )

# print(f"After metadata match: {len(df)}")


# # ============================================================
# # CHECK LABELS
# # ============================================================

# print("\nOriginal label distribution:")
# print(df["label"].value_counts())


# # ============================================================
# # ONE VIDEO PER SOURCE
# # ============================================================

# # VERY IMPORTANT:
# #
# # A single original real video can have many fake versions.
# #
# # Example:
# #
# # original_video.mp4
# #       |
# #       +-- fake_1.mp4
# #       +-- fake_2.mp4
# #       +-- fake_3.mp4
# #       +-- ...
# #
# # We select only ONE video for each source.

# df = (
#     df
#     .groupby(
#         ["label", "source_id"],
#         group_keys=False
#     )
#     .sample(
#         n=1,
#         random_state=RANDOM_SEED
#     )
#     .reset_index(drop=True)
# )


# print("\nAfter selecting ONE video per source:")

# print(df["label"].value_counts())

# print("\nUnique sources:")

# print(
#     df.groupby("label")["source_id"].nunique()
# )


# # ============================================================
# # BALANCE REAL AND FAKE
# # ============================================================

# real = df[df["label"] == "REAL"].copy()
# fake = df[df["label"] == "FAKE"].copy()

# print("\nAvailable after source filtering:")
# print(f"REAL: {len(real)}")
# print(f"FAKE: {len(fake)}")


# if len(real) < VIDEOS_PER_CLASS:
#     raise ValueError(
#         f"Not enough REAL videos. "
#         f"Available: {len(real)}, "
#         f"Required: {VIDEOS_PER_CLASS}"
#     )

# if len(fake) < VIDEOS_PER_CLASS:
#     raise ValueError(
#         f"Not enough FAKE videos. "
#         f"Available: {len(fake)}, "
#         f"Required: {VIDEOS_PER_CLASS}"
#     )


# real = real.sample(
#     n=VIDEOS_PER_CLASS,
#     random_state=RANDOM_SEED
# )

# fake = fake.sample(
#     n=VIDEOS_PER_CLASS,
#     random_state=RANDOM_SEED
# )


# balanced = pd.concat(
#     [real, fake],
#     ignore_index=True
# )


# # Shuffle
# balanced = balanced.sample(
#     frac=1,
#     random_state=RANDOM_SEED
# ).reset_index(drop=True)


# # ============================================================
# # SPLIT REAL AND FAKE SEPARATELY
# # ============================================================

# train_parts = []
# val_parts = []
# test_parts = []


# for label in ["REAL", "FAKE"]:

#     class_df = balanced[
#         balanced["label"] == label
#     ].copy()

#     class_df = class_df.sample(
#         frac=1,
#         random_state=RANDOM_SEED
#     ).reset_index(drop=True)

#     n = len(class_df)

#     train_end = int(n * TRAIN_RATIO)

#     val_end = train_end + int(
#         n * VAL_RATIO
#     )

#     train_parts.append(
#         class_df.iloc[:train_end]
#     )

#     val_parts.append(
#         class_df.iloc[train_end:val_end]
#     )

#     test_parts.append(
#         class_df.iloc[val_end:]
#     )


# # ============================================================
# # COMBINE SPLITS
# # ============================================================

# train = pd.concat(
#     train_parts,
#     ignore_index=True
# )

# val = pd.concat(
#     val_parts,
#     ignore_index=True
# )

# test = pd.concat(
#     test_parts,
#     ignore_index=True
# )


# # Shuffle each split

# train = train.sample(
#     frac=1,
#     random_state=RANDOM_SEED
# ).reset_index(drop=True)

# val = val.sample(
#     frac=1,
#     random_state=RANDOM_SEED
# ).reset_index(drop=True)

# test = test.sample(
#     frac=1,
#     random_state=RANDOM_SEED
# ).reset_index(drop=True)


# # ============================================================
# # SOURCE LEAKAGE CHECK
# # ============================================================

# train_sources = set(train["source_id"])
# val_sources = set(val["source_id"])
# test_sources = set(test["source_id"])


# train_val_overlap = train_sources & val_sources
# train_test_overlap = train_sources & test_sources
# val_test_overlap = val_sources & test_sources


# print("\nSOURCE LEAKAGE CHECK")
# print("--------------------")

# print(
#     "Train ∩ Val :",
#     len(train_val_overlap)
# )

# print(
#     "Train ∩ Test:",
#     len(train_test_overlap)
# )

# print(
#     "Val ∩ Test  :",
#     len(val_test_overlap)
# )


# if train_val_overlap:
#     raise ValueError(
#         "SOURCE LEAKAGE DETECTED: Train and Val overlap!"
#     )

# if train_test_overlap:
#     raise ValueError(
#         "SOURCE LEAKAGE DETECTED: Train and Test overlap!"
#     )

# if val_test_overlap:
#     raise ValueError(
#         "SOURCE LEAKAGE DETECTED: Val and Test overlap!"
#     )


# # ============================================================
# # SAVE DATASETS
# # ============================================================

# output_dir = Path(
#     "./data/subset"
# )

# output_dir.mkdir(
#     parents=True,
#     exist_ok=True
# )


# columns = [
#     "kaggle_path",
#     "filename",
#     "label",
#     "audio_label",
#     "video_label",
#     "original",
#     "source_id"
# ]


# train[columns].to_csv(
#     output_dir / "train.csv",
#     index=False
# )

# val[columns].to_csv(
#     output_dir / "val.csv",
#     index=False
# )

# test[columns].to_csv(
#     output_dir / "test.csv",
#     index=False
# )


# # ============================================================
# # FINAL SUMMARY
# # ============================================================

# print("\n" + "=" * 60)
# print("FINAL DATASET")
# print("=" * 60)


# print("\nTRAIN:")
# print(
#     train["label"].value_counts()
# )


# print("\nVALIDATION:")
# print(
#     val["label"].value_counts()
# )


# print("\nTEST:")
# print(
#     test["label"].value_counts()
# )


# print("\nTOTAL:")
# print(
#     pd.concat(
#         [train, val, test]
#     )["label"].value_counts()
# )


# print("\nTotal videos:", len(
#     pd.concat(
#         [train, val, test]
#     )
# ))


# print("\nSaved files:")

# print(
#     output_dir / "train.csv"
# )

# print(
#     output_dir / "val.csv"
# )

# print(
#     output_dir / "test.csv"
# )

# print("\nSUCCESS!")




import pandas as pd
import random
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

MATCHED_FILE = Path("matched_5197.csv")

OUTPUT_DIR = Path("data/subset")

RANDOM_SEED = 42

# Total target:
# 200 REAL + 200 FAKE = 400 videos
VIDEOS_PER_CLASS = 200

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING SOURCE-LEVEL LEAKAGE-SAFE SUBSET")
print("=" * 60)

df = pd.read_csv(MATCHED_FILE)

print(f"Total matched videos: {len(df)}")

print("\nOriginal labels:")
print(df["label"].value_counts())


# ============================================================
# IMPORTANT:
# source_id already exists in matched_5197.csv
# ============================================================

required_columns = [
    "kaggle_path",
    "filename",
    "label",
    "audio_label",
    "video_label",
    "original",
    "source_id"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CREATE SOURCE GROUPS
# ============================================================

# Every source_id must belong to exactly ONE split.
#
# This prevents:
#
#       same source → TRAIN
#       same source → TEST
#
# which would cause data leakage.

source_groups = (
    df.groupby("source_id")
    .agg(
        num_videos=("filename", "count"),
        num_real=("label", lambda x: (x == "REAL").sum()),
        num_fake=("label", lambda x: (x == "FAKE").sum())
    )
    .reset_index()
)

print("\nSource statistics:")
print(
    f"Total unique sources: {len(source_groups)}"
)

print(
    f"Sources containing REAL videos: "
    f"{(source_groups.num_real > 0).sum()}"
)

print(
    f"Sources containing FAKE videos: "
    f"{(source_groups.num_fake > 0).sum()}"
)


# ============================================================
# SHUFFLE SOURCES
# ============================================================

random.seed(RANDOM_SEED)

sources = source_groups["source_id"].tolist()

random.shuffle(sources)


# ============================================================
# SOURCE-LEVEL SPLIT
# ============================================================

# We split SOURCES, not individual videos.

n_sources = len(sources)

train_end = int(
    n_sources * TRAIN_RATIO
)

val_end = train_end + int(
    n_sources * VAL_RATIO
)

train_sources = set(
    sources[:train_end]
)

val_sources = set(
    sources[train_end:val_end]
)

test_sources = set(
    sources[val_end:]
)


print("\nSource-level split:")
print(
    f"Train sources: {len(train_sources)}"
)

print(
    f"Validation sources: {len(val_sources)}"
)

print(
    f"Test sources: {len(test_sources)}"
)


# ============================================================
# VERIFY SOURCE LEAKAGE
# ============================================================

print("\nSOURCE LEAKAGE CHECK")
print("--------------------")

print(
    "Train ∩ Val:",
    len(train_sources & val_sources)
)

print(
    "Train ∩ Test:",
    len(train_sources & test_sources)
)

print(
    "Val ∩ Test:",
    len(val_sources & test_sources)
)


if train_sources & val_sources:
    raise ValueError(
        "SOURCE LEAKAGE BETWEEN TRAIN AND VAL!"
    )

if train_sources & test_sources:
    raise ValueError(
        "SOURCE LEAKAGE BETWEEN TRAIN AND TEST!"
    )

if val_sources & test_sources:
    raise ValueError(
        "SOURCE LEAKAGE BETWEEN VAL AND TEST!"
    )


# ============================================================
# GET VIDEOS FOR EACH SPLIT
# ============================================================

train_df = df[
    df["source_id"].isin(train_sources)
].copy()

val_df = df[
    df["source_id"].isin(val_sources)
].copy()

test_df = df[
    df["source_id"].isin(test_sources)
].copy()


# ============================================================
# FUNCTION TO SELECT BALANCED VIDEOS
# ============================================================

def select_balanced(data, videos_per_class, seed):

    real = data[
        data["label"] == "REAL"
    ].copy()

    fake = data[
        data["label"] == "FAKE"
    ].copy()

    print("\nAvailable in split:")
    print("REAL:", len(real))
    print("FAKE:", len(fake))

    # We need enough videos from both classes.
    if len(real) < videos_per_class:
        return None

    if len(fake) < videos_per_class:
        return None

    real = real.sample(
        n=videos_per_class,
        random_state=seed
    )

    fake = fake.sample(
        n=videos_per_class,
        random_state=seed
    )

    result = pd.concat(
        [real, fake],
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)


# ============================================================
# IMPORTANT:
# Try to find a balanced dataset from the source split.
# ============================================================

# Instead of forcing 200/class immediately, first inspect
# how many REAL videos are available in each split.

print("\n" + "=" * 60)
print("VIDEOS AVAILABLE PER SPLIT")
print("=" * 60)

for name, split in [
    ("TRAIN", train_df),
    ("VAL", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    print(
        split["label"].value_counts()
    )

    print(
        "Unique sources:",
        split["source_id"].nunique()
    )


# ============================================================
# TARGETS BASED ON AVAILABLE DATA
# ============================================================

# We need balanced train/val/test.
#
# Desired:
#
# TRAIN = 140 REAL + 140 FAKE
# VAL   = 30 REAL + 30 FAKE
# TEST  = 30 REAL + 30 FAKE
#
# Total = 200 REAL + 200 FAKE

TRAIN_PER_CLASS = 140
VAL_PER_CLASS = 30
TEST_PER_CLASS = 30


# ============================================================
# SELECT DATA
# ============================================================

train_selected = select_balanced(
    train_df,
    TRAIN_PER_CLASS,
    RANDOM_SEED
)

val_selected = select_balanced(
    val_df,
    VAL_PER_CLASS,
    RANDOM_SEED + 1
)

test_selected = select_balanced(
    test_df,
    TEST_PER_CLASS,
    RANDOM_SEED + 2
)


# ============================================================
# CHECK WHETHER TARGET IS POSSIBLE
# ============================================================

if train_selected is None:
    raise ValueError(
        "\nTRAIN does not contain enough REAL/FAKE videos "
        "for the requested 140 + 140."
    )

if val_selected is None:
    raise ValueError(
        "\nVALIDATION does not contain enough REAL/FAKE videos "
        "for the requested 30 + 30."
    )

if test_selected is None:
    raise ValueError(
        "\nTEST does not contain enough REAL/FAKE videos "
        "for the requested 30 + 30."
    )


# ============================================================
# FINAL SOURCE LEAKAGE CHECK
# ============================================================

train_sources_final = set(
    train_selected["source_id"]
)

val_sources_final = set(
    val_selected["source_id"]
)

test_sources_final = set(
    test_selected["source_id"]
)


print("\n" + "=" * 60)
print("FINAL SOURCE LEAKAGE CHECK")
print("=" * 60)

print(
    "Train ∩ Val:",
    len(train_sources_final & val_sources_final)
)

print(
    "Train ∩ Test:",
    len(train_sources_final & test_sources_final)
)

print(
    "Val ∩ Test:",
    len(val_sources_final & test_sources_final)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


columns = [
    "kaggle_path",
    "filename",
    "label",
    "audio_label",
    "video_label",
    "original",
    "source_id"
]


train_selected[columns].to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

val_selected[columns].to_csv(
    OUTPUT_DIR / "val.csv",
    index=False
)

test_selected[columns].to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("\nTRAIN")
print(
    train_selected["label"].value_counts()
)

print("\nVALIDATION")
print(
    val_selected["label"].value_counts()
)

print("\nTEST")
print(
    test_selected["label"].value_counts()
)

print("\nTOTAL VIDEOS:")
print(
    len(train_selected)
    + len(val_selected)
    + len(test_selected)
)

print("\nSaved:")

print(
    OUTPUT_DIR / "train.csv"
)

print(
    OUTPUT_DIR / "val.csv"
)

print(
    OUTPUT_DIR / "test.csv"
)

print("\nSUCCESS!")