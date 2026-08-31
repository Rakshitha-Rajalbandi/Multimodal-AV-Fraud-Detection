from pathlib import Path
import csv


def create_metadata(split):
    raw_dir = Path("data/raw") / split
    output_file = Path("data/metadata") / f"{split}.csv"

    rows = []

    for label_name, label in [("FAKE", 0), ("REAL", 1)]:
        label_dir = raw_dir / label_name

        for video_path in sorted(label_dir.glob("*.mp4")):
            video_id = video_path.stem

            rows.append({
                "video_id": video_id,
                "label": label,
                "label_name": label_name
            })

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_id", "label", "label_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{split}: {len(rows)} videos")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    create_metadata("train")
    create_metadata("test")