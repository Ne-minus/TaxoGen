import pandas as pd
from pathlib import Path

# ===== paths =====
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"

IMAGES_ROOT = Path("/workspace/data")
OUT_TXT = "missing_images_report.txt"

EXTENSIONS = [".png", ".jpg", ".jpeg"]


def find_existing_image(images_root: Path, model_name: str, wordnet_id: str):
    model_dir = images_root / str(model_name)
    candidates = [model_dir / f"{wordnet_id}{ext}" for ext in EXTENSIONS]

    for path in candidates:
        if path.exists():
            return path

    return None


def scan_split(csv_path: str, split_name: str, images_root: Path):
    df = pd.read_csv(csv_path)

    missing_records = []

    for idx, row in df.iterrows():
        wordnet_id = str(row["wordnet_id"])
        model_a = str(row["model_a"])
        model_b = str(row["model_b"])

        image_a = find_existing_image(images_root, model_a, wordnet_id)
        image_b = find_existing_image(images_root, model_b, wordnet_id)

        if image_a is None:
            missing_records.append({
                "split": split_name,
                "row_idx": idx,
                "side": "A",
                "wordnet_id": wordnet_id,
                "model": model_a,
                "expected_dir": str(images_root / model_a),
            })

        if image_b is None:
            missing_records.append({
                "split": split_name,
                "row_idx": idx,
                "side": "B",
                "wordnet_id": wordnet_id,
                "model": model_b,
                "expected_dir": str(images_root / model_b),
            })

    return missing_records


all_missing = []
all_missing.extend(scan_split(TRAIN_CSV, "train", IMAGES_ROOT))
all_missing.extend(scan_split(VAL_CSV, "val", IMAGES_ROOT))
all_missing.extend(scan_split(TEST_CSV, "test", IMAGES_ROOT))

with open(OUT_TXT, "w", encoding="utf-8") as f:
    if not all_missing:
        f.write("No missing images found.\n")
    else:
        f.write(f"Total missing entries: {len(all_missing)}\n\n")
        for rec in all_missing:
            f.write(
                f"split={rec['split']}\t"
                f"row_idx={rec['row_idx']}\t"
                f"side={rec['side']}\t"
                f"wordnet_id={rec['wordnet_id']}\t"
                f"model={rec['model']}\t"
                f"expected_dir={rec['expected_dir']}\n"
            )

print(f"Done. Missing entries: {len(all_missing)}")
print(f"Saved report to: {OUT_TXT}")