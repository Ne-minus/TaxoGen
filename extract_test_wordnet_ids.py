"""
Extract all unique WordNet IDs from the test split.

Outputs:
  - wordnet_ids.txt  : one ID per line
  - wordnet_meta.csv : wordnet_id + core_synset + definition (for reference)

Usage:
    python extract_test_wordnet_ids.py \
        --test_csv TaxoGen/splits/test.csv \
        --out_dir  TaxoGen/splits
"""

import argparse
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default="TaxoGen/splits/test.csv")
    p.add_argument("--out_dir",  default="TaxoGen/splits")
    args = p.parse_args()

    df = pd.read_csv(args.test_csv)

    id_col = "wordnet_id"
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. Available: {list(df.columns)}")

    unique_ids = sorted(df[id_col].dropna().unique())
    print(f"Total rows in test: {len(df)}")
    print(f"Unique WordNet IDs: {len(unique_ids)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Plain list
    ids_path = os.path.join(args.out_dir, "wordnet_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(str(x) for x in unique_ids) + "\n")
    print(f"Saved {len(unique_ids)} IDs → {ids_path}")

    # ID + synset + definition for reference
    meta_cols = [id_col]
    for col in ("core_synset", "definition", "core_lemma"):
        if col in df.columns:
            meta_cols.append(col)

    meta = df[meta_cols].drop_duplicates(subset=id_col).sort_values(id_col).reset_index(drop=True)
    meta_path = os.path.join(args.out_dir, "wordnet_meta.csv")
    meta.to_csv(meta_path, index=False)
    print(f"Saved metadata → {meta_path}")
    print(meta.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
