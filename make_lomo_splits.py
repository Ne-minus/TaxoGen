#!/usr/bin/env python3
"""Build leave-one-model-out CSV folds from existing WordNet-safe splits.

For each image generator model:
  * train: original train rows where the held-out model is absent
  * val:   original val rows where the held-out model is absent
  * test:  original test rows where the held-out model is present

The original train/val/test assignment is never reshuffled, so WordNet IDs stay
in the same base split.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"model_a", "model_b", "wordnet_id"}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["wordnet_id"] = df["wordnet_id"].astype(str)
    return df


def check_wordnet_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    ids = {name: set(df["wordnet_id"]) for name, df in splits.items()}
    names = sorted(ids)
    overlaps = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = len(ids[a] & ids[b])
            if n:
                overlaps.append(f"{a}/{b}: {n}")
    if overlaps:
        joined = ", ".join(overlaps)
        raise ValueError(f"Base splits overlap by wordnet_id ({joined}).")


def all_models(splits: dict[str, pd.DataFrame]) -> list[str]:
    models = set()
    for df in splits.values():
        models.update(df["model_a"].dropna().astype(str))
        models.update(df["model_b"].dropna().astype(str))
    return sorted(models)


def absent(df: pd.DataFrame, model: str) -> pd.DataFrame:
    mask = (df["model_a"].astype(str) != model) & (df["model_b"].astype(str) != model)
    return df.loc[mask].copy()


def present(df: pd.DataFrame, model: str) -> pd.DataFrame:
    mask = (df["model_a"].astype(str) == model) | (df["model_b"].astype(str) == model)
    return df.loc[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leave-one-model-out folds without reshuffling WordNet splits."
    )
    parser.add_argument("--splits_dir", default="splits")
    parser.add_argument("--out_dir", default="splits/lomo_by_model")
    parser.add_argument(
        "--allow_empty_test",
        action="store_true",
        help="Write a fold even if the held-out model has no rows in base test.",
    )
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    base = {
        "train": read_split(splits_dir / "train.csv"),
        "val": read_split(splits_dir / "val.csv"),
        "test": read_split(splits_dir / "test.csv"),
    }
    check_wordnet_disjoint(base)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for model in all_models(base):
        fold_dir = out_dir / slugify(model)
        fold_dir.mkdir(parents=True, exist_ok=True)

        train = absent(base["train"], model)
        val = absent(base["val"], model)
        test = present(base["test"], model)
        if test.empty and not args.allow_empty_test:
            raise ValueError(f"Held-out model {model!r} has no rows in base test split.")

        train.to_csv(fold_dir / "train.csv", index=False)
        val.to_csv(fold_dir / "val.csv", index=False)
        test.to_csv(fold_dir / "test.csv", index=False)

        manifest_rows.append(
            {
                "model": model,
                "fold": fold_dir.name,
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "train_wordnet_ids": train["wordnet_id"].nunique(),
                "val_wordnet_ids": val["wordnet_id"].nunique(),
                "test_wordnet_ids": test["wordnet_id"].nunique(),
                "removed_train_rows": len(base["train"]) - len(train),
                "removed_val_rows": len(base["val"]) - len(val),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} folds to {out_dir}")
    print(manifest[["fold", "train_rows", "val_rows", "test_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
