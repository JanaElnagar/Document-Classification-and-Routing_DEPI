"""Load the processed dataset and its fixed train / val / test splits."""
from pathlib import Path

import pandas as pd

from .preprocess import clean_text

DEFAULT_PATH = Path("data/processed/dataset.csv")


def load_splits(path=DEFAULT_PATH):
    """Return {"train": df, "val": df, "test": df} with cleaned `text` and `label`."""
    df = pd.read_csv(path)
    missing = {"text", "label", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
