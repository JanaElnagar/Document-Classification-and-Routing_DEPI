"""Class-imbalance and augmentation experiments. Owner: Member 1. Requirement: FR14.

TODO:
- Compare: no handling vs class_weight='balanced' vs random oversampling vs SMOTE on TF-IDF
- Report per-class F1 for each setting (docrouter.evaluate.compute_metrics)
- Synthetic samples: training split only, never val/test
"""


def train(train_df, val_df):
    """Train on train_df["text"], train_df["label"]; tune on val_df. Return a fitted model."""
    raise NotImplementedError("Member 1: implement Class-imbalance and augmentation experiments.")
