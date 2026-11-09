"""Confidence calibration, threshold tuning, compression. Owner: Member 6. Requirement: FR15, FR21.

TODO:
- Calibrate the chosen model on val_df (temperature scaling or CalibratedClassifierCV)
- Pick the routing threshold: plot automation rate vs routing accuracy on val_df
- Write the chosen threshold into configs/routing.yaml
- Export the selected model to ONNX or quantize it; compare latency
"""


def train(train_df, val_df):
    """Train on train_df["text"], train_df["label"]; tune on val_df. Return a fitted model."""
    raise NotImplementedError("Member 6: implement Confidence calibration, threshold tuning, compression.")
