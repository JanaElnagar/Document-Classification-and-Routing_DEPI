"""Fine-tuned Hugging Face Transformer. Owner: Member 5. Requirement: FR13.

TODO:
- Start with distilbert-base-uncased, max_length 512
- transformers Trainer with eval on val_df each epoch; early stopping
- Wrap the result so it exposes classes_ / predict_proba like the other models
- Log params, metrics and the model to MLflow
"""


def train(train_df, val_df):
    """Train on train_df["text"], train_df["label"]; tune on val_df. Return a fitted model."""
    raise NotImplementedError("Member 5: implement Fine-tuned Hugging Face Transformer.")
