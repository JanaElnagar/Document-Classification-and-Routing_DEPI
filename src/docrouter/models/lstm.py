"""BiLSTM with pretrained word embeddings. Owner: Member 4. Requirement: FR12.

TODO:
- Tokenize with the same cleaned text (docrouter.preprocess.clean_text)
- Embeddings: GloVe 100d (or train your own); max length 400 tokens
- BiLSTM -> dropout -> linear; class weights for imbalance
- Run on Kaggle GPU; requirements-dl.txt has torch
"""


def train(train_df, val_df):
    """Train on train_df["text"], train_df["label"]; tune on val_df. Return a fitted model."""
    raise NotImplementedError("Member 4: implement BiLSTM with pretrained word embeddings.")
