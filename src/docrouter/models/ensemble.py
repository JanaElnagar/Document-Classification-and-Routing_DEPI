"""TF-IDF + Random Forest / XGBoost. Owner: Member 3. Requirement: FR11.

TODO:
- Reuse the TfidfVectorizer settings from train_baseline.py
- Try RandomForestClassifier and XGBClassifier (pip install xgboost)
- Feature selection: chi2 / SelectKBest on the TF-IDF matrix
- Log each run to MLflow with the same metric names as the baseline
"""


def train(train_df, val_df):
    """Train on train_df["text"], train_df["label"]; tune on val_df. Return a fitted model."""
    raise NotImplementedError("Member 3: implement TF-IDF + Random Forest / XGBoost.")
