"""TF-IDF + linear classifier baseline, tracked in MLflow (FR10, FR18). Owner: Member 2.

Usage:
    python -m docrouter.train_baseline --model logreg
    python -m docrouter.train_baseline --model linearsvc
    python -m docrouter.train_baseline --model nb
"""
import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import LABELS
from .data import load_splits
from .evaluate import compute_metrics, save_confusion_matrix

SEED = 42


def build_pipeline(model: str, max_features: int, ngram_max: int) -> Pipeline:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, ngram_max), min_df=2,
                                 max_features=max_features, sublinear_tf=True)
    if model == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    elif model == "linearsvc":
        # LinearSVC has no probabilities; calibration adds them (routing needs confidence)
        clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=3)
    elif model == "nb":
        clf = ComplementNB()
    else:
        raise ValueError(f"unknown model: {model}")
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/dataset.csv")
    parser.add_argument("--model", default="logreg", choices=["logreg", "linearsvc", "nb"])
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--experiment", default="doc-routing")
    parser.add_argument("--out", default="models/baseline.joblib")
    args = parser.parse_args()

    # local SQLite tracking unless MLFLOW_TRACKING_URI says otherwise
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment)

    splits = load_splits(args.data)
    train, val, test = splits["train"], splits["val"], splits["test"]
    labels = [l for l in LABELS if l in set(train["label"])]

    with mlflow.start_run(run_name=f"tfidf-{args.model}"):
        mlflow.log_params({"model": args.model, "max_features": args.max_features,
                           "ngram_max": args.ngram_max, "seed": SEED,
                           "n_train": len(train), "n_val": len(val), "n_test": len(test)})
        pipe = build_pipeline(args.model, args.max_features, args.ngram_max)
        pipe.fit(train["text"], train["label"])

        for name, part in [("val", val), ("test", test)]:
            if len(part) == 0:
                continue
            pred = pipe.predict(part["text"])
            metrics = compute_metrics(part["label"], pred, labels)
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})
            png = save_confusion_matrix(part["label"], pred, labels, f"reports/confusion_{args.model}_{name}.png")
            mlflow.log_artifact(str(png))
            print(f"{name}: macro F1 = {metrics['macro_f1']:.3f}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, args.out)
        mlflow.sklearn.log_model(pipe, name="model")
        print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
