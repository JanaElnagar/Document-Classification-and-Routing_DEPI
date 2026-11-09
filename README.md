# AI Document Classification & Routing System

Classifies incoming document text into six categories (Application, Complaint, Contract, Correspondence, Invoice, Request) and recommends the department it should go to. Documents the model is unsure about go to manual review.

ML course graduation project · team of six · final delivery 4 December 2026.

## How it fits together

```
text ──► clean_text ──► model (predict_proba) ──► routing.yaml ──► category + confidence + department
         (preprocess.py)  (models/*.joblib)        (threshold)       │
                                                                    └─► predictions.db (no document text)
```

- **Training:** every model reads the same `data/processed/dataset.csv` (fixed train / val / test splits) and logs to MLflow.
- **Serving:** FastAPI (`/classify`, `/classify/batch`, `/health`) and a Streamlit demo that calls it.

## Repository layout

```
configs/routing.yaml          category -> department, review threshold (edit without retraining)
data/processed/dataset.csv    built by notebooks/02 (not in git, share via Kaggle dataset)
notebooks/01_enron_labeling   label Request / Correspondence emails
notebooks/02_build_dataset    sample all six categories, write dataset.csv
src/docrouter/preprocess.py   clean_text: the ONLY text cleaning, used in training and the API
src/docrouter/data.py         load_splits()
src/docrouter/evaluate.py     shared metrics + confusion matrix (use these for every model)
src/docrouter/train_baseline.py  TF-IDF + LogReg / LinearSVC / NB, logged to MLflow
src/docrouter/models/         one file per member's experiment (stubs with TODOs)
src/docrouter/routing.py      probabilities -> routing decision
src/docrouter/api.py          FastAPI service
app/streamlit_app.py          demo UI
tests/                        pytest, run by GitHub Actions on every push
```

## Setup

```bash
git clone <this repo> && cd doc-routing
bash setup.sh            # Windows: setup.bat  (creates .venv and installs requirements.txt)
export PYTHONPATH=src    # Windows: set PYTHONPATH=src
pytest -q                # should pass before you change anything
```

The notebooks don't need this: Kaggle already has pandas, scikit-learn, datasets and transformers. LSTM / Transformer work also uses `requirements-dl.txt` (torch), which Kaggle has too.

## Day-to-day commands

| What | Command |
|---|---|
| Build the dataset | run `notebooks/02_build_dataset.ipynb` on Kaggle, save `dataset.csv` to `data/processed/` |
| Train the baseline | `python -m docrouter.train_baseline --model logreg` (also `linearsvc`, `nb`) |
| See experiments | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 |
| Run the API | `uvicorn docrouter.api:app --reload --app-dir src` → http://localhost:8000/docs |
| Run the demo | `streamlit run app/streamlit_app.py` (API must be running) |
| Everything in Docker | `docker compose up --build` (needs `models/baseline.joblib`) |

## Who owns what

| Member | ML work | Code | Supporting task |
|---|---|---|---|
| 1 | Class imbalance and augmentation | `models/imbalance.py` | dataset, data card |
| 2 | TF-IDF + linear models | `train_baseline.py` | `preprocess.py` |
| 3 | TF-IDF + ensembles | `models/ensemble.py` | `evaluate.py`, evaluation report |
| 4 | BiLSTM | `models/lstm.py` | MLflow setup |
| 5 | Transformer fine-tuning | `models/transformer.py` | error analysis |
| 6 | Calibration, threshold, compression | `models/calibration.py` | `api.py`, `app/`, Docker |

Every model follows the same contract (see `src/docrouter/models/__init__.py`): it exposes `classes_`, `predict_proba(texts)` and `predict(texts)`, so the API and the evaluation work with any of them.

## Team rules

1. **Never commit to `main` directly.** Branch per task (`member4/lstm-baseline`), open a pull request, and get one review.
2. **Tests must pass** (GitHub Actions runs `pytest` on every PR).
3. **No data or model files in git.** `.gitignore` blocks them; share through a Kaggle dataset.
4. **Every training run goes to MLflow.** Results that aren't in MLflow don't count for model selection.
5. **Use `clean_text` and `evaluate.compute_metrics`**, never your own copies, so all models are comparable.
6. **Don't tune on the test split.** Use `val` for every choice; `test` is touched once per final model.

## Data sources and limitations

| Category | Source | Limitation to report |
|---|---|---|
| Request, Correspondence | Enron emails, LLM-reviewed (`label_source` column) | no inter-annotator agreement measured |
| Complaint | CFPB consumer complaint narratives | personal details masked as `XXXX` (removed by `clean_text`) |
| Application | Resume dataset (24 job fields) | resumes stand in for applications |
| Contract | CUAD commercial contracts (first 3,000 characters) | much longer than other categories |
| Invoice | OCR text of synthetic invoices | synthetic layouts |

One source per category means a model can learn **source style** instead of category. If the baseline scores near 100%, check errors per `source` before trusting it.
