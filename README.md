# Fraud Detection with PyOD AutoEncoder

This repo trains an AutoEncoder (from [PyOD](https://pyod.readthedocs.io/)) to detect fraud on the Kaggle credit card dataset.

## Environment Setup

### Option A: Local + VS Code
1. Install Python 3.9+
2. Create and activate a virtual environment:
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
   - Windows: `python -m venv .venv && .venv\\Scripts\\activate`
3. `pip install -r requirements.txt`

### Option B: Amazon SageMaker Studio Lab (Free)
- Create a new Python environment, upload the files, and run the script via terminal:  
  `python fraud_autoencoder_pyod.py --data_path creditcard.csv`

## Data

Download **creditcard.csv** from Kaggle (Anonymized Credit Card Transactions). Put the file in this folder or pass `--data_path` with the correct path.

## Run

```bash
python fraud_autoencoder_pyod.py --data_path creditcard.csv --epochs 30 --batch_size 128
```
Outputs will be saved to `./outputs/`:
- `metrics.json` — ROC-AUC, PR-AUC, precision/recall/F1
- `scores_predictions.csv`
- `hist_scores.png`, `pr_curve.png`, `roc_curve.png`, `confusion_matrix.png`
- `run_summary.txt` — copy/paste or screenshot into your report

## GitHub

1. Create a new GitHub repo (public or private).
2. Add these files, commit, and push.
3. Share the repo URL in your submission.
