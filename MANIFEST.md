MANIFEST
Project: Fraud Detection with PyOD AutoEncoder
Created: 2025-08-10
1) Repository Contents
fraud_autoencoder_pyod.py — Main training & evaluation script using PyOD AutoEncoder.
requirements.txt — Python dependencies for running the project.
README.md — Setup instructions, dataset download, and run guide.
MANIFEST.md — This file; describes contents, inputs, outputs, and submission mapping.
2) Expected Input
creditcard.csv — Kaggle anonymized credit-card transactions dataset.
Must contain a Class column (0 = legitimate, 1 = fraud).
Place in the project root or specify with --data_path.
3) Outputs (generated in ./outputs/ after running the script)
metrics.json — ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.
scores_predictions.csv — Per-row anomaly scores & predictions.
hist_scores.png — Histogram of anomaly scores.
pr_curve.png — Precision–Recall curve.
roc_curve.png — ROC curve.
confusion_matrix.png — Confusion matrix heatmap.
run_summary.txt — Human-readable run summary (recommended for screenshot).
4) How to Run
Create and activate a virtual environment:
macOS/Linux:
python -m venv .venv && source .venv/bin/activate
Windows:
python -m venv .venv && .venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Download creditcard.csv from Kaggle and place it in this folder.
Run:
python fraud_autoencoder_pyod.py --data_path creditcard.csv --epochs 30 --batch_size 128
