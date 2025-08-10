import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# PyOD AutoEncoder
from pyod.models.auto_encoder import AutoEncoder


def load_data(data_path: str):
    """Load the Kaggle credit card dataset. Expects 'Class' as label (0=legit, 1=fraud)."""
    df = pd.read_csv(data_path)
    if 'Class' not in df.columns:
        raise ValueError("Expected a column named 'Class' with 0/1 labels.")

    # Separate features and labels
    y = df['Class'].astype(int).values
    X = df.drop(columns=['Class']).values

    # Scale features (AutoEncoder benefits from normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, df.columns.drop('Class')


def train_autoencoder(X_train, epochs=30, batch_size=128, contamination=None, random_state=42):
    """
    Train the PyOD AutoEncoder.
    - contamination: expected proportion of outliers in the data (float in (0, 0.5]). If None, defaults to 0.02.
    """
    if contamination is None:
        contamination = 0.02  # Conservative default; credit card fraud rate is low.

    clf = AutoEncoder(
        hidden_neurons=[64, 32, 32, 64],
        epochs=epochs,
        batch_size=batch_size,
        contamination=contamination,
        random_state=random_state,
        verbose=1,
    )
    clf.fit(X_train)
    return clf


def evaluate(clf, X, y_true, out_dir):
    """Evaluate model, save metrics and plots."""
    # Decision scores: higher means more abnormal
    scores = clf.decision_function(X)
    # Binary predictions using threshold inferred from contamination
    y_pred = clf.predict(X)  # 1 = outlier, 0 = inlier

    roc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum())
    }

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    # Save predictions for inspection
    out_df = pd.DataFrame({
        "y_true": y_true,
        "score": scores,
        "y_pred": y_pred
    })
    out_df.to_csv(os.path.join(out_dir, "scores_predictions.csv"), index=False)

    # ----- Plots (matplotlib, single figure per chart & no explicit colors) -----
    # 1) Histogram of anomaly scores
    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_scores.png"), dpi=150)
    plt.close()

    # 2) Precision-Recall curve
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close()

    # 3) ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    # 4) Confusion matrix as image
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Pred vs True)")
    plt.xlabel("Predicted (0=inlier,1=outlier)")
    plt.ylabel("True (0=legit,1=fraud)")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    return metrics_path, out_df.shape[0], report


def main():
    parser = argparse.ArgumentParser(description="PyOD AutoEncoder Fraud Detection")
    parser.add_argument("--data_path", type=str, required=True, help="Path to creditcard.csv")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--test_size", type=float, default=0.25, help="Holdout size")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--contamination", type=float, default=None, help="Expected outlier proportion, e.g., 0.001")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load & split
    X, y, feature_names = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # If contamination not provided, estimate from train labels (fallback to small value if all 0s)
    contamination = args.contamination
    if contamination is None:
        pos_rate = max(1e-4, (y_train == 1).mean())
        contamination = float(pos_rate)

    clf = train_autoencoder(X_train, epochs=args.epochs, batch_size=args.batch_size, contamination=contamination, random_state=args.random_state)

    # Evaluate on test set
    metrics_path, n_rows, cls_report = evaluate(clf, X_test, y_test, args.out_dir)

    # Save a lightweight run summary
    summary_txt = os.path.join(args.out_dir, "run_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("PyOD AutoEncoder Fraud Detection\n")
        f.write(f"Data: {args.data_path}\n")
        f.write(f"Rows (test set): {n_rows}\n")
        f.write(f"Epochs: {args.epochs}, Batch size: {args.batch_size}\n")
        f.write(f"Contamination: {contamination}\n\n")
        with open(metrics_path, "r") as mf:
            f.write(mf.read())
        f.write("\n\nClassification Report:\n")
        f.write(cls_report)

    print("DONE. Results saved to:", args.out_dir)


if __name__ == "__main__":
    main()
