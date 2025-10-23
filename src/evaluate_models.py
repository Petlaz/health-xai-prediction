"""Model evaluation and error analysis for baseline classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.train_models import FeedForwardNN
from src.utils import (
    ensure_directory,
    load_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "health_clean.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
CONFUSION_DIR = PROJECT_ROOT / "results" / "confusion_matrices"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
METRICS_SUMMARY_PATH = METRICS_DIR / "metrics_summary.csv"
MISCLASSIFIED_PATH = METRICS_DIR / "misclassified_samples.csv"
REPORTS_DIR = METRICS_DIR / "classification_reports"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"
DATA_SPLITS_PATH = MODEL_DIR / "data_splits.joblib"
NN_MODEL_PATH = MODEL_DIR / "neural_network.pt"
NN_CONFIG_PATH = MODEL_DIR / "neural_network_config.json"
TARGET_COLUMN = "hltprhc"

MODEL_PATHS = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "random_forest": MODEL_DIR / "random_forest.joblib",
    "xgboost": MODEL_DIR / "xgboost_classifier.joblib",
}
NEURAL_MODEL_NAME = "neural_network"


def load_splits() -> Dict[str, pd.DataFrame]:
    """Load cached data splits or recreate them if unavailable."""
    if DATA_SPLITS_PATH.exists():
        print(f"[INFO] Loading data splits from {DATA_SPLITS_PATH}")
        splits = joblib.load(DATA_SPLITS_PATH)
        return splits

    print("[WARN] Data splits not found; recreating from dataset.")
    df = pd.read_csv(DATA_PATH)
    from src.train_models import stratified_split  # local import to avoid cycles

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    joblib.dump(splits, DATA_SPLITS_PATH)
    return splits


def load_models() -> Tuple[Dict[str, object], object]:
    """Load trained models, scaler, and neural network architecture."""
    from sklearn.preprocessing import StandardScaler

    missing_models = [name for name, path in MODEL_PATHS.items() if not path.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Trained models are missing. "
            f"Please run `python src/train_models.py` first. Missing: {missing_models}"
        )

    scaler: StandardScaler = load_model(SCALER_PATH)

    models = {
        name: joblib.load(path)
        for name, path in MODEL_PATHS.items()
    }

    if not NN_CONFIG_PATH.exists() or not NN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Neural network artefacts missing. Ensure training step completed."
        )

    with NN_CONFIG_PATH.open(encoding="utf-8") as handle:
        nn_config = json.load(handle)

    nn_model = FeedForwardNN(
        input_dim=nn_config["input_dim"],
        hidden_dims=tuple(nn_config.get("hidden_dims", (64, 32))),
    )
    load_model(NN_MODEL_PATH, nn_model)
    models[NEURAL_MODEL_NAME] = nn_model

    return models, scaler


def evaluate_models() -> pd.DataFrame:
    """Run evaluation across baseline models and persist results."""
    ensure_directory(METRICS_DIR)
    ensure_directory(CONFUSION_DIR)
    ensure_directory(PLOTS_DIR)
    ensure_directory(REPORTS_DIR)

    splits = load_splits()
    models, scaler = load_models()

    X_val = splits["X_val"]
    X_test = splits["X_test"]
    y_val = splits["y_val"].astype(int)
    y_test = splits["y_test"].astype(int)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    metrics_records = []
    misclassified_frames = []

    datasets = {
        "validation": (X_val, X_val_scaled, y_val),
        "test": (X_test, X_test_scaled, y_test),
    }

    for model_name, model in models.items():
        for dataset_name, (X_df, X_scaled, y_true) in datasets.items():
            if model_name in {"logistic_regression", NEURAL_MODEL_NAME}:
                features = X_scaled
            else:
                features = X_df.to_numpy()

            if model_name == NEURAL_MODEL_NAME:
                with torch.no_grad():
                    tensor = torch.tensor(features, dtype=torch.float32)
                    logits = model(tensor)
                    y_score = torch.sigmoid(logits).numpy()
            else:
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(features)[:, 1]
                elif hasattr(model, "decision_function"):
                    decision = model.decision_function(features)
                    y_score = 1 / (1 + np.exp(-decision))
                else:
                    raise AttributeError(
                        f"Model {model_name} does not support probability predictions."
                    )

            y_pred = (y_score >= 0.5).astype(int)

            roc_auc = np.nan
            if len(np.unique(y_true)) > 1:
                roc_auc = roc_auc_score(y_true, y_score)

            metrics_records.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, zero_division=0),
                    "roc_auc": roc_auc,
                }
            )

            plot_confusion_matrix(
                y_true, y_pred, model_name, dataset_name, CONFUSION_DIR
            )
            plot_roc_curve(y_true, y_score, model_name, dataset_name, PLOTS_DIR)
            plot_precision_recall_curve(
                y_true, y_score, model_name, dataset_name, PLOTS_DIR
            )

            report = classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T
            report_path = REPORTS_DIR / f"{model_name}_{dataset_name}_classification_report.csv"
            report_df.to_csv(report_path)
            print(f"[INFO] Classification report saved to {report_path}")

            if dataset_name == "test":
                misclassified = X_df.copy()
                misclassified["y_true"] = y_true.values
                misclassified["y_pred"] = y_pred
                misclassified["y_score"] = y_score
                misclassified["model"] = model_name
                misclassified["error_type"] = np.where(
                    misclassified["y_true"] > misclassified["y_pred"],
                    "False Negative",
                    "False Positive",
                )
                misclassified_frames.append(
                    misclassified[misclassified["y_true"] != misclassified["y_pred"]]
                )

        print(f"[INFO] Completed evaluation for {model_name}.")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(METRICS_SUMMARY_PATH, index=False)
    print(f"[INFO] Metrics summary saved to {METRICS_SUMMARY_PATH}")

    if misclassified_frames:
        misclassified_df = pd.concat(misclassified_frames).sort_values(
            by=["model", "y_score"], ascending=[True, False]
        )
        misclassified_df.to_csv(MISCLASSIFIED_PATH, index=False)
        print(f"[INFO] Misclassified samples saved to {MISCLASSIFIED_PATH}")
        print("[INFO] Top 5 misclassified examples:")
        print(misclassified_df.head().to_string())
    else:
        print("[INFO] No misclassified samples detected on the test set.")

    return metrics_df


if __name__ == "__main__":
    evaluate_models()

# Example usage: python -m src.evaluate_models
