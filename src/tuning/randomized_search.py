"""Randomized search utilities for Week 3–4 model tuning."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.neural_network import HealthNN, get_device
from src.utils import (
    check_model_fit,
    ensure_directory,
    get_top_model,
    log_model_diagnostic,
    save_model,
    save_top_model,
)

warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
DEFAULT_N_JOBS = int(os.getenv("TUNING_N_JOBS", "-1"))


def _create_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 40,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Randomized search for Logistic Regression prioritising recall."""

    model = LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
    )
    param_distributions = [
        {"penalty": ["l2"], "C": uniform(0.001, 10)},
        {"penalty": ["l1"], "C": uniform(0.001, 10)},
        {
            "penalty": ["elasticnet"],
            "C": uniform(0.001, 10),
            "l1_ratio": uniform(0, 1),
        },
    ]

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="recall",
        cv=5,
        verbose=2,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 30,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Randomized search for Random Forest with constrained depth."""

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
    )
    param_distributions = {
        "n_estimators": randint(200, 600),
        "max_depth": [8, 10, 12, 14],
        "min_samples_split": randint(3, 8),
        "min_samples_leaf": randint(2, 6),
        "max_features": ["sqrt", 0.5],
        "bootstrap": [True],
        "criterion": ["gini", "entropy"],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="recall",
        cv=5,
        verbose=2,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 50,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Randomized search for XGBoost classifier."""

    class_ratio = y_train.value_counts()
    scale_pos_weight = class_ratio[0] / class_ratio[1]

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        scale_pos_weight=scale_pos_weight,
    )

    param_distributions = {
        "n_estimators": randint(200, 800),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "reg_lambda": uniform(0.5, 3),
        "reg_alpha": uniform(0, 1),
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="recall",
        cv=5,
        verbose=2,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_iter: int = 15,
    random_state: int = 42,
    device: torch.device | None = None,
) -> Tuple[HealthNN, Dict[str, Any], float]:
    """Random search over HealthNN hyperparameters."""

    rng = np.random.default_rng(random_state)
    device = device or get_device()
    print(f"[INFO] Using device: {device}")

    input_dim = X_train.shape[1]
    best_recall = -np.inf
    best_params: Dict[str, Any] = {}
    best_state = None

    # tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32, device=device).unsqueeze(1)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight_value = neg_count / max(pos_count, 1.0)
    pos_weight = torch.tensor(pos_weight_value, device=device)

    for trial in range(1, n_iter + 1):
        hidden_dim = int(rng.choice([64, 96, 128]))
        dropout = float(rng.uniform(0.25, 0.5))
        lr = float(rng.uniform(5e-4, 2e-3))
        weight_decay = float(rng.uniform(5e-5, 5e-3))
        batch_size = int(rng.choice([64, 96, 128]))
        max_epochs = 40
        patience = 6
        patience_counter = 0

        model = HealthNN(input_dim, hidden_dim, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_trial_recall = -np.inf
        best_trial_state = None

        for epoch in range(max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_probs = torch.sigmoid(val_logits)
                recall = recall_score(
                    y_val_tensor.cpu().numpy().ravel(),
                    (val_probs.cpu().numpy().ravel() > 0.5).astype(int),
                )

            if recall > best_trial_recall + 1e-4:
                best_trial_recall = recall
                best_trial_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        print(
            f"[Trial {trial}] Recall={best_trial_recall:.3f} | hidden={hidden_dim}, lr={lr:.5f}, "
            f"dropout={dropout:.2f}, weight_decay={weight_decay:.5f}, batch={batch_size}, epochs={epoch + 1}"
        )

        if best_trial_state is not None and best_trial_recall > best_recall:
            best_recall = best_trial_recall
            best_params = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
            }
            best_state = best_trial_state

    if best_state is None:
        raise RuntimeError("Neural network tuning failed to produce a model.")

    best_model = HealthNN(input_dim, best_params["hidden_dim"], best_params["dropout"])
    best_model.load_state_dict(best_state)
    return best_model, best_params, best_recall


def run_all_tuning(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Execute tuning for all baseline models and persist artefacts."""

    _create_dirs()
    diagnostics_path = METRICS_DIR / "model_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics_path.unlink()

    splits_path = MODELS_DIR / "data_splits.joblib"
    if not splits_path.exists():
        raise FileNotFoundError("Baseline data splits not found. Run src.train_models first.")

    splits = joblib.load(splits_path)
    X_train: pd.DataFrame = splits["X_train"]
    X_val: pd.DataFrame = splits["X_val"]
    y_train: pd.Series = splits["y_train"]
    y_val: pd.Series = splits["y_val"]

    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError("Standard scaler not found. Ensure src.train_models has completed.")
    scaler = joblib.load(scaler_path)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    tuning_results: Dict[str, Dict[str, Any]] = {}

    # Logistic Regression
    lr_search = tune_logistic_regression(X_train_scaled, y_train, random_state=random_state)
    best_lr = lr_search.best_estimator_
    train_recall = recall_score(y_train, best_lr.predict(X_train_scaled))
    val_recall = recall_score(y_val, best_lr.predict(X_val_scaled))
    diagnosis = check_model_fit(train_recall, val_recall)
    log_model_diagnostic("LogisticRegression_Tuned", train_recall, val_recall, diagnosis)
    save_model(best_lr, MODELS_DIR / "logistic_regression_tuned.joblib")
    (MODELS_DIR / "logistic_regression_tuned_params.json").write_text(
        json.dumps(lr_search.best_params_, indent=2),
        encoding="utf-8",
    )
    tuning_results["LogisticRegression_Tuned"] = {
        "best_params": lr_search.best_params_,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "diagnosis": diagnosis,
    }

    # Random Forest
    rf_search = tune_random_forest(X_train, y_train, random_state=random_state)
    best_rf = rf_search.best_estimator_
    train_recall = recall_score(y_train, best_rf.predict(X_train))
    val_recall = recall_score(y_val, best_rf.predict(X_val))
    diagnosis = check_model_fit(train_recall, val_recall)
    log_model_diagnostic("RandomForest_Tuned", train_recall, val_recall, diagnosis)
    save_model(best_rf, MODELS_DIR / "random_forest_tuned.joblib")
    (MODELS_DIR / "random_forest_tuned_params.json").write_text(
        json.dumps(rf_search.best_params_, indent=2),
        encoding="utf-8",
    )
    tuning_results["RandomForest_Tuned"] = {
        "best_params": rf_search.best_params_,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "diagnosis": diagnosis,
    }

    # XGBoost
    xgb_search = tune_xgboost(X_train, y_train, random_state=random_state)
    best_xgb = xgb_search.best_estimator_
    train_recall = recall_score(y_train, best_xgb.predict(X_train))
    val_recall = recall_score(y_val, best_xgb.predict(X_val))
    diagnosis = check_model_fit(train_recall, val_recall)
    log_model_diagnostic("XGBoost_Tuned", train_recall, val_recall, diagnosis)
    save_model(best_xgb, MODELS_DIR / "xgboost_tuned.joblib")
    (MODELS_DIR / "xgboost_tuned_params.json").write_text(
        json.dumps(xgb_search.best_params_, indent=2),
        encoding="utf-8",
    )
    tuning_results["XGBoost_Tuned"] = {
        "best_params": xgb_search.best_params_,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "diagnosis": diagnosis,
    }

    # Neural Network
    nn_model, nn_params, nn_val_recall = tune_neural_network(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        n_iter=15,
        random_state=random_state,
        device=get_device(),
    )
    nn_model.eval()
    with torch.no_grad():
        train_logits = nn_model(torch.tensor(X_train_scaled.values, dtype=torch.float32))
        val_logits = nn_model(torch.tensor(X_val_scaled.values, dtype=torch.float32))
        train_probs = torch.sigmoid(train_logits).cpu().numpy().ravel()
        val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()
    train_recall = recall_score(y_train, (train_probs > 0.5).astype(int))
    val_recall = recall_score(y_val, (val_probs > 0.5).astype(int))
    diagnosis = check_model_fit(train_recall, val_recall)
    log_model_diagnostic("NeuralNetwork_Tuned", train_recall, val_recall, diagnosis)
    save_model(nn_model, MODELS_DIR / "neural_network_tuned.pt")
    (MODELS_DIR / "neural_network_tuned_params.json").write_text(
        json.dumps({**nn_params, "val_recall": nn_val_recall}, indent=2),
        encoding="utf-8",
    )
    tuning_results["NeuralNetwork_Tuned"] = {
        "best_params": nn_params,
        "train_recall": train_recall,
        "val_recall": val_recall,
        "diagnosis": diagnosis,
    }

    print("\n" + "=" * 60)
    print("🔍 Evaluating Top Performing Model Across All Tuned Models")
    print("=" * 60)
    model_objects = {
        "LogisticRegression_Tuned": best_lr,
        "RandomForest_Tuned": best_rf,
        "XGBoost_Tuned": best_xgb,
        "NeuralNetwork_Tuned": nn_model,
    }
    save_top_model(model_objects)

    return tuning_results


__all__ = [
    "tune_logistic_regression",
    "tune_random_forest",
    "tune_xgboost",
    "tune_neural_network",
    "run_all_tuning",
]

# Run: python -m src.tuning.randomized_search