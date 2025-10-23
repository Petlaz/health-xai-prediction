"""Baseline model training pipeline for the Health XAI project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils import ensure_directory, save_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "health_clean.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"
DATA_SPLITS_PATH = MODEL_DIR / "data_splits.joblib"
NN_MODEL_PATH = MODEL_DIR / "neural_network.pt"
NN_CONFIG_PATH = MODEL_DIR / "neural_network_config.json"
TARGET_COLUMN = "hltprhc"
RANDOM_STATE = 42


class FeedForwardNN(nn.Module):
    """Simple feed-forward neural network for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (64, 32)):
        super().__init__()
        self.hidden_dims = hidden_dims
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[1], 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape {df.shape} from {path}")
    return df


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Produce stratified train/validation/test splits (70/15/15)."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )
    print(
        "[INFO] Completed stratified split: "
        f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    random_state: int = RANDOM_STATE,
) -> FeedForwardNN:
    """Train a simple feed-forward neural network."""
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FeedForwardNN(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 128
    max_epochs = 50
    patience = 5
    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        print(
            f"[NN] Epoch {epoch:02d} | train_loss={epoch_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("[NN] Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model.cpu()


def train_all_models() -> Dict[str, Path]:
    """Train baseline models and persist artefacts to disk."""
    ensure_directory(MODEL_DIR)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = load_dataset()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    save_model(scaler, SCALER_PATH)

    model_paths: Dict[str, Path] = {}
    model_paths["standard_scaler"] = SCALER_PATH

    # Logistic Regression
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    log_reg.fit(X_train_scaled, y_train)
    model_paths["logistic_regression"] = save_model(
        log_reg, MODEL_DIR / "logistic_regression.joblib"
    )
    print(f"✅ Trained Logistic Regression on {X_train.shape[0]} samples.")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    model_paths["random_forest"] = save_model(rf, MODEL_DIR / "random_forest.joblib")
    print(f"✅ Trained Random Forest on {X_train.shape[0]} samples.")

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        reg_lambda=1.0,
        use_label_encoder=False,
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    model_paths["xgboost"] = save_model(xgb, MODEL_DIR / "xgboost_classifier.joblib")
    print(f"✅ Trained XGBoost on {X_train.shape[0]} samples.")

    # Neural Network
    nn_model = train_neural_network(
        X_train_scaled,
        y_train.to_numpy(),
        X_val_scaled,
        y_val.to_numpy(),
        input_dim=X_train.shape[1],
    )
    model_paths["neural_network"] = save_model(nn_model, NN_MODEL_PATH)
    with NN_CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"input_dim": X_train.shape[1], "hidden_dims": list(nn_model.hidden_dims)}, handle, indent=2)
    print(f"✅ Trained Neural Network on {X_train.shape[0]} samples.")

    # Persist data splits for evaluation
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
    joblib.dump(splits, DATA_SPLITS_PATH)
    print(f"[INFO] Saved data splits to {DATA_SPLITS_PATH}")

    return model_paths


if __name__ == "__main__":
    train_all_models()

# Example usage: python -m src.train_models