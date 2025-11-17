"""Explainability utilities for generating SHAP and LIME artefacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_tabular import LimeTabularExplainer

from src.evaluate_models import (
    SCALED_FEATURE_MODELS,
    TUNED_NEURAL_MODEL_NAME,
    load_models,
    load_splits,
)
from src.utils import ensure_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPLAINABILITY_DIR = PROJECT_ROOT / "results" / "explainability"

CLASS_NAMES = ["No Heart Condition", "Heart Condition"]


@dataclass
class ModelExplainabilityConfig:
    """Configuration for a model explainability run."""

    model_key: str
    pretty_name: str
    shap_method: str  # "tree" or "kernel"


MODEL_CONFIGS: List[ModelExplainabilityConfig] = [
    ModelExplainabilityConfig("random_forest_tuned", "RandomForest_Tuned", "tree"),
    ModelExplainabilityConfig("xgboost_tuned", "XGBoost_Tuned", "tree"),
    ModelExplainabilityConfig(TUNED_NEURAL_MODEL_NAME, "NeuralNetwork_Tuned", "kernel"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP and LIME artefacts.")
    parser.add_argument(
        "--dataset",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split used for explanations.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Number of rows sampled for global SHAP plots.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=40,
        help="Background sample size for Kernel SHAP.",
    )
    parser.add_argument(
        "--kernel-nsamples",
        type=int,
        default=100,
        help="Number of SHAP Kernel samples (controls runtime).",
    )
    parser.add_argument(
        "--lime-instances",
        type=int,
        default=2,
        help="How many local LIME explanations to persist per model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling operations.",
    )
    return parser.parse_args()


def as_dataframe(data: pd.DataFrame | np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Coerce input data to a DataFrame with consistent column ordering."""
    if isinstance(data, pd.DataFrame):
        return data.loc[:, feature_names]

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=feature_names)


def make_predict_function(
    model_name: str,
    model,
    scaler,
    feature_names: Sequence[str],
) -> Callable[[pd.DataFrame | np.ndarray], np.ndarray]:
    """Return a probability prediction function compatible with SHAP/LIME."""

    def _predict(data: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = as_dataframe(data, feature_names)

        if model_name in SCALED_FEATURE_MODELS:
            features = scaler.transform(df)
        else:
            features = df

        if model_name == TUNED_NEURAL_MODEL_NAME:
            with torch.no_grad():
                tensor = torch.tensor(features, dtype=torch.float32)
                probs = torch.sigmoid(model(tensor)).numpy()
        else:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[:, 1]
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(features)
                probs = 1.0 / (1.0 + np.exp(-decision))
            else:
                raise AttributeError(f"{model_name} does not expose probability predictions.")

        probs = probs.reshape(-1, 1)
        return np.hstack([1 - probs, probs])

    return _predict


def select_local_indices(probs: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Pick representative indices for positive/negative local explanations."""
    indices = list(range(len(probs)))
    positive = next((idx for idx in indices if probs[idx] >= threshold), None)
    negative = next((idx for idx in indices if probs[idx] < threshold), None)

    if positive is None and indices:
        positive = int(np.argmax(probs))
    if negative is None and indices:
        negative = int(np.argmin(probs))

    unique_indices = []
    for idx in (positive, negative):
        if idx is not None and idx not in unique_indices:
            unique_indices.append(idx)
    return unique_indices


def extract_positive_shap_values(shap_values, expected_value):
    """Return SHAP values and expectation for the positive class."""
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            values = shap_values[1]
            exp_val = expected_value[1] if isinstance(expected_value, Iterable) else expected_value
        else:
            values = shap_values[0]
            exp_val = expected_value if not isinstance(expected_value, Iterable) else expected_value[0]
        return values, exp_val

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        values = shap_values[..., -1]
        if isinstance(expected_value, Iterable):
            exp_vals = np.asarray(expected_value)
            exp_val = exp_vals[-1] if exp_vals.ndim > 0 else float(exp_vals)
        else:
            exp_val = expected_value
        return values, float(np.ravel([exp_val])[0])

    if hasattr(shap_values, "values"):
        # shap.Explanation
        values = shap_values.values
        base = shap_values.base_values
        if isinstance(base, np.ndarray):
            if base.ndim == 1:
                exp_val = float(np.mean(base))
            else:
                exp_val = float(np.mean(base[:, -1]))
        else:
            exp_val = float(base)
        return values, exp_val

    return shap_values, expected_value if np.ndim(expected_value) == 0 else expected_value[0]


def save_shap_summary(
    model_name: str,
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    output_dir: Path,
    kind: str = "dot",
) -> Path:
    """Persist a SHAP summary plot."""
    ensure_directory(output_dir)
    path = output_dir / f"{model_name}_shap_summary_{kind}.png"
    plt.figure(figsize=(10, 6))
    plot_type = "dot" if kind == "dot" else "bar"
    kwargs = {"show": False, "plot_type": plot_type}
    if kind == "dot":
        kwargs["color"] = plt.get_cmap("coolwarm")
    shap.summary_plot(shap_values, X_sample, **kwargs)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def save_shap_force_plot(
    model_name: str,
    expected_value: float,
    shap_row: np.ndarray,
    feature_row: pd.Series,
    output_dir: Path,
    label_suffix: str,
) -> Path:
    """Persist a SHAP force plot as HTML."""
    ensure_directory(output_dir)
    plt.figure(figsize=(9, 2.5))
    shap.force_plot(
        float(np.ravel([expected_value])[0]),
        np.array(shap_row, dtype=float),
        feature_row,
        matplotlib=True,
        show=False,
    )
    path = output_dir / f"{model_name}_force_{label_suffix}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def generate_tree_shap(
    model_name: str,
    model,
    X_sample: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, float]:
    """Compute SHAP values for tree-based models."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    positive_values, expected_value = extract_positive_shap_values(shap_values, explainer.expected_value)
    dot_path = save_shap_summary(model_name, X_sample, positive_values, output_dir, kind="dot")
    bar_path = save_shap_summary(model_name, X_sample, positive_values, output_dir, kind="bar")
    print(f"[SHAP] Saved summary plots to {dot_path} and {bar_path}")
    return positive_values, expected_value


def generate_kernel_shap(
    model_name: str,
    predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
    background: pd.DataFrame,
    X_sample: pd.DataFrame,
    nsamples: int,
    output_dir: Path,
) -> tuple[np.ndarray, float]:
    """Compute SHAP values using the KernelExplainer."""
    explainer = shap.KernelExplainer(lambda data: predict_fn(data)[:, 1], background)
    shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
    positive_values, expected_value = extract_positive_shap_values(shap_values, explainer.expected_value)
    dot_path = save_shap_summary(model_name, X_sample, positive_values, output_dir, kind="dot")
    bar_path = save_shap_summary(model_name, X_sample, positive_values, output_dir, kind="bar")
    print(f"[SHAP] Saved summary plots to {dot_path} and {bar_path}")
    return positive_values, expected_value


def save_lime_explanations(
    model_name: str,
    explainer: LimeTabularExplainer,
    predict_fn: Callable[[pd.DataFrame | np.ndarray], np.ndarray],
    instances: List[pd.Series],
    instance_labels: List[str],
    output_dir: Path,
) -> List[Path]:
    """Persist LIME explanations for provided instances."""
    lime_paths: List[Path] = []
    for row, label in zip(instances, instance_labels):
        exp = explainer.explain_instance(
            row.to_numpy(),
            predict_fn,
            num_features=10,
            top_labels=1,
        )
        path = output_dir / f"{model_name}_lime_{label}.html"
        exp.save_to_file(str(path))
        lime_paths.append(path)
    return lime_paths


def build_lime_explainer(
    X_train: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_features: Sequence[int] | None = None,
) -> LimeTabularExplainer:
    """Initialise a LIME tabular explainer."""
    categorical_features = list(categorical_features or [])
    return LimeTabularExplainer(
        training_data=X_train.loc[:, feature_names].to_numpy(),
        feature_names=list(feature_names),
        class_names=CLASS_NAMES,
        categorical_features=categorical_features if categorical_features else None,
        mode="classification",
        discretize_continuous=True,
    )


def infer_categorical_indices(columns: Sequence[str]) -> List[int]:
    """Return indices for one-hot encoded categorical columns."""
    return [idx for idx, name in enumerate(columns) if name.startswith("categorical__")]


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    splits = load_splits()
    X_train = splits["X_train"]
    feature_names = list(X_train.columns)

    dataset_prefix = "val" if args.dataset == "validation" else "test"
    X_target = splits[f"X_{dataset_prefix}"]
    sample_size = min(args.sample_size, len(X_target))
    sample_df = X_target.sample(n=sample_size, random_state=args.random_state).copy()
    sample_df = sample_df.reset_index(drop=True)

    models, scaler = load_models(input_dim=X_train.shape[1], include_tuned=True)

    lime_explainer = build_lime_explainer(
        X_train,
        feature_names,
        categorical_features=infer_categorical_indices(feature_names),
    )

    background_size = min(args.background_size, len(X_train))
    kernel_background = X_train.sample(n=background_size, random_state=args.random_state)

    ensure_directory(EXPLAINABILITY_DIR)
    summary_records: List[Dict[str, object]] = []

    for config in MODEL_CONFIGS:
        model = models.get(config.model_key)
        if model is None:
            print(f"[WARN] {config.model_key} not found. Skipping.")
            continue

        model_dir = ensure_directory(EXPLAINABILITY_DIR / config.pretty_name)
        predict_fn = make_predict_function(config.model_key, model, scaler, feature_names)

        probs = predict_fn(sample_df)[:, 1]
        selected_indices = select_local_indices(probs)
        instance_rows = [sample_df.iloc[idx] for idx in selected_indices]
        instance_labels = [f"idx{idx}_p{probs[idx]:.2f}" for idx in selected_indices]

        if config.shap_method == "tree":
            shap_values, expected_value = generate_tree_shap(
                config.model_key,
                model,
                sample_df,
                model_dir,
            )
        else:
            shap_values, expected_value = generate_kernel_shap(
                config.model_key,
                predict_fn,
                kernel_background,
                sample_df,
                nsamples=args.kernel_nsamples,
                output_dir=model_dir,
            )

        importance = np.mean(np.abs(shap_values), axis=0)
        importance_series = pd.Series(importance, index=sample_df.columns).sort_values(ascending=False)
        top_features_path = model_dir / f"{config.model_key}_top_features.csv"
        importance_series.to_csv(top_features_path, header=["mean_abs_shap"])

        # Persist force plots for the selected rows.
        force_paths = []
        for idx, label in zip(selected_indices, instance_labels):
            force_path = save_shap_force_plot(
                config.model_key,
                expected_value,
                shap_values[idx],
                sample_df.iloc[idx],
                model_dir,
                label,
            )
            print(f"[SHAP] Saved force plot to {force_path}")
            force_paths.append(force_path)
        lime_paths = save_lime_explanations(
            config.model_key,
            lime_explainer,
            predict_fn,
            instance_rows,
            instance_labels,
            model_dir,
        )

        summary_records.append(
            {
                "model": config.pretty_name,
                "dataset": args.dataset,
                "sample_size": sample_size,
                "lime_examples": ", ".join(Path(path).name for path in lime_paths),
                "shap_force_examples": ", ".join(path.name for path in force_paths),
                "top_features_csv": top_features_path.name,
            }
        )

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = EXPLAINABILITY_DIR / f"xai_summary_{args.dataset}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Logged explainability outputs to {summary_path}")
    else:
        print("[WARN] No explainability outputs were generated.")


if __name__ == "__main__":
    main()

# Run the script with: python -m src.explainability \
  #--dataset validation \
  #--sample-size 120 \
  #--background-size 35 \
  #--kernel-nsamples 80

# Dataset test: python -m src.explainability \
  #--dataset test \
  #--sample-size 120 \
  #--background-size 35 \
  #--kernel-nsamples 80

