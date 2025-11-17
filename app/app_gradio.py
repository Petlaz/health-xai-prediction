"""Interactive Gradio demo with model thresholds and SHAP context."""

from __future__ import annotations

import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.preprocessing import StandardScaler

from src.models.neural_network import HealthNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "health_clean.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
THRESHOLD_PATH = METRICS_DIR / "threshold_recommendations.csv"
TARGET_COLUMN = "hltprhc"
SCALER_PATH = MODEL_DIR / "standard_scaler.joblib"

LOGISTIC_PATH = MODEL_DIR / "logistic_regression_tuned.joblib"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest_tuned.joblib"
XGBOOST_PATH = MODEL_DIR / "xgboost_tuned.joblib"
NEURAL_STATE_PATH = MODEL_DIR / "neural_network_tuned.pt"
NEURAL_CONFIG_PATH = MODEL_DIR / "neural_network_tuned_params.json"

# Key numeric features surfaced in the UI (top SHAP drivers).
DISPLAY_FEATURES = [
    "numeric__health",
    "numeric__dosprt",
    "numeric__flteeff",
    "numeric__slprl",
    "numeric__weighta",
    "numeric__cgtsmok",
    "numeric__alcfreq",
    "numeric__happy",
]

FEATURE_LABELS = {
    "numeric__health": "Self-reported health (1 very good – 5 very bad)",
    "numeric__dosprt": "Sports frequency",
    "numeric__flteeff": "Everything felt an effort",
    "numeric__slprl": "Restless sleep score",
    "numeric__weighta": "Weight (kg)",
    "numeric__cgtsmok": "Smoking frequency",
    "numeric__alcfreq": "Alcohol frequency",
    "numeric__happy": "Happiness score",
}


@dataclass
class ModelWrapper:
    key: str
    display: str
    requires_scaling: bool
    predict_fn: callable
    shap_explainer: shap.Explainer | None

    def predict_proba(self, features: np.ndarray, scaler: StandardScaler) -> float:
        data = scaler.transform(features) if self.requires_scaling else features
        return float(self.predict_fn(data))


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column `{TARGET_COLUMN}`.")
    return df


def load_thresholds() -> Dict[str, float]:
    if not THRESHOLD_PATH.exists():
        return {}
    df = pd.read_csv(THRESHOLD_PATH)
    return {row["model"]: float(row["threshold"]) for _, row in df.iterrows()}


def load_scaler() -> StandardScaler:
    return joblib.load(SCALER_PATH)


def _load_neural_model(input_dim: int) -> HealthNN:
    with NEURAL_CONFIG_PATH.open(encoding="utf-8") as handle:
        params = json.load(handle)
    hidden_dim = int(params.get("hidden_dim", 128))
    dropout = float(params.get("dropout", 0.3))
    model = HealthNN(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    state_dict = torch.load(NEURAL_STATE_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_models(
    scaler: StandardScaler, background_df: pd.DataFrame
) -> Dict[str, ModelWrapper]:
    wrappers: Dict[str, ModelWrapper] = {}
    background = background_df.sample(n=min(len(background_df), 80), random_state=42)
    scaled_background = scaler.transform(background)

    # Logistic Regression
    logistic = joblib.load(LOGISTIC_PATH)

    def logistic_predict(features: np.ndarray) -> float:
        return logistic.predict_proba(features)[:, 1][0]

    wrappers["logistic_regression_tuned"] = ModelWrapper(
        key="logistic_regression_tuned",
        display="Logistic Regression (Tuned)",
        requires_scaling=True,
        predict_fn=logistic_predict,
        shap_explainer=None,  # Kernel SHAP would be slow; skip for now.
    )

    # Random Forest (tree SHAP)
    random_forest = joblib.load(RANDOM_FOREST_PATH)
    rf_explainer = shap.TreeExplainer(random_forest)

    def rf_predict(features: np.ndarray) -> float:
        return random_forest.predict_proba(features)[:, 1][0]

    wrappers["random_forest_tuned"] = ModelWrapper(
        key="random_forest_tuned",
        display="Random Forest (Tuned)",
        requires_scaling=False,
        predict_fn=rf_predict,
        shap_explainer=rf_explainer,
    )

    # XGBoost (tree SHAP)
    xgb_model = joblib.load(XGBOOST_PATH)
    xgb_explainer = shap.TreeExplainer(xgb_model)

    def xgb_predict(features: np.ndarray) -> float:
        return xgb_model.predict_proba(features)[:, 1][0]

    wrappers["xgboost_tuned"] = ModelWrapper(
        key="xgboost_tuned",
        display="XGBoost (Tuned)",
        requires_scaling=False,
        predict_fn=xgb_predict,
        shap_explainer=xgb_explainer,
    )

    # Neural network (Kernel SHAP on reduced background)
    if NEURAL_STATE_PATH.exists():
        nn_model = _load_neural_model(input_dim=background.shape[1])

        def nn_predict(features: np.ndarray) -> float:
            tensor = torch.tensor(features, dtype=torch.float32)
            logits = nn_model(tensor).detach().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            return float(probs[0])

        # Kernel SHAP inside Docker is too heavy; skip per-request explanations.
        wrappers["neural_network_tuned"] = ModelWrapper(
            key="neural_network_tuned",
            display="Neural Network (Recall-first)",
            requires_scaling=True,
            predict_fn=nn_predict,
            shap_explainer=None,
        )

    return wrappers


def build_feature_template(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, float]]]:
    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    medians = df[feature_columns].median()
    template = medians.to_numpy()
    display_stats = {}
    for feature in DISPLAY_FEATURES:
        series = df[feature]
        display_stats[feature] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "value": float(series.median()),
        }
    return feature_columns, template, display_stats


def infer(
    model_key: str,
    threshold: float,
    *feature_values: float,
    wrappers: Dict[str, ModelWrapper],
    scaler: StandardScaler,
    feature_columns: List[str],
    template_values: np.ndarray,
) -> Tuple[str, Dict[str, float], List[List[str]]]:
    model = wrappers[model_key]
    instance = template_values.copy()

    feature_map = {name: value for name, value in zip(DISPLAY_FEATURES, feature_values)}
    for feature, value in feature_map.items():
        if feature in feature_columns:
            idx = feature_columns.index(feature)
            instance[idx] = value

    instance_df = pd.DataFrame([instance], columns=feature_columns)
    matrix = instance_df.to_numpy()
    proba = model.predict_proba(matrix, scaler)
    predicted_label = int(proba >= threshold)
    label_text = "Positive (At-risk)" if predicted_label else "Negative"

    metrics = {
        "model": model.display,
        "probability": round(proba, 4),
        "threshold": round(threshold, 2),
        "classification": label_text,
    }

    explanation_rows: List[List[str]] = []
    explainer = model.shap_explainer
    if explainer is not None:
        shap_input = scaler.transform(matrix) if model.requires_scaling else matrix
        shap_values = explainer.shap_values(shap_input)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        values = shap_values[0]
        contributions = sorted(
            zip(feature_columns, values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        for feature, value in contributions:
            explanation_rows.append(
                [FEATURE_LABELS.get(feature, feature), f"{value:+.4f}"]
            )
    else:
        explanation_rows.append(
            ["Explainability", "Available for RandomForest/XGBoost in this build."]
        )

    return label_text, metrics, explanation_rows


def build_interface():
    df = load_dataset()
    thresholds = load_thresholds()
    feature_columns, template_values, display_stats = build_feature_template(df)
    scaler = load_scaler()
    wrappers = load_models(scaler, df[feature_columns])

    default_model = (
        "xgboost_tuned" if "xgboost_tuned" in wrappers else next(iter(wrappers.keys()))
    )
    default_threshold = thresholds.get(default_model, 0.65)

    with gr.Blocks(title="Health Risk Explainability Demo") as demo:
        gr.Markdown(
            "## Health Risk Prediction + SHAP Highlights\n"
            "Toggle between tuned models, adjust the clinical threshold, "
            "and inspect how high-impact lifestyle and wellbeing features steer the prediction."
        )

        model_dropdown = gr.Dropdown(
            choices=[(wrapper.display, key) for key, wrapper in wrappers.items()],
            value=default_model,
            label="Model",
        )

        threshold_slider = gr.Slider(
            minimum=0.2,
            maximum=0.85,
            step=0.01,
            value=default_threshold,
            label="Decision Threshold",
        )

        sliders: List[gr.Slider] = []
        with gr.Row():
            for feature in DISPLAY_FEATURES:
                stats = display_stats[feature]
                sliders.append(
                    gr.Slider(
                        minimum=stats["min"],
                        maximum=stats["max"],
                        value=stats["value"],
                        step=0.01,
                        label=FEATURE_LABELS.get(feature, feature),
                    )
                )

        predict_btn = gr.Button("Predict", variant="primary")

        prediction_label = gr.Label(label="Predicted Class")
        metrics_output = gr.JSON(label="Prediction Details")
        shap_table = gr.Dataframe(
            headers=["Feature", "Contribution"],
            label="Top SHAP Contributions",
            wrap=True,
        )

        predict_btn.click(
            fn=lambda model_key, threshold, *feature_vals: infer(
                model_key,
                threshold,
                *feature_vals,
                wrappers=wrappers,
                scaler=scaler,
                feature_columns=feature_columns,
                template_values=template_values,
            ),
            inputs=[model_dropdown, threshold_slider, *sliders],
            outputs=[prediction_label, metrics_output, shap_table],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    share_flag = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME")
    port_env = os.getenv("GRADIO_SERVER_PORT")
    try:
        server_port = int(port_env) if port_env else None
    except (TypeError, ValueError):
        server_port = None

    launch_kwargs = {"share": share_flag}
    if server_name:
        launch_kwargs["server_name"] = server_name
    if server_port:
        launch_kwargs["server_port"] = server_port

    app.launch(**launch_kwargs)
