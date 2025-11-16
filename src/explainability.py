"""Explainability helpers for the health-xai-prediction project.

Provides small, well-tested functions to:
- load tuned models and scaler from disk
- provide predict_proba-style wrappers for use with LIME/SHAP
- build LIME and SHAP explainers
- generate and save explanation visualisations

The functions are intentionally lightweight so they can be imported
by notebooks and the Gradio demo.

Notes
-----
This module assumes model artefacts live under `results/models/` and
that processed data lives under `data/processed/` as used across the
project notebooks.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
import logging

import joblib
import numpy as np
import pandas as pd

try:
	import torch
except Exception:  # pragma: no cover - optional runtime
	torch = None

try:
	import lime.lime_tabular as lime_tabular
except Exception:  # pragma: no cover - optional runtime
	lime_tabular = None

try:
	import shap
except Exception:  # pragma: no cover - optional runtime
	shap = None

from matplotlib import pyplot as plt

# Import HealthNN for proper model loading
try:
	from models.neural_network import HealthNN
except ImportError:
	try:
		from src.models.neural_network import HealthNN
	except ImportError:
		HealthNN = None

LOG = logging.getLogger(__name__)


def _resolve_model_path(candidates: Iterable[str]) -> Optional[str]:
	for p in candidates:
		if os.path.exists(p):
			return p
	return None


def load_models(model_dir: str = "results/models") -> Dict[str, Any]:
	"""Load tuned models and scaler from disk.

	The function is permissive about filenames and will try a few
	commonly used names from the project. Returns a dict with keys:
	'xgboost', 'random_forest', 'neural_network', 'scaler' (where
	available).
	"""
	models: Dict[str, Any] = {}

	# XGBoost
	xgb_candidates = [
		os.path.join(model_dir, "xgboost_tuned.joblib"),
		os.path.join(model_dir, "xgboost.joblib"),
		os.path.join(model_dir, "xgboost_classifier.joblib"),
		os.path.join(model_dir, "xgboost_tuned.pkl"),
	]
	xgb_path = _resolve_model_path(xgb_candidates)
	if xgb_path:
		models["xgboost"] = joblib.load(xgb_path)
		LOG.info("Loaded XGBoost model from %s", xgb_path)

	# Random Forest
	rf_candidates = [
		os.path.join(model_dir, "random_forest_tuned.joblib"),
		os.path.join(model_dir, "random_forest.joblib"),
		os.path.join(model_dir, "random_forest.pkl"),
	]
	rf_path = _resolve_model_path(rf_candidates)
	if rf_path:
		models["random_forest"] = joblib.load(rf_path)
		LOG.info("Loaded RandomForest model from %s", rf_path)

	# Neural network (PyTorch)
	nn_candidates = [
		os.path.join(model_dir, "neural_network_tuned.pt"),
		os.path.join(model_dir, "neural_network.pt"),
		os.path.join(model_dir, "best_model.pt"),
	]
	nn_path = _resolve_model_path(nn_candidates)
	if nn_path and torch is not None and HealthNN is not None:
		# Load checkpoint - typically a bare state dict from HealthNN
		checkpoint = torch.load(nn_path, map_location="cpu")
		
		# Infer input_dim from first layer weights
		if isinstance(checkpoint, dict):
			first_layer_key = next((k for k in checkpoint.keys() if "layers.0" in k and "weight" in k), None)
			if first_layer_key:
				input_dim = checkpoint[first_layer_key].shape[1]
				# Infer hidden_dim (usually 128)
				hidden_dim = checkpoint[first_layer_key].shape[0]
				# Instantiate model and load state
				model = HealthNN(input_dim=input_dim, hidden_dim=hidden_dim)
				model.load_state_dict(checkpoint)
				models["neural_network"] = model
				LOG.info("Loaded Neural Network model from %s (input_dim=%d, hidden_dim=%d)", nn_path, input_dim, hidden_dim)
			else:
				LOG.warning("Could not infer input_dim from checkpoint at %s", nn_path)
		else:
			LOG.warning("Unexpected checkpoint format at %s: %s", nn_path, type(checkpoint))

	# Scaler
	scaler_candidates = [
		os.path.join(model_dir, "standard_scaler.joblib"),
		os.path.join(model_dir, "scaler.joblib"),
	]
	scaler_path = _resolve_model_path(scaler_candidates)
	if scaler_path:
		models["scaler"] = joblib.load(scaler_path)
		LOG.info("Loaded scaler from %s", scaler_path)

	return models


def get_predict_fn(model: Any, model_type: str = "sklearn") -> Callable[[np.ndarray], np.ndarray]:
	"""Return a predict_proba-style function for LIME/SHAP.

	Args:
		model: Loaded model object (sklearn-like or torch.nn.Module)
		model_type: 'sklearn'|'xgboost'|'pytorch'

	Returns:
		A function f(X: np.ndarray) -> np.ndarray with shape (n_samples, 2)
		that returns probabilities for [class_0, class_1].
	"""

	if model_type in ("sklearn", "xgboost"):
		def _sklearn_predict(X: np.ndarray) -> np.ndarray:
			return model.predict_proba(X)

		return _sklearn_predict

	if model_type == "pytorch":
		if torch is None:
			raise RuntimeError("PyTorch is not available in the runtime")

		def _torch_predict(X: np.ndarray) -> np.ndarray:
			model.eval()
			with torch.no_grad():
				tensor = torch.FloatTensor(X)
				out = model(tensor)
				# handle logits / single-output
				probs = torch.sigmoid(out).cpu().numpy()
				# ensure shape (n,)
				probs = probs.reshape(-1)
				return np.vstack([1 - probs, probs]).T

		return _torch_predict

	raise ValueError(f"Unknown model_type: {model_type}")


def build_lime_explainer(X_train: pd.DataFrame, feature_names: List[str], class_names: Optional[List[str]] = None):
	"""Build a LIME Tabular explainer.

	Returns the explainer instance. LIME is optional; the function will
	raise a helpful error if the package is missing.
	"""
	if lime_tabular is None:
		raise RuntimeError("LIME is not installed in the environment")

	if class_names is None:
		class_names = ["No Heart Condition", "Heart Condition"]

	explainer = lime_tabular.LimeTabularExplainer(
		training_data=X_train.values,
		feature_names=feature_names,
		class_names=class_names,
		mode="classification",
	)
	return explainer


def build_shap_explainers(models: Dict[str, Any], background: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
	"""Construct SHAP explainers for available models.

	Uses TreeExplainer for tree-based models and KernelExplainer for
	arbitrary models (e.g., neural networks) if `shap` is available.
	"""
	if shap is None:
		raise RuntimeError("SHAP is not installed in the environment")

	explainers: Dict[str, Any] = {}

	# background sample for KernelExplainer
	if background is None:
		background = shap.sample(pd.DataFrame(np.zeros((1, 1))), 1)

	if "xgboost" in models:
		explainers["xgboost"] = shap.TreeExplainer(models["xgboost"])

	if "random_forest" in models:
		explainers["random_forest"] = shap.TreeExplainer(models["random_forest"])

	if "neural_network" in models:
		predict_fn = get_predict_fn(models["neural_network"], model_type="pytorch")
		# KernelExplainer expects a function returning probability for class 1
		def _kernel_fn(x):
			proba = predict_fn(x)
			return proba[:, 1]

		explainers["neural_network"] = shap.KernelExplainer(_kernel_fn, background.values)

	return explainers


def explain_instance_lime(
	explainer, predict_fn: Callable[[np.ndarray], np.ndarray],
	instance: pd.Series, num_features: int = 10
) -> Any:
	"""Explain a single instance with LIME and return the explanation object."""
	exp = explainer.explain_instance(instance.values, predict_fn, num_features=num_features)
	return exp


def explain_instance_shap(explainer, instance: pd.DataFrame) -> np.ndarray:
	"""Compute shap values for a single instance.

	Returns the raw shap values array (for the positive class when
	TreeExplainer returns a list).
	"""
	shap_values = explainer.shap_values(instance)
	if isinstance(shap_values, list):
		# common for binary classification tree explainers
		return shap_values[1]
	return shap_values


def save_figure(fig: plt.Figure, path: str, dpi: int = 300) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches="tight", dpi=dpi)


def apply_threshold(probas: np.ndarray, thresh: float = 0.5) -> np.ndarray:
	"""Map probability array (n,2) to binary labels using threshold on
	class 1 probability.
	"""
	return (probas[:, 1] >= thresh).astype(int)


__all__ = [
	"load_models",
	"get_predict_fn",
	"build_lime_explainer",
	"build_shap_explainers",
	"explain_instance_lime",
	"explain_instance_shap",
	"save_figure",
	"apply_threshold",
]
