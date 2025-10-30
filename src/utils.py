"""Shared utility helpers for the Health XAI project."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ColumnStats:
    """Capture lightweight statistics for a dataset column."""

    name: str
    unique_values: set[str] = field(default_factory=set)
    numeric: bool = True
    example: Optional[str] = None

    def update(self, raw_value: str) -> None:
        value = raw_value.strip()
        if value in {"", "NA", "nan"}:
            return

        normalised = _normalise_value(value)

        if self.example is None:
            self.example = normalised

        if self.numeric:
            try:
                float(value)
            except ValueError:
                self.numeric = False

        self.unique_values.add(normalised)

    def unique_count(self) -> int:
        return len(self.unique_values)


@dataclass
class FeatureInfo:
    """Container for feature metadata used in the data dictionary."""

    name: str
    description: str
    feature_type: str
    example: str
    categorical_values: Optional[str] = None
    status: str = "documented"


def _normalise_value(value: str) -> str:
    """Convert numeric-looking strings to a compact representation."""
    try:
        float_value = float(value)
    except ValueError:
        return value

    if float_value.is_integer():
        return str(int(float_value))
    return f"{float_value:.3f}".rstrip("0").rstrip(".")


def _load_dataset_stats(data_path: Path) -> Tuple[List[str], Dict[str, ColumnStats]]:
    """Scan the processed dataset and collect column statistics."""
    with data_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dataset appears to be empty or malformed.")

        feature_names = [name for name in reader.fieldnames if name]
        stats = {name: ColumnStats(name=name) for name in feature_names}

        for row in reader:
            for name in feature_names:
                stats[name].update(row.get(name, ""))

    return feature_names, stats


def _load_feature_mapping() -> Dict[str, str]:
    """Load feature descriptions from the mapping file."""
    mapping_path = PROJECT_ROOT / "data" / "processed" / "feature_names.csv"
    descriptions: Dict[str, str] = {}

    if not mapping_path.exists():
        return descriptions

    with mapping_path.open(encoding="utf-8") as handle:
        data_lines = [line for line in handle if not line.startswith("#")]
        if not data_lines:
            return descriptions

        reader = csv.DictReader(data_lines)
        for row in reader:
            cleaned = row.get("cleaned_name", "").strip()
            original = row.get("original_name", "").strip()
            description = row.get("description", "").strip()
            if cleaned:
                descriptions[cleaned] = description
            if original and original not in descriptions:
                descriptions[original] = description

    return descriptions


def _parse_documented_features(markdown_text: str) -> List[str]:
    """Extract feature names already present in the markdown document."""
    documented_features = set()
    table_matches = re.findall(r"^\|\s*([A-Za-z0-9_]+)\s*\|", markdown_text, flags=re.MULTILINE)
    heading_matches = re.findall(r"^###\s+([A-Za-z0-9_]+)", markdown_text, flags=re.MULTILINE)
    bullet_matches = re.findall(r"-\s+\*\*([A-Za-z0-9_]+)\*\*", markdown_text)

    documented_features.update(table_matches)
    documented_features.update(heading_matches)
    documented_features.update(bullet_matches)

    return sorted(documented_features)


def _extract_value_mapping(description: str) -> Dict[str, str]:
    """Attempt to extract categorical value mappings from a feature description."""
    if not description:
        return {}

    mapping: Dict[str, str] = {}
    match = re.search(r"\(([^()]+)\)", description)
    if not match:
        return mapping

    content = match.group(1)
    parts = re.split(r"[;,]\s*", content)
    for part in parts:
        trimmed = part.strip()
        if not trimmed or " to " in trimmed:
            continue
        if "=" in trimmed:
            key, value = [token.strip() for token in trimmed.split("=", 1)]
        elif " " in trimmed:
            key, value = trimmed.split(" ", 1)
            key = key.strip()
            value = value.strip()
        else:
            continue

        if key and value:
            mapping[key] = value

    return mapping


def _split_feature_name(feature: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Return the pipeline prefix, base feature name, and category value if present."""
    if "__" not in feature:
        return None, feature, None

    prefix, remainder = feature.split("__", 1)
    category = None

    if prefix == "categorical" and "_" in remainder:
        base, category = remainder.split("_", 1)
    else:
        base = remainder

    return prefix, base, category


def _resolve_description(
    feature: str,
    descriptions: Dict[str, str],
    prefix: Optional[str],
    base_name: str,
    category: Optional[str],
) -> str:
    """Resolve the most appropriate description for a feature."""
    description = descriptions.get(feature, "").strip()
    if description:
        return description

    base_description = descriptions.get(base_name, "").strip()

    if prefix == "numeric":
        if base_description:
            base_without_period = base_description.rstrip(".")
            return f"{base_without_period} (scaled numeric feature)."
        return "*[Description pending clarification]*"

    if prefix == "categorical":
        if category:
            if base_description:
                base_without_period = base_description.rstrip(".")
                return f"{base_without_period} (one-hot encoded: {category})."
            return f"{base_name} == {category} (one-hot encoded)."
        if base_description:
            return f"{base_description} (one-hot encoded)."
        return "*[Description pending clarification]*"

    return base_description if base_description else "*[Description pending clarification]*"


def _infer_feature_type(stats: ColumnStats, prefix: Optional[str]) -> str:
    """Infer a human-readable feature type from column statistics."""
    unique_count = stats.unique_count()

    if prefix == "categorical":
        if unique_count <= 2:
            return "Binary"
        return "Categorical"

    if stats.numeric:
        if unique_count <= 2:
            return "Binary"
        if unique_count <= 10:
            return "Ordinal"
        return "Continuous"

    return "Categorical"


def _gather_categorical_note(
    stats: ColumnStats,
    prefix: Optional[str],
    base_name: str,
    category: Optional[str],
    descriptions: Dict[str, str],
) -> Optional[str]:
    """Return a formatted categorical value summary where applicable."""
    if prefix == "categorical" and category:
        return f"0: Not {category}; 1: {category}"

    base_description = descriptions.get(base_name, "")
    mapping = _extract_value_mapping(base_description)
    if mapping:
        return "; ".join(f"{key}: {value}" for key, value in mapping.items())

    unique_values = sorted(stats.unique_values)
    if not stats.numeric or stats.unique_count() <= 10:
        preview = ", ".join(unique_values[:10])
        if stats.unique_count() > 10:
            preview += ", ..."
        return preview

    return None


def ensure_directory(path: Path | str) -> Path:
    """Ensure that the provided directory exists."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_model(model, path: Path | str) -> Path:
    """Persist a trained model (sklearn/xgboost/torch) to disk."""
    path = Path(path)
    ensure_directory(path.parent)

    if hasattr(model, "state_dict"):
        import torch

        torch.save(model.state_dict(), path)
    else:
        joblib.dump(model, path)

    print(f"[INFO] Saved model to {path}")
    return path


def load_model(path: Path | str, model=None):
    """Load a persisted model; optionally load weights into a provided torch module."""
    path = Path(path)
    if model is not None:
        import torch

        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model

    return joblib.load(path)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    model_name: str,
    dataset_name: str,
    save_dir: Path | str,
) -> Optional[Path]:
    """Plot and save a confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    if len(set(y_true)) < 2:
        print(f"[WARN] Confusion matrix skipped for {model_name} ({dataset_name}) due to single class.")
        return None

    cm = confusion_matrix(y_true, y_pred)
    save_dir = ensure_directory(save_dir)
    path = save_dir / f"{model_name}_{dataset_name}_confusion_matrix.png"

    annot = [[f"TN (0→0)\n{cm[0, 0]}", f"FP (0→1)\n{cm[0, 1]}"], [f"FN (1→0)\n{cm[1, 0]}", f"TP (1→1)\n{cm[1, 1]}"]]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False)
    plt.title(f"{model_name} — {dataset_name.title()} Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks([0.5, 1.5], ["Predicted 0 (Negative)", "Predicted 1 (Positive)"])
    plt.yticks([0.5, 1.5], ["Actual 0 (Negative)", "Actual 1 (Positive)"], rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved confusion matrix to {path}")
    return path


def plot_roc_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    model_name: str,
    dataset_name: str,
    save_dir: Path | str,
) -> Optional[Path]:
    """Plot and save a ROC curve."""
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt

    if len(np.unique(list(y_true))) < 2:
        print(f"[WARN] ROC curve skipped for {model_name} ({dataset_name}) due to single class.")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    save_dir = ensure_directory(save_dir)
    path = save_dir / f"{model_name}_{dataset_name}_roc_curve.png"

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} — {dataset_name.title()} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved ROC curve to {path}")
    return path


def plot_precision_recall_curve(
    y_true: Iterable[int],
    y_score: Iterable[float],
    model_name: str,
    dataset_name: str,
    save_dir: Path | str,
) -> Optional[Path]:
    """Plot and save a Precision-Recall curve."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    import matplotlib.pyplot as plt

    if len(np.unique(list(y_true))) < 2:
        print(f"[WARN] Precision-recall curve skipped for {model_name} ({dataset_name}) due to single class.")
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    save_dir = ensure_directory(save_dir)
    path = save_dir / f"{model_name}_{dataset_name}_precision_recall_curve.png"

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.title(f"{model_name} — {dataset_name.title()} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"[INFO] Saved Precision-Recall curve to {path}")
    return path


def _build_index(features: List[FeatureInfo]) -> str:
    """Generate a GitHub-compatible index of features."""
    links = [f"[{info.name}](#{info.name.lower()})" for info in features]
    return " · ".join(links)


def _build_feature_table(features: List[FeatureInfo]) -> List[str]:
    """Create a Markdown table summarising features."""
    lines = [
        "| Feature | Description | Type | Example |",
        "|---------|-------------|------|---------|",
    ]
    for info in features:
        lines.append(
            f"| {info.name} | {info.description} | {info.feature_type} | {info.example} |"
        )
    return lines


def _build_detailed_sections(features: List[FeatureInfo]) -> List[str]:
    """Create per-feature Markdown sections."""
    lines: List[str] = ["## Detailed Feature Notes"]
    for info in features:
        lines.append("")
        lines.append(f"### {info.name}")
        lines.append(f"- **{info.name}** — {info.description}")
        lines.append(f"- **Type:** {info.feature_type}")
        lines.append(f"- **Example:** {info.example}")
        if info.categorical_values:
            lines.append(f"- **Values:** {{{info.categorical_values}}}")
    return lines


def _write_summary(features: List[FeatureInfo]) -> None:
    """Persist the summary CSV and print console overview."""
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "data_dictionary_check.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature_name", "status", "categorical_values"],
        )
        writer.writeheader()
        for info in features:
            writer.writerow(
                {
                    "feature_name": info.name,
                    "status": info.status,
                    "categorical_values": info.categorical_values or "",
                }
            )

    total = len(features)
    documented = sum(1 for info in features if info.status == "documented")
    missing = total - documented

    print(f"✅ Total features: {total}")
    print(f"📘 Documented: {documented}")
    print(f"⚠️ Newly added: {missing}")
    print(f"📁 Summary saved to {summary_path}")


def update_data_dictionary() -> None:
    """Synchronise the markdown data dictionary with the processed dataset."""
    data_path = PROJECT_ROOT / "data" / "processed" / "health_clean.csv"
    dictionary_path = PROJECT_ROOT / "data" / "data_dictionary.md"
    backup_path = PROJECT_ROOT / "data" / "data_dictionary_backup.md"

    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    feature_names, stats = _load_dataset_stats(data_path)
    descriptions = _load_feature_mapping()
    existing_markdown = dictionary_path.read_text(encoding="utf-8") if dictionary_path.exists() else ""
    documented_features = _parse_documented_features(existing_markdown)
    documented_set = set(documented_features)

    features: List[FeatureInfo] = []
    for feature in sorted(feature_names):
        column_stats = stats[feature]
        prefix, base_name, category = _split_feature_name(feature)
        description = _resolve_description(feature, descriptions, prefix, base_name, category)
        feature_type = _infer_feature_type(column_stats, prefix)
        example = column_stats.example or "N/A"
        categorical_note = _gather_categorical_note(column_stats, prefix, base_name, category, descriptions)
        status = "documented" if feature in documented_set else "missing"

        features.append(
            FeatureInfo(
                name=feature,
                description=description,
                feature_type=feature_type,
                example=example,
                categorical_values=categorical_note,
                status=status,
            )
        )

    index_line = _build_index(features)
    table_lines = _build_feature_table(features)
    section_lines = _build_detailed_sections(features)

    overview = (
        "## Overview\n"
        "This document describes the variables available in `data/raw/heart_data.csv` after "
        "preprocessing and feature-name standardisation. Use it as a reference when building models, "
        "generating explanations (LIME/SHAP), and drafting project reports.\n"
    )

    dictionary_lines = [
        "# Data Dictionary — Health_XAI_Project",
        "",
        overview.strip(),
        "",
        "## Index of Features",
        "",
        index_line,
        "",
        "## Feature Table",
        "",
        *table_lines,
        "",
        *section_lines,
        "",
    ]

    dictionary_content = "\n".join(dictionary_lines)

    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    if dictionary_path.exists():
        backup_path.write_text(existing_markdown, encoding="utf-8")
    else:
        backup_path.write_text(dictionary_content, encoding="utf-8")

    dictionary_path.write_text(dictionary_content, encoding="utf-8")
    _write_summary(features)


__all__ = [
    "update_data_dictionary",
    "ensure_directory",
    "save_model",
    "load_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
]
