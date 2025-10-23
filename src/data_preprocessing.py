"""Data ingestion, exploratory analysis, and preprocessing pipeline utilities."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

plt.switch_backend("agg")
warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TARGET_COLUMN = "hltprhc"
RANDOM_STATE = 42
FEATURE_MAP_PATH = PROCESSED_DIR / "feature_names.csv"
FEATURE_ABBREVIATIONS = {
    "hltprhc": "Heart condition",
    "hltprhb": "Blood pressure",
    "hltprdi": "Diabetes",
}


def ensure_directories(paths: Iterable[Path]) -> None:
    """Ensure that every directory in *paths* exists."""
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)


def clean_column_name(name: str) -> str:
    """Standardise a column name by lowercasing and replacing separators."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^\w]+", "_", cleaned)
    cleaned = re.sub(r"_{2,}", "_", cleaned)
    return cleaned.strip("_")


def save_feature_name_mapping(original: List[str], cleaned: List[str]) -> None:
    """Persist original-to-cleaned feature name mapping."""
    ensure_directories([PROCESSED_DIR])
    mapping_df = pd.DataFrame({"original_name": original, "cleaned_name": cleaned})

    with FEATURE_MAP_PATH.open("w", encoding="utf-8") as handle:
        if FEATURE_ABBREVIATIONS:
            handle.write("# Abbreviated feature explanations\n")
            for key, value in FEATURE_ABBREVIATIONS.items():
                handle.write(f"# {key} = {value}\n")
        mapping_df.to_csv(handle, index=False)


def load_dataset(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset and perform basic cleanup."""
    print(f"[INFO] Loading dataset from {data_path.resolve()}")
    df = pd.read_csv(data_path)

    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        print(f"[INFO] Dropping auxiliary columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    df.columns = df.columns.str.strip()
    original_columns = df.columns.tolist()
    cleaned_columns = [clean_column_name(name) for name in original_columns]
    df.columns = cleaned_columns

    save_feature_name_mapping(original_columns, cleaned_columns)

    df = df.dropna(subset=[TARGET_COLUMN])

    print("[INFO] Feature names after cleaning:")
    for feature in df.columns:
        print(f"        {feature}")

    for column in df.columns:
        if column == TARGET_COLUMN:
            continue
        df[column] = pd.to_numeric(df[column], errors="ignore")

    return df


def summarise_dataset(df: pd.DataFrame) -> None:
    """Print and persist high-level dataset statistics."""
    metrics_dir = RESULTS_METRICS_DIR
    ensure_directories([metrics_dir])

    print("[INFO] Dataset overview")
    print(f"        Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    duplicates = df.duplicated().sum()
    print(f"        Duplicate rows: {duplicates}")

    column_summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str),
            "null_count": df.isna().sum(),
            "null_pct": (df.isna().mean() * 100).round(2),
            "unique_values": df.nunique(dropna=True),
        }
    )
    column_summary.to_csv(metrics_dir / "column_summary.csv", index=False)

    missing_summary = (
        df.isna().sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
        .sort_values(by="missing", ascending=False)
    )
    missing_summary["missing_pct"] = (
        missing_summary["missing"] / len(df) * 100
    ).round(2)
    missing_summary.to_csv(metrics_dir / "missing_summary.csv", index=False)

    class_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    class_counts.to_csv(metrics_dir / "target_distribution.csv")

    describe_numeric = df.select_dtypes(include=[np.number]).describe().T
    describe_numeric.to_csv(metrics_dir / "numeric_descriptive_stats.csv")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        describe_categorical = (
            df[categorical_cols].describe(include=["object", "category"]).T
        )
        describe_categorical.to_csv(metrics_dir / "categorical_descriptive_stats.csv")

    print(f"[INFO] Column summary stored at {metrics_dir / 'column_summary.csv'}")


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Detect potential outliers using the IQR rule for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    outlier_stats: Dict[str, Dict[str, float]] = {}
    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_stats[column] = {
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": round(outlier_mask.mean() * 100, 2),
        }

    outlier_df = pd.DataFrame.from_dict(outlier_stats, orient="index")
    outlier_df.index.name = "feature"
    outlier_df.reset_index(inplace=True)
    outlier_df.to_csv(RESULTS_METRICS_DIR / "outlier_summary.csv", index=False)
    print(
        f"[INFO] Outlier analysis saved to "
        f"{RESULTS_METRICS_DIR / 'outlier_summary.csv'}"
    )
    return outlier_df


def get_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature lists excluding the target."""
    features = [column for column in df.columns if column != TARGET_COLUMN]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in features if column not in numeric_features
    ]

    print(f"[INFO] Numeric features: {len(numeric_features)}")
    print(f"[INFO] Categorical features: {len(categorical_features)}")
    return numeric_features, categorical_features


def calculate_vif(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for numeric features."""
    if len(numeric_features) < 2:
        print("[WARN] VIF requires at least two numeric features; skipping.")
        return pd.DataFrame()

    imputer = SimpleImputer(strategy="median")
    numeric_data = pd.DataFrame(
        imputer.fit_transform(df[numeric_features]), columns=numeric_features
    )

    vif_data = []
    for idx, column in enumerate(numeric_features):
        y = numeric_data[column]
        X = numeric_data.drop(columns=[column])

        if X.shape[1] == 0:
            vif = np.nan
        else:
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            vif = np.inf if r_squared >= 1 else 1.0 / max(1 - r_squared, 1e-6)
        vif_data.append({"feature": column, "vif": round(float(vif), 3)})

    vif_df = pd.DataFrame(vif_data)
    vif_df.to_csv(RESULTS_METRICS_DIR / "vif_summary.csv", index=False)
    print(f"[INFO] VIF summary saved to {RESULTS_METRICS_DIR / 'vif_summary.csv'}")
    return vif_df


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create a correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        print("[WARN] Not enough numeric features for correlation heatmap.")
        return

    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, square=True)
    plt.title("Correlation Heatmap")
    plot_path = RESULTS_PLOTS_DIR / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {plot_path}")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Generate histogram and boxplot for each numeric feature."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("[WARN] No numeric features found for distribution plots.")
        return

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        if series.empty:
            continue

        plt.figure(figsize=(8, 5))
        sns.histplot(series, kde=True, bins=30)
        plt.title(f"Histogram - {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        hist_path = RESULTS_PLOTS_DIR / f"{column}_hist.png"
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=series)
        plt.title(f"Boxplot - {column}")
        plt.xlabel(column)
        box_path = RESULTS_PLOTS_DIR / f"{column}_box.png"
        plt.tight_layout()
        plt.savefig(box_path)
        plt.close()

    print("[INFO] Numeric distribution plots saved.")


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Plot bar charts for categorical feature frequency distributions."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) == 0:
        print("[INFO] No categorical features requiring distribution plots.")
        return

    for column in categorical_cols:
        series = df[column].astype(str)
        plt.figure(figsize=(10, 6))
        counts = series.value_counts().head(20)
        sns.barplot(x=counts.values, y=counts.index, palette="viridis")
        plt.title(f"Top Categories - {column}")
        plt.xlabel("Frequency")
        plt.ylabel(column)
        plot_path = RESULTS_PLOTS_DIR / f"{column}_bar.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print("[INFO] Categorical distribution plots saved.")


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def transform_to_dataframe(
    transformer: ColumnTransformer, features: pd.DataFrame
) -> pd.DataFrame:
    """Apply the fitted transformer and return a DataFrame with feature names."""
    transformed = transformer.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = transformer.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=features.index)


def save_processed_datasets(
    transformer: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    full_features: pd.DataFrame,
    full_target: pd.Series,
) -> None:
    """Persist processed datasets to disk."""
    print("[INFO] Saving processed datasets to disk.")

    train_processed = transform_to_dataframe(transformer, X_train)
    val_processed = transform_to_dataframe(transformer, X_val)
    test_processed = transform_to_dataframe(transformer, X_test)

    train_processed[TARGET_COLUMN] = y_train.values
    val_processed[TARGET_COLUMN] = y_val.values
    test_processed[TARGET_COLUMN] = y_test.values

    train_processed.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_processed.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_processed.to_csv(PROCESSED_DIR / "test.csv", index=False)

    full_processed = transform_to_dataframe(transformer, full_features)
    full_processed[TARGET_COLUMN] = full_target.values
    full_processed.to_csv(PROCESSED_DIR / "health_clean.csv", index=False)
    print(f"[INFO] Processed dataset saved to {PROCESSED_DIR / 'health_clean.csv'}")


def run_preprocessing_pipeline() -> None:
    """Execute the complete EDA and preprocessing steps."""
    ensure_directories([PROCESSED_DIR, RESULTS_PLOTS_DIR, RESULTS_METRICS_DIR])

    df = load_dataset()
    summarise_dataset(df)
    detect_outliers_iqr(df)
    numeric_features, categorical_features = get_feature_groups(df)
    calculate_vif(df, numeric_features)
    plot_correlation_heatmap(df)
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    features_df = df.drop(columns=[TARGET_COLUMN])
    target_series = df[TARGET_COLUMN].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features_df,
        target_series,
        test_size=0.30,
        stratify=target_series,
        random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)

    save_processed_datasets(
        preprocessor,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        features_df,
        target_series,
    )
    print("[INFO] Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
