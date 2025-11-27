"""Data ingestion and preprocessing pipeline utilities (EDA handled in notebooks)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TARGET_COLUMN = "hltprhc"
RANDOM_STATE = 42
FEATURE_MAP_PATH = PROCESSED_DIR / "feature_names.csv"
FEATURE_ABBREVIATIONS = {
    "hltprhc": "Heart condition",
    "hltprhb": "Blood pressure",
    "hltprdi": "Diabetes",
}

FEATURE_DESCRIPTIONS = {
    "cntry": "Country code of respondent (ISO-2).",
    "happy": "Self-rated happiness on a 0–10 scale.",
    "sclmeet": "Frequency of social meetings with friends, relatives, or colleagues.",
    "inprdsc": "Frequency of participation in organised social, religious, or community activities.",
    "health": "Self-rated general health (1 very good to 5 very bad).",
    "ctrlife": "Feeling of control over life (0 no control to 10 complete control).",
    "etfruit": "Frequency of fruit consumption.",
    "eatveg": "Frequency of vegetable consumption.",
    "dosprt": "Frequency of doing sports or physical exercise.",
    "cgtsmok": "Cigarette smoking status or frequency.",
    "alcfreq": "Alcohol consumption frequency.",
    "height": "Self-reported height in centimeters.",
    "weighta": "Self-reported weight in kilograms.",
    "bmi": "Body mass index derived from height and weight.",
    "fltdpr": "How often felt depressed in the last week.",
    "flteeff": "How often felt everything was an effort in the last week.",
    "slprl": "How often sleep was restless in the last week.",
    "wrhpp": "How often felt happy in the last week (reverse coded).",
    "fltlnl": "How often felt lonely in the last week.",
    "enjlf": "How often enjoyed life in the last week (reverse coded).",
    "fltsd": "How often felt sad in the last week.",
    "hltprhc": "Doctor diagnosed heart or circulation problems (1 yes 0 no).",
    "hltprhb": "Doctor diagnosed high blood pressure (1 yes 0 no).",
    "hltprdi": "Doctor diagnosed diabetes (1 yes 0 no).",
    "gndr": "Gender of respondent (1 male 2 female).",
    "paccnois": "Perceived noise problems in the local area (1 yes 0 no).",
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
    """Persist original-to-cleaned feature name mapping including descriptions."""
    ensure_directories([PROCESSED_DIR])
    descriptions = [FEATURE_DESCRIPTIONS.get(name, "") for name in cleaned]
    mapping_df = pd.DataFrame(
        {"original_name": original, "cleaned_name": cleaned, "description": descriptions}
    )

    with FEATURE_MAP_PATH.open("w", encoding="utf-8") as handle:
        if FEATURE_ABBREVIATIONS:
            handle.write("# Abbreviated feature explanations\n")
            for key, value in FEATURE_ABBREVIATIONS.items():
                handle.write(f"# {key} = {value}\n")
        mapping_df.to_csv(handle, index=False)


def load_dataset(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset and perform basic cleanup."""
    print(f"[INFO] Loading dataset from {data_path.resolve()}")
    df = pd.read_csv(data_path, na_values=["NA", ""])

    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed") or not col.strip()]
    if unnamed_cols:
        print(f"[INFO] Dropping auxiliary columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    df.columns = df.columns.str.strip()
    original_columns = df.columns.tolist()
    cleaned_columns = [clean_column_name(name) for name in original_columns]
    df.columns = cleaned_columns
    cleaned_to_original = dict(zip(cleaned_columns, original_columns))

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    df = df.dropna(subset=[TARGET_COLUMN])

    for column in df.columns:
        if column == "cntry":
            continue
        if df[column].dtype == object:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if {"height", "weighta"}.issubset(df.columns):
        height_m = (df["height"] / 100.0).where(lambda s: s > 0, np.nan)
        bmi = df["weighta"] / np.square(height_m)
        bmi = bmi.replace([np.inf, -np.inf], np.nan)
        df["bmi"] = bmi
        df = df.drop(columns=["height", "weighta"])
        cleaned_to_original.pop("height", None)
        cleaned_to_original.pop("weighta", None)
        cleaned_to_original["bmi"] = "BMI (derived)"
        print("[INFO] Added BMI feature and removed raw height/weight columns.")

    if "cntry" in df.columns:
        df = df.drop(columns=["cntry"])
        cleaned_to_original.pop("cntry", None)
        print("[INFO] Dropped 'cntry' feature (no categorical features remain).")

    final_columns = df.columns.tolist()
    final_originals = [cleaned_to_original.get(col, col) for col in final_columns]
    save_feature_name_mapping(final_originals, final_columns)

    print(f"[INFO] Loaded dataframe with shape {df.shape}")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Impute missing values using median for numeric and mode for categorical features."""
    df = df.copy()

    for column in numeric_features:
        median_value = df[column].median()
        if pd.isna(median_value):
            continue
        df[column] = df[column].fillna(median_value)

    for column in categorical_features:
        mode_series = df[column].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "unknown"
        df[column] = df[column].fillna(fill_value)

    return df


def cap_outliers_iqr(
    df: pd.DataFrame,
    numeric_features: List[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """Apply IQR-based capping to reduce the influence of extreme outliers."""
    df = df.copy()
    for column in numeric_features:
        series = df[column].dropna()
        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        df[column] = df[column].clip(lower=lower, upper=upper)

    return df


def get_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature lists excluding the target."""
    features = [column for column in df.columns if column != TARGET_COLUMN]
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [
        column for column in features if column not in numeric_features
    ]

    print(f"[INFO] Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"[INFO] Categorical features ({len(categorical_features)}): {categorical_features}")
    return numeric_features, categorical_features


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
    return PROCESSED_DIR / "health_clean.csv"


def run_preprocessing_pipeline() -> None:
    """Execute the complete EDA and preprocessing steps."""
    ensure_directories([PROCESSED_DIR])

    df = load_dataset()
    numeric_features, categorical_features = get_feature_groups(df)
    df = handle_missing_values(df, numeric_features, categorical_features)
    df = cap_outliers_iqr(df, numeric_features)

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

    processed_path = save_processed_datasets(
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
    processed_df = pd.read_csv(processed_path)
    print(f"[INFO] Loaded processed dataframe with shape {processed_df.shape}")
    get_feature_groups(processed_df)
    print("[INFO] Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()

# Run the script:  python -m src.data_preprocessing
