# src/preprocessing.py
# ---------------------------------------------------------
# Prepare a target-specific wildfire risk dataset for
# Random Forest modeling.
#
# Input:
#   One CSV from data/variants/
#
# Assumption:
#   The file contains region-month level aggregated data
#   plus exactly one risk target column derived from one
#   wildfire metric.
# ---------------------------------------------------------

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLUMNS = ["Region", "Season"]

# Map each target label to the source metric that created it.
# That source metric must be removed from predictors to avoid leakage.
TARGET_SOURCE_METRIC = {
    "risk_total_burned_area": "total_burned_area",
    "risk_fire_count": "fire_count",
    "risk_average_fire_size": "average_fire_size",
    "risk_worst_case_fire_size": "worst_case_fire_size",
}


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    print(f"Loaded dataset: {filepath}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


def infer_target_column(df: pd.DataFrame, filepath: str) -> str:
    target_col = Path(filepath).stem

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    return target_col


def build_feature_matrix(df: pd.DataFrame, target_col: str):
    # Drop all risk columns except the chosen target, to avoid leakage
    other_risk_columns = [
        col for col in df.columns
        if col.startswith("risk_") and col != target_col
    ]

    # Drop the target-generating source metric too
    source_metric = TARGET_SOURCE_METRIC.get(target_col)
    if source_metric is None:
        raise ValueError(
            f"No source metric mapping found for target '{target_col}'. "
            "Update TARGET_SOURCE_METRIC in preprocessing.py."
        )

    drop_columns = [target_col, source_metric] + other_risk_columns

    X_df = df.drop(columns=drop_columns, errors="ignore").copy()
    y_raw = df[target_col].copy()

    # One-hot encode only columns that are actually present
    categorical_present = [col for col in CATEGORICAL_COLUMNS if col in X_df.columns]

    X_df = pd.get_dummies(
        X_df,
        columns=categorical_present,
        drop_first=False
    )

    X_df = X_df.astype(float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    feature_names = list(X_df.columns)

    print(f"\nTarget column: {target_col}")
    print(f"Source metric removed from predictors: {source_metric}")

    print("\nTarget distribution:")
    print(y_raw.value_counts(dropna=False))

    print(f"\nNumber of features used: {len(feature_names)}")
    print("Features used:")
    print(feature_names)

    print("\nTarget classes:")
    print(list(label_encoder.classes_))

    return X_df, y, feature_names, label_encoder


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\nTraining shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath: str, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    df = load_dataset(filepath)
    target_col = infer_target_column(df, filepath)

    X, y, feature_names, label_encoder = build_feature_matrix(df, target_col)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    return {
        "df": df,
        "target_col": target_col,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
    }