# preprocessing.py
# ----------------
# This file prepares the raw CSV data for machine learning.
# Performs loading, encoding categories, splitting & scaling.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── Column Definitions ─────────────────────────────────────────────────────────

# Numeric feature inputs model will learn from 
NUMERIC_FEATURES = [
    "Temperature (°C)",
    "Humidity (%)",
    "Rainfall (mm)",
    "Wind Speed (km/h)",
    "num_fires",
    "avg_size_ha",

]

# Categorical (Text) Features - will be converted into numerical values
CATEGORICAL_FEATURES = ["Region", "Season"]

# Metric to be Predicted
TARGET = "dominant_risk"

def load_data(filepath):
    """
    Reads the CSV file and prints a summary to verify dataset has correctly loaded
    """

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.")
    print(f"\nTarget class distribution:\n{df[TARGET].value_counts()}\n")
    return df

def encode_features(df):
    """
    Coverts data into a format that the model can use: 
    1. One-hot encoding: turn text categories into binary columns
    2. Label encoding: turn target labels into integers: 
    """

    # One-hot encoding of text-based features
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Drop year and month features: irrelevant 
    drop_cols = [TARGET, "Year", "Month"]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]

    X = df_encoded[feature_cols].astype(float)

    # Encode target columns as integers 
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Target classes: {list(le.classes_)}\n")

    return X, y, feature_cols, le

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and test sets, then scales the features 
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training Set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def preprocess_pipline(filepath, test_size=0.2, random_state=42):
    """
    Runs the above preprocessing steps returning everything in a dictionary
    """

    df = load_data(filepath)
    X, y, feature_names, label_encoder = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, test_size, random_state)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "scaler": scaler,

    }


