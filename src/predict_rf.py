# src/predict_rf.py
# ---------------------------------------------------------
# Load a trained Random Forest model and predict wildfire
# risk for one new observation.
# ---------------------------------------------------------

from pathlib import Path
import pandas as pd
import joblib


CATEGORICAL_FEATURES = ["Region", "Season"]

# Choose which trained target model to use
TARGET_NAME = "risk_average_fire_size.csv"
MODEL_PATH = Path("models") / "random_forest" / TARGET_NAME / "rf_final.pkl"


def prepare_input(input_dict, feature_names):
    """
    Convert raw input dictionary into the exact feature layout
    expected by the saved model.
    """
    df = pd.DataFrame([input_dict])

    categorical_present = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=categorical_present, drop_first=False)

    # Add any missing training columns
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep only training columns, in training order
    df_encoded = df_encoded[feature_names].astype(float)

    return df_encoded


def predict_risk(input_dict):
    """
    Predict wildfire risk class and class probabilities for one input row.
    """
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_names = bundle["feature_names"]
    target_name = bundle["target_name"]

    X_input = prepare_input(input_dict, feature_names)

    pred_idx = model.predict(X_input)[0]
    pred_class = label_encoder.inverse_transform([pred_idx])[0]

    probabilities = model.predict_proba(X_input)[0]
    probability_dict = {
        label_encoder.classes_[i]: round(float(prob), 4)
        for i, prob in enumerate(probabilities)
    }

    return {
        "target_name": target_name,
        "predicted_class": pred_class,
        "probabilities": probability_dict,
    }


if __name__ == "__main__":
    # Example input for testing
    # IMPORTANT:
    # This sample must match the current modeling schema.
    sample = {
        "Region": "Kamloops",
        "Year": 2023,
        "Month": 8,
        "Season": "Summer",
        "total_burned_area": 500.0,
        "fire_count": 12,
        "average_fire_size": 41.7,
        "temperature_c": 28.0,
        "humidity_pct": 22.0,
        "rainfall_mm": 5.0,
        "wind_speed_kmh": 18.0,
    }

    result = predict_risk(sample)

    print("\n--- Prediction -----------------------------------")
    print(f"Target:          {result['target_name']}")
    print(f"Input:           {sample}")
    print(f"Predicted risk:  {result['predicted_class']}")
    print("\nClass probabilities:")

    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.4f}  {bar}")