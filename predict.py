# predict.py
# ----------------
# Loads saved model and predicts wildfire risk for a new set of conditions

import numpy as np
import pandas as pd
import joblib 

CATEGORICAL_FEATURES = ["Region", "Season"]

def predict_risk(input_dict):
    """
    Takes a dictionary of environmental conditions and returns a risk prediction.
    """
    # Load saved model, scaler, and label encoder
    bundle = joblib.load("models/logistic_regression.pk1")
    model = bundle["model"]
    scaler = bundle["scaler"]
    le = bundle["label encoder"]

    df = pd.DataFrame([input_dict])
    df_enc = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False)

    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[expected_cols].astype(float)

    # Scale and predict
    X_scaled = scaler.transform(df_enc)
    pred_idx   = model.predict(X_scaled)[0]
    pred_class = le.classes_[pred_idx]

    # predict_proba gives a probability for each class — useful for reporting confidence
    proba      = model.predict_proba(X_scaled)[0]
    proba_dict = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {"predicted_class": pred_class, "probabilities": proba_dict}

# ── Testing Model ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Edit these values to test different conditions
    sample = {
        "Temperature (°C)": 34.0,
        "Humidity (%)": 15.0,
        "Rainfall (mm)": 0.0,
        "Wind Speed (km/h)": 30.0,
        "num_fires": 80,
        "avg_size_ha": 200.0,
        "Region": "Kamloops",
        "Season": "Fall",
    }

    result = predict_risk(sample)

    print("\n── Prediction ───────────────────────────────")
    print(f"Input:           {sample}")
    print(f"Predicted risk:  {result['predicted_class']}")
    print("\nClass probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.4f}  {bar}")
    