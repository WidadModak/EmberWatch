# random_forest.py
# ----------------
# Trains and evaluates a Random Forest model for wildfire risk prediction

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import joblib 
from preprocessing import preprocess_pipline


# ── File Paths ─────────────────────────────────────────────────────────
DATA_PATH = "data/merged_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Step 1: Preprocessing ──────────────────────────────────────────────
print("=" * 50)
print("STEP 1 - Preprocessing")
print("=" * 50)

# Same pipeline ensures identical feature space → fair comparison with LR
data = preprocess_pipline(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
le = data["label_encoder"]
feature_names = data["feature_names"]

class_names = list(le.classes_)
print(f"Classes: {class_names}\n")


# ── Step 2: Baseline Model (RF) ────────────────────────────────────────
print("=" * 50)
print("STEP 2 - Baseline Model")
print("=" * 50)

# Depth is capped to prevent overly specific splits, 
# while balanced weights counter class imbalance in training
baseline_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Test Accuracy: {baseline_acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=class_names, zero_division=0))


# ── Step 3: Cross-validation ───────────────────────────────────────────
print("=" * 50)
print("STEP 3 - 5-Fold Cross-Validation")
print("=" * 50)

# RF is stochastic → CV gives a more reliable estimate than a single split
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Scores per fold: {cv_scores.round(4)}\n")
print(f"Mean: {cv_scores.mean():.4f}\n")
print(f"Std: {cv_scores.std():.4f}\n")


# ── Step 4: Hyperparameter Tuning ──────────────────────────────────────
print("=" * 50)
print("STEP 4 - Hyperparameter Tuning")
print("=" * 50)

# Keep grid small → RF is already strong, avoid wasting time brute-forcing
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-macro: {grid_search.best_score_:.4f}\n")

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)
best_acc = accuracy_score(y_test, y_pred_best)

print(f"Tuned Model Test Accuracy: {best_acc:.4f}\n")
print("Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))


# ── Step 5: Confusion Matrix ───────────────────────────────────────────
print("=" * 50)
print("STEP 5 - Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    ax=ax, colorbar=True, cmap="Blues"
)

ax.set_title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_rf_balanced.png", dpi=150)
plt.close()

print(f"Saved: {RESULTS_DIR}/confusion_matrix_rf_balanced.png\n")


# ── Step 6: Feature Importance ─────────────────────────────────────────
print("=" * 50)
print("STEP 6 — Feature Importance")
print("=" * 50)

# Higher importance means the feature is used more often to make effective splits across trees
importances = best_model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

imp_df.to_csv(f"{RESULTS_DIR}/feature_importance_rf_balanced.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=imp_df, y="Feature", x="Importance", ax=ax)
ax.set_title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance_rf_balanced.png", dpi=150)
plt.close()

print(f"Saved: {RESULTS_DIR}/feature_importance_rf_balanced.png\n")


# ── Step 7: Cross-validation Plot ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores)
ax.axhline(cv_scores.mean(), linestyle="--",
        label=f"Mean = {cv_scores.mean():.3f}")

ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy")
ax.legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cv_scores_rf_balanced.png", dpi=150)
plt.close()

print(f"Saved: {RESULTS_DIR}/cv_scores_rf_balanced.png\n")


# ── Step 8: Save Model ─────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# No scaling dependency for trees → but keep structure consistent with LR for predict.py
joblib.dump(
    {"model": best_model, "scaler": data["scaler"], "label_encoder": le},
    "models/random_forest_balanced.pkl"
)

print("Saved model to: models/random_forest_balanced.pkl\n")


# ── Summary ────────────────────────────────────────────────────────────
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Baseline Accuracy  : {baseline_acc:.4f}")
print(f"  Tuned Accuracy     : {best_acc:.4f}")
print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Best Params        : {grid_search.best_params_}")