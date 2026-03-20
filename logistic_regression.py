# logistic_regression.py
# ----------------
# Trains and evaluates a Logistic Regression model for wildfire risk prediction

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import joblib 

# Import our preprocessing functions from the other file 
from preprocessing import preprocess_pipline

# ── File Paths ─────────────────────────────────────────────────────────
DATA_PATH = "data/merged_dataset.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True) # Create results folder

# ── Step 1: Load and preprocess data ─────────────────────────────────────────────────────────
print("=" * 50)
print("STEP 1 - Preprocessing")
print("=" * 50)

data = preprocess_pipline(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
le = data["label_encoder"]
feature_names = data["feature_names"]

# Get class names that appear in data
class_names = list(le.classes_)
print(f"Classes: {class_names}\n")

# ── Step 2: Train Baseline Model (LR) ─────────────────────────────────────────────────────────    
print("=" * 50)
print("STEP 2 - Baseline Model")
print("=" * 50)

baseline_model = LogisticRegression(
    solver = "lbfgs",
    max_iter=1000,
    random_state=42,
)
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Test Accuracy: {baseline_acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=class_names, zero_division=0))

# ── Step 3: Cross-validation ─────────────────────────────────────────────────────────
print("=" * 50)
print("STEP 3 - 5-Fold Cross-Validation")
print("=" * 50)

cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Scores per fold: {cv_scores.round(4)}\n")
print(f"Mean: {cv_scores.mean():.4f}\n")
print(f"Std: {cv_scores.std():.4f}\n")

# ── Step 4: Hyperparameter tuning with GridSearchCV ─────────────────────────────────────────────────────────
print("=" * 50)
print("STEP 4 - Hyperparameter Tuning")
print("=" * 50)

param_grid = {
    "C":        [0.01, 0.1, 1, 10, 100],   # regularization values to try
    "max_iter": [1000],
}

grid_search = GridSearchCV(
    LogisticRegression(solver="lbfgs", random_state=42),
    param_grid,
    cv=5,
    scoring="f1_macro",   # f1_macro penalizes the model for ignoring minority classes
    n_jobs=-1,             # use all CPU cores to speed it up
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1-macro: {grid_search.best_score_:.4f}\n")

best_model  = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_acc    = accuracy_score(y_test, y_pred_best)

print(f"Tuned Model Test Accuracy: {best_acc:.4f}\n")
print("Classification Report (Tuned Model):")
print(classification_report(y_test, y_pred_best, target_names=class_names, zero_division=0))

# ── Step 5: Confusion Matrix Plot ─────────────────────────────────────────────────────────    
# Diagonal = correct predictions
print("=" * 50)
print("STEP 5 - Confusion Matrix")
print("=" * 50)

cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    ax=ax, colorbar=True, cmap="Blues"
)
ax.set_title("Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_lr.png", dpi=150)
plt.close()
print(f"Saved: {RESULTS_DIR}/confusion_matrix_lr.png\n")

# ── Step 6: Feature Coefficients ─────────────────────────────────────────────────────────
# Positive: feature pushes toward that class
# Negative: feature pushes away from that class
# Larger abs value means a stronger influence
print("=" * 50)
print("STEP 6 — Feature Coefficients")
print("=" * 50)

rows = []
for i, cls in enumerate(class_names):
    for feat, coef in zip(feature_names, best_model.coef_[i]):
        rows.append({"Class": cls, "Feature": feat, "Coefficient": coef})

coef_df = pd.DataFrame(rows)
coef_df.to_csv(f"{RESULTS_DIR}/feature_coefficients_lr.csv", index=False)

# Plot coefficients for each class
fig, axes = plt.subplots(1, len(class_names), figsize=(5 * len(class_names), 5), sharey=True)
if len(class_names) == 1:
    axes = [axes]

for ax, cls in zip(axes, class_names):
    sub = coef_df[coef_df["Class"] == cls].sort_values("Coefficient")
    colors = ["#d73027" if v > 0 else "#4575b4" for v in sub["Coefficient"]]
    ax.barh(sub["Feature"], sub["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Class: {cls}")
    ax.set_xlabel("Coefficient")

plt.suptitle("Feature Coefficients by Risk Class", y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_coefficients_lr.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {RESULTS_DIR}/feature_coefficients_lr.png\n")

# ── Step 7: Cross-validation score plot ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores, color="#4575b4")
ax.axhline(cv_scores.mean(), color="#d73027", linestyle="--",
           label=f"Mean = {cv_scores.mean():.3f}")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross-Validation Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cv_scores_lr.png", dpi=150)
plt.close()
print(f"Saved: {RESULTS_DIR}/cv_scores_lr.png\n")

# ── Step 8: Save Model ─────────────────────────────────────────────────────────   
# joblib saves the trained model to disk so predict.py can load it later
os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": best_model, "scaler": data["scaler"], "label_encoder": le},
    "models/logistic_regression.pkl"
)
print("Saved model to: models/logistic_regression.pkl\n")


# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"  Baseline Accuracy  : {baseline_acc:.4f}")
print(f"  Tuned Accuracy     : {best_acc:.4f}")
print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Best Params        : {grid_search.best_params_}")