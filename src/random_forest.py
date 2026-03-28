# src/random_forest.py
# ---------------------------------------------------------
# Train and evaluate one tuned Random Forest model
# for a selected wildfire risk target variant.
# ---------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.preprocessing import preprocess_pipeline


# =========================================================
# Configuration
# =========================================================

TARGET_FILE = "data/variants/risk_average_fire_size.csv"
RANDOM_STATE = 42
CV_FOLDS = 5

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(y_true, y_pred, class_names, save_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax,
        colorbar=True,
        cmap="Blues"
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_cv_plot(cv_scores, save_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Fold {i+1}" for i in range(len(cv_scores))], cv_scores)
    ax.axhline(
        cv_scores.mean(),
        linestyle="--",
        label=f"Mean = {cv_scores.mean():.3f}"
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Macro")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_feature_importance(model, feature_names, csv_path: Path, plot_path: Path, title: str) -> None:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    target_path = Path(TARGET_FILE)
    target_name = target_path.stem

    results_dir = Path("results") / "random_forest" / target_name
    models_dir = Path("models") / "random_forest" / target_name

    ensure_dir(results_dir)
    ensure_dir(models_dir)

    print("=" * 60)
    print(f"RANDOM FOREST :: {target_name}")
    print("=" * 60)

    # Preprocessing
    data = preprocess_pipeline(str(target_path))

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_encoder = data["label_encoder"]
    feature_names = data["feature_names"]
    target_col = data["target_col"]

    class_names = list(label_encoder.classes_)

    print("\nTarget column:")
    print(target_col)

    print("\nClasses:")
    print(class_names)

    # Cross-validation strategy
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Hyperparameter tuning
    print("\nRunning hyperparameter tuning...")

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Macro: {grid_search.best_score_:.4f}")

    # CV scores for the tuned model
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    print(f"\nCV F1-Macro Scores: {np.round(cv_scores, 4)}")
    print(f"Mean CV F1-Macro: {cv_scores.mean():.4f}")
    print(f"Std CV F1-Macro: {cv_scores.std():.4f}")

    # Fit final model on full training data
    best_model.fit(X_train, y_train)

    # Test evaluation
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    # Save outputs
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        save_path=results_dir / "confusion_matrix_rf.png",
        title=f"Random Forest — {target_name}"
    )

    save_cv_plot(
        cv_scores=cv_scores,
        save_path=results_dir / "cv_scores_rf.png",
        title=f"5-Fold CV F1 Macro — {target_name}"
    )

    save_feature_importance(
        model=best_model,
        feature_names=feature_names,
        csv_path=results_dir / "feature_importance_rf.csv",
        plot_path=results_dir / "feature_importance_rf.png",
        title=f"Feature Importance — {target_name}"
    )

    # Save metrics summary
    metrics_summary = pd.DataFrame([{
        "target_name": target_name,
        "target_column": target_col,
        "best_params": str(grid_search.best_params_),
        "cv_f1_macro_mean": cv_scores.mean(),
        "cv_f1_macro_std": cv_scores.std(),
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
    }])

    metrics_summary.to_csv(results_dir / "metrics_summary_rf.csv", index=False)

    # Save model artifact
    joblib.dump(
        {
            "model": best_model,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "target_name": target_name,
            "target_column": target_col,
        },
        models_dir / "rf_final.pkl"
    )

    print(f"\nSaved model to: {models_dir / 'rf_final.pkl'}")
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()