"""Train and save five classifiers for the BITS ML Assignment 2.

Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository.
The script creates:
- model/*.joblib: fitted pipelines (preprocessing + classifier)
- test_data.csv: untouched stratified test split, including target
- metrics.csv: six required evaluation metrics for all five models
- model/metadata.json: feature schema and split information

Run from the project root:
    python train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.25
TARGET_COLUMN = "target"
MODEL_DIRECTORY = Path("model")


def safe_feature_name(name: str) -> str:
    """Convert a feature label to a stable CSV-friendly snake_case name."""
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def prediction_score(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return a continuous score for binary ROC-AUC calculation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return model.predict(features)


def calculate_metrics(
    model: Pipeline, features: pd.DataFrame, labels: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = prediction_score(model, features)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, scores),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    MODEL_DIRECTORY.mkdir(exist_ok=True)

    raw = load_breast_cancer(as_frame=True)
    features = raw.data.copy()
    features.columns = [safe_feature_name(c) for c in features.columns]

    # In sklearn's copy, 0=malignant and 1=benign. Recode so that the
    # clinically important malignant class is the positive class.
    labels = (raw.target == 0).astype(int).rename(TARGET_COLUMN)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    numeric_features = list(features.columns)
    scaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    unscaled_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", unscaled_preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("preprocess", unscaled_preprocessor),
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Gaussian Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "support_vector_machine.joblib",
    }

    metric_rows: list[dict[str, float | str]] = []
    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODEL_DIRECTORY / filenames[model_name])
        metric_rows.append(
            {"ML Model Name": model_name, **calculate_metrics(pipeline, x_test, y_test)}
        )

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv("test_data.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv("metrics.csv", index=False, float_format="%.6f")

    metadata = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "repository": "UCI Machine Learning Repository",
        "instances": int(features.shape[0]),
        "features": int(features.shape[1]),
        "feature_names": numeric_features,
        "target_column": TARGET_COLUMN,
        "target_mapping": {"0": "benign", "1": "malignant"},
        "positive_class": "malignant",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "model_files": filenames,
    }
    (MODEL_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Training complete. Test-set results:\n")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(models)} fitted models to: {MODEL_DIRECTORY.resolve()}")
    print(f"Saved test data to: {Path('test_data.csv').resolve()}")


if __name__ == "__main__":
    main()
