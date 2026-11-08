"""Interactive evaluation app for five saved breast-cancer classifiers.

The app evaluates previously fitted models on an uploaded test CSV. It never
fits or retrains a model, satisfying the assignment requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"
METADATA_FILE = MODEL_DIR / "metadata.json"
DEFAULT_TEST_FILE = BASE_DIR / "test_data.csv"

st.set_page_config(
    page_title="Diagnostic Model Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;}
      .hero {
        padding: 1.2rem 1.4rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #0b7285 55%, #12b886 100%);
        color: white; margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(16,42,67,.18);
      }
      .hero h1 {margin: 0; font-size: 2.05rem;}
      .hero p {margin: .35rem 0 0 0; opacity: .92;}
      .small-note {font-size: .88rem; color: #52606d;}
      div[data-testid="stMetric"] {
        border: 1px solid #d9e2ec; padding: .65rem .8rem;
        border-radius: 12px; background: #f8fbfd;
      }
    </style>
    <div class="hero">
      <h1>Diagnostic Model Lab</h1>
      <p>Compare five pre-trained classifiers on uploaded Wisconsin diagnostic test data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_project_assets() -> tuple[dict, dict[str, object]]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Saved model metadata is missing. Run `python train_models.py` first."
        )
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    loaded_models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metadata["model_files"].items()
    }
    return metadata, loaded_models


def continuous_scores(model: object, features: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return model.predict(features)


def evaluate(model: object, features: pd.DataFrame, labels: pd.Series):
    predictions = model.predict(features)
    scores = continuous_scores(model, features)
    auc = roc_auc_score(labels, scores) if labels.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": auc,
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }
    return metrics, predictions


def validate_uploaded_data(data: pd.DataFrame, feature_names: list[str], target: str):
    required = feature_names + [target]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    unexpected_target_values = sorted(set(data[target].dropna().unique()) - {0, 1})
    if unexpected_target_values:
        raise ValueError("The target column must contain only 0 (benign) and 1 (malignant).")

    if data[target].isna().any():
        raise ValueError("The target column contains missing values.")

    numeric = data[feature_names].apply(pd.to_numeric, errors="coerce")
    entirely_invalid = [column for column in feature_names if numeric[column].isna().all()]
    if entirely_invalid:
        raise ValueError(
            "These feature columns contain no numeric values: " + ", ".join(entirely_invalid)
        )
    return numeric, data[target].astype(int)


try:
    metadata, models = load_project_assets()
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Evaluation controls")
    st.caption("No model is retrained in this application.")
    uploaded_file = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="The CSV must contain all 30 model features and the target column.",
    )
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded_file is None)
    selected_model = st.selectbox("Select a classifier", list(models.keys()))
    st.divider()
    st.subheader("Target definition")
    st.write("**1 = malignant**")
    st.write("**0 = benign**")
    st.caption(
        f"Expected schema: {metadata['features']} features + "
        f"`{metadata['target_column']}`."
    )

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name
    except Exception as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
elif use_sample:
    if not DEFAULT_TEST_FILE.exists():
        st.error("Bundled test_data.csv was not found.")
        st.stop()
    data = pd.read_csv(DEFAULT_TEST_FILE)
    source_label = DEFAULT_TEST_FILE.name
else:
    st.info("Upload a test CSV or enable the bundled sample data.")
    st.stop()

try:
    x_test, y_test = validate_uploaded_data(
        data, metadata["feature_names"], metadata["target_column"]
    )
except ValueError as error:
    st.error(str(error))
    with st.expander("Required columns"):
        st.code("\n".join(metadata["feature_names"] + [metadata["target_column"]]))
    st.stop()

st.caption(
    f"Data source: `{source_label}` | Rows: **{len(data)}** | "
    f"Positive cases: **{int(y_test.sum())}** | Negative cases: **{int((y_test == 0).sum())}**"
)

single_tab, comparison_tab, data_tab = st.tabs(
    ["Selected model", "All-model comparison", "Uploaded data"]
)

with single_tab:
    selected_metrics, selected_predictions = evaluate(
        models[selected_model], x_test, y_test
    )
    st.subheader(selected_model)
    metric_columns = st.columns(6)
    for column, (metric_name, value) in zip(metric_columns, selected_metrics.items()):
        column.metric(metric_name, "N/A" if pd.isna(value) else f"{value:.3f}")

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(y_test, selected_predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5.2, 4.0))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=axis,
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_test,
            selected_predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        report_frame = pd.DataFrame(report).transpose()
        st.dataframe(report_frame.style.format("{:.3f}"), use_container_width=True)

    prediction_output = data.copy()
    prediction_output["predicted_target"] = selected_predictions
    prediction_output["actual_class"] = y_test.map({0: "benign", 1: "malignant"})
    prediction_output["predicted_class"] = pd.Series(selected_predictions).map(
        {0: "benign", 1: "malignant"}
    ).values
    st.download_button(
        "Download predictions",
        prediction_output.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

with comparison_tab:
    comparison_rows = []
    for model_name, model in models.items():
        metrics, _ = evaluate(model, x_test, y_test)
        comparison_rows.append({"ML Model Name": model_name, **metrics})
    comparison = pd.DataFrame(comparison_rows).set_index("ML Model Name")
    st.subheader("Performance of all saved models")
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#b7f7dd"),
        use_container_width=True,
    )
    st.bar_chart(comparison[["Accuracy", "AUC", "F1", "MCC"]])
    winner = comparison["MCC"].idxmax()
    st.success(f"Highest MCC on this uploaded test data: {winner}")

with data_tab:
    st.subheader("Test-data preview")
    st.dataframe(data.head(25), use_container_width=True)
    st.markdown(
        '<p class="small-note">Only the uploaded/bundled test rows are evaluated. '
        "The fitted pipelines loaded from disk remain unchanged.</p>",
        unsafe_allow_html=True,
    )
