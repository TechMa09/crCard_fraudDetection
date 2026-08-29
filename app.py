"""Streamlit dashboard for the credit card fraud XGBoost model."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ACCENT = "#3b82f6"

st.set_page_config(page_title="Fraud Model Dashboard", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_test_split():
    df = pd.read_csv("credit_card_fraud_10k.csv").drop(columns=["transaction_id"])
    X = pd.get_dummies(df.drop(columns=["is_fraud"]), columns=["merchant_category"])
    y = df["is_fraud"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return df, X_test, y_test


model = load_model()
df, X_test, y_test = load_test_split()
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

st.title("Credit Card Fraud Model Dashboard")
st.caption(
    f"XGBoost classifier evaluated on a held-out test set "
    f"({len(y_test):,} transactions, {y_test.mean():.2%} fraud rate)."
)

# --- headline metrics ---------------------------------------------------
report = classification_report(y_test, y_pred, output_dict=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision (fraud)", f"{report['1']['precision']:.3f}")
col2.metric("Recall (fraud)", f"{report['1']['recall']:.3f}")
col3.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.3f}")
col4.metric("PR-AUC", f"{average_precision_score(y_test, y_proba):.3f}")

st.divider()

# --- confusion matrix + PR curve ----------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"], cmap="Blues", ax=ax, colorbar=False
    )
    st.pyplot(fig)

with right:
    st.subheader("Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    PrecisionRecallDisplay.from_predictions(
        y_test, y_proba, name="XGBoost", ax=ax, curve_kwargs={"color": ACCENT}
    )
    st.pyplot(fig)

st.divider()

# --- feature importance --------------------------------------------------
st.subheader("What Drives a Fraud Prediction")
importances = pd.Series(
    model.feature_importances_, index=X_test.columns
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(importances.index, importances.values, color=ACCENT)
ax.set_xlabel("Importance")
fig.tight_layout()
st.pyplot(fig)

st.divider()

# --- classification report table -----------------------------------------
st.subheader("Classification Report")
report_df = pd.DataFrame(report).T.rename(index={"0": "Legit", "1": "Fraud"})
st.dataframe(report_df.style.format("{:.3f}"), width="stretch")
