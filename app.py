"""
Streamlit Application

Requirements:
1. Upload CSV
2. Select model
3. Load saved model
4. Show metrics
5. Show confusion matrix
"""

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# PAGE TITLE
# ---------------------------------------

st.set_page_config(page_title="Mushroom Classification")

st.title("🍄 Mushroom Classification")

st.write(
    "BITS Assignment - ML Classification Models"
)

# ---------------------------------------
# MODEL OPTIONS
# ---------------------------------------

model_files = {
    "Logistic Regression":
    "logistic.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Naive Bayes":
        "naive_bayes.pkl",

    "SVM":
        "svm.pkl"
}

selected_model = st.selectbox(
    "Select Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# ---------------------------------------
# PREDICTION
# ---------------------------------------

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")

    st.write(data.head())

    model = joblib.load(
        model_files[selected_model]
    )

    y_true = data["class"]

    X = data.drop("class", axis=1)

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    mcc = matthews_corrcoef(y_true, y_pred)

    auc = roc_auc_score(y_true, y_prob)

    # ----------------------------------
    # METRICS
    # ----------------------------------

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        round(accuracy, 4)
    )

    col2.metric(
        "AUC",
        round(auc, 4)
    )

    col3.metric(
        "Precision",
        round(precision, 4)
    )

    col1.metric(
        "Recall",
        round(recall, 4)
    )

    col2.metric(
        "F1 Score",
        round(f1, 4)
    )

    col3.metric(
        "MCC",
        round(mcc, 4)
    )

    # ----------------------------------
    # CONFUSION MATRIX
    # ----------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(5,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # ----------------------------------
    # CLASSIFICATION REPORT
    # ----------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_true,
        y_pred
    )

    st.text(report)
