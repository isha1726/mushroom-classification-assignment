"""
BITS Assignment
Mushroom Classification

This script:
1. Loads dataset
2. Handles missing values
3. Encodes categorical columns
4. Splits data into train/test
5. Trains all 5 models
6. Calculates evaluation metrics
7. Saves models
8. Saves test_data.csv
"""

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

# ----------------------------------
# LOAD DATASET
# ----------------------------------

df = pd.read_csv("mushroom.csv")

print("Dataset Shape:", df.shape)

# ----------------------------------
# HANDLE MISSING VALUES
# ----------------------------------

df.replace("?", "missing", inplace=True)

# ----------------------------------
# ENCODE ALL CATEGORICAL COLUMNS
# ----------------------------------

encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Save encoders
joblib.dump(encoders, "encoder.pkl")

# ----------------------------------
# TARGET COLUMN
# class column:
# e = edible
# p = poisonous
# ----------------------------------

target_col = "class"

X = df.drop(target_col, axis=1)
y = df[target_col]

# ----------------------------------
# TRAIN TEST SPLIT
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------
# SAVE TEST DATA FOR STREAMLIT
# ----------------------------------

test_df = X_test.copy()
test_df["class"] = y_test

test_df.to_csv("test_data.csv", index=False)

# ----------------------------------
# MODELS
# ----------------------------------

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=500),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(probability=True)
}

# ----------------------------------
# MODEL TRAINING
# ----------------------------------

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    mcc = matthews_corrcoef(y_test, y_pred)

    auc = roc_auc_score(y_test, y_prob)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, filename)

    print(f"{name} saved")

# ----------------------------------
# SAVE METRICS
# ----------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

results_df.to_csv(
    "metrics_comparison.csv",
    index=False
)

print("\nRESULTS")
print(results_df)
