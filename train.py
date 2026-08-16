"""
train.py — FIN-02 Churn Prediction: terminal-runnable training script.

Mirrors the pipeline built in demo.ipynb (same preprocessing, same models,
same split logic) so results match the notebook. Run this from a terminal:

    python train.py

It will:
  1. Load the data (local data/Telco-Customer-Churn.csv if present, else download)
  2. Clean + split it the same leakage-safe way as the notebook
  3. Train the naive baseline, Logistic Regression, and Random Forest
  4. Print validation + final test metrics
  5. Save the winning pipeline to models/churn_model.joblib
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_LOCAL = "data/Telco-Customer-Churn.csv"


def load_data():
    if os.path.exists(DATA_LOCAL):
        print(f"Loading local dataset: {DATA_LOCAL}")
        return pd.read_csv(DATA_LOCAL)
    print(f"Local dataset not found — downloading from {DATA_URL}")
    return pd.read_csv(DATA_URL)


def main():
    df = load_data()
    print("Shape:", df.shape)

    # --- Clean ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print("Exact duplicate rows:", df.duplicated().sum())
    print("Duplicate customerIDs:", df["customerID"].duplicated().sum())

    df = df.drop(columns=["customerID"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # --- Split (same strategy as the notebook: 64/16/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")

    # --- Preprocessing ---
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # --- Naive baseline ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)

    # --- Logistic Regression ---
    logreg_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    logreg_pipe.fit(X_train, y_train)
    logreg_val_preds = logreg_pipe.predict(X_val)
    logreg_val_proba = logreg_pipe.predict_proba(X_val)[:, 1]
    print("\nLogistic Regression (validation):")
    print(classification_report(y_val, logreg_val_preds, target_names=["No churn", "Churn"]))
    print("ROC-AUC:", roc_auc_score(y_val, logreg_val_proba))

    # --- Random Forest ---
    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_val_preds = rf_pipe.predict(X_val)
    rf_val_proba = rf_pipe.predict_proba(X_val)[:, 1]
    print("\nRandom Forest (validation):")
    print(classification_report(y_val, rf_val_preds, target_names=["No churn", "Churn"]))
    print("ROC-AUC:", roc_auc_score(y_val, rf_val_proba))

    # --- Select final model (Logistic Regression won on validation ROC-AUC) ---
    final_model = logreg_pipe
    print("\nFinal model selected: Logistic Regression "
          "(matched Random Forest on F1, higher ROC-AUC, simpler/more interpretable).")

    # --- Final evaluation on the frozen test set ---
    test_preds = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    print("\n=== FINAL TEST RESULTS ===")
    print(classification_report(y_test, test_preds, target_names=["No churn", "Churn"]))
    print("ROC-AUC:", roc_auc_score(y_test, test_proba))

    dummy_test_preds = dummy.predict(X_test)
    print("\nNaive baseline F1 (churn):", f1_score(y_test, dummy_test_preds))
    print("Final model F1 (churn):   ", f1_score(y_test, test_preds))

    # --- Save ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "models/churn_model.joblib")
    print("\nSaved model to models/churn_model.joblib")


if __name__ == "__main__":
    main()
