from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="data/features.csv")
    ap.add_argument("--out", type=str, default="models/baseline_logreg.joblib")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.features)

    # target
    y = df["conflict"].astype(int)

    # basic feature set (keep it simple + interpretable)
    feature_cols_num = ["h0_nm", "v0_ft"]
    feature_cols_cat = ["encounter_type"]


    X = df[feature_cols_num + feature_cols_cat].copy()

    # preprocessing: numeric -> impute + scale; categorical -> impute + onehot
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, feature_cols_num),
            ("cat", cat_pipe, feature_cols_cat),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
    )

    clf = Pipeline(steps=[
        ("pre", pre),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    clf.fit(X_train, y_train)

    # predicted probabilities for positive class
    p_test = clf.predict_proba(X_test)[:, 1]
    y_hat = (p_test >= 0.5).astype(int)

    auc = roc_auc_score(y_test, p_test)
    ap_score = average_precision_score(y_test, p_test)
    acc = accuracy_score(y_test, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_hat, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_hat)

    print("\n=== Baseline Logistic Regression ===")
    print(f"Rows: {len(df)} | Pos rate: {y.mean():.3f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"PR AUC:  {ap_score:.4f}")
    print(f"Acc:     {acc:.4f}")
    print(f"Prec:    {pr:.4f}")
    print(f"Recall:  {rc:.4f}")
    print(f"F1:      {f1:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(cm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print("\nSaved model to:", out_path)


if __name__ == "__main__":
    main()
