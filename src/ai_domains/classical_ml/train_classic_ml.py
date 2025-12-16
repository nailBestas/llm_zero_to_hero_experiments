from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_data(test_size: float = 0.2, random_state: int = 42):
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate(model: LogisticRegression, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("=== Iris - Logistic Regression ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, target_names = load_data(
        test_size=args.test_size, random_state=args.seed
    )
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test, target_names)


if __name__ == "__main__":
    main()
