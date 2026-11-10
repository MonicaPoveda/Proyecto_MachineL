"""Train and compare phishing classifiers, then export the winner.

Compares three algorithms on a stratified held-out test set:
  * Random Forest
  * SVM (RBF, with probability estimates)
  * Decision Tree

Metrics reported per model: accuracy, precision, recall, F1 and the confusion
matrix. The winner is chosen by test F1 (tie-break: accuracy) and serialized
for production together with its feature columns and metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from modeling.data import build_dataset
from modeling.features import FEATURE_COLUMNS, feature_vector

RANDOM_STATE = 42
LABEL_COLUMN = "label"
THRESHOLD = 0.5


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_feature_matrix(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([feature_vector(url) for url in urls], columns=FEATURE_COLUMNS)


def build_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare phishing classifiers."
    )
    parser.add_argument(
        "--malicious-limit",
        type=int,
        default=5000,
        help="Maximum malicious URLs to load (default: 5000).",
    )
    parser.add_argument(
        "--legit-limit",
        type=int,
        default=5000,
        help="Maximum legitimate URLs to load (default: 5000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <project_root>/modeling_output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or (_project_root() / "modeling_output")
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building dataset ===")
    data = build_dataset(
        malicious_limit=args.malicious_limit,
        legit_limit=args.legit_limit,
        random_state=RANDOM_STATE,
    )
    X = build_feature_matrix(data["url"])
    y = data[LABEL_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {X.shape[1]}\n")

    models = build_models()
    rows: list[dict] = []
    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        rows.append(metrics)

        joblib.dump(model, models_dir / f"{name}.joblib")

        print(f"  F1       = {metrics['f1']:.4f}")
        print(f"  Precision= {metrics['precision']:.4f}")
        print(f"  Recall   = {metrics['recall']:.4f}")
        print(f"  Accuracy = {metrics['accuracy']:.4f}")
        print(f"  Confusion [[TN, FP], [FN, TP]] = {metrics['confusion_matrix']}")
        print(
            classification_report(
                y_test,
                model.predict(X_test),
                target_names=["legitimate", "phishing"],
                zero_division=0,
            )
        )
        print()

    comparison = pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    winner_name = str(comparison.iloc[0]["model"])
    winner_row = comparison.iloc[0]

    metadata = {
        "model": winner_name,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "classes": [0, 1],
        "threshold": THRESHOLD,
        "metrics": winner_row.drop(labels=["model"]).to_dict(),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "random_state": RANDOM_STATE,
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    joblib.dump(models[winner_name], output_dir / "winner.joblib")

    print("=== Model comparison (held-out test set) ===")
    print(
        comparison[["model", "f1", "precision", "recall", "accuracy"]].to_string(
            index=False
        )
    )
    print(f"\nWinner: {winner_name} (F1 = {winner_row['f1']:.4f})")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
