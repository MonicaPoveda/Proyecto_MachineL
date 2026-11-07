"""
CRISP-DM - Fase 4: Modelado
Entrena y compara modelos usando validacion cruzada sobre train.csv.
El conjunto test.csv se reserva para la fase 5 (Evaluacion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "is_phishing"
RANDOM_STATE = 42

SCORING = {
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def project_root() -> Path:
    """Obtiene la raiz del proyecto desde la ubicacion de este archivo."""
    return Path(__file__).resolve().parents[3]


def load_training_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga y valida el conjunto de entrenamiento de la fase 3."""
    if not train_path.is_file():
        raise FileNotFoundError(
            f"No se encontro {train_path}. Ejecuta primero la fase 3 "
            "y verifica que se haya generado phase3_output/train.csv."
        )

    data = pd.read_csv(train_path)
    if LABEL_COLUMN not in data.columns:
        raise ValueError(f"El archivo debe incluir la columna '{LABEL_COLUMN}'.")

    if data.empty:
        raise ValueError("El conjunto de entrenamiento esta vacio.")

    X = data.drop(columns=[LABEL_COLUMN])
    y = data[LABEL_COLUMN]

    if X.empty:
        raise ValueError("No se encontraron columnas de caracteristicas.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("Todas las caracteristicas deben ser numericas.")
    if X.isnull().any().any():
        raise ValueError("Hay valores nulos en las caracteristicas.")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Hay valores infinitos o no finitos en las caracteristicas.")
    if set(y.unique()) != {0, 1}:
        raise ValueError("La etiqueta is_phishing debe contener las clases 0 y 1.")

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Se necesitan al menos dos registros por clase.")

    return X, y.astype(int)


def build_searches(cv_folds: int, n_jobs: int) -> dict[str, GridSearchCV]:
    """Define los algoritmos y rangos iniciales de hiperparametros."""
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    logistic_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            )
        ]
    )

    gradient_boosting_pipeline = Pipeline(
        steps=[
            (
                "model",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
            )
        ]
    )

    configurations = {
        "logistic_regression": (
            logistic_pipeline,
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            random_forest_pipeline,
            {
                "model__n_estimators": [200],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "gradient_boosting": (
            gradient_boosting_pipeline,
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }

    searches = {}
    for name, (pipeline, parameters) in configurations.items():
        searches[name] = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters,
            scoring=SCORING,
            refit="f1",
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True,
            error_score="raise",
        )

    return searches


def run_modeling(
    train_path: Path,
    output_dir: Path,
    cv_folds: int = 3,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Entrena, ajusta y compara los modelos mediante validacion cruzada."""
    if cv_folds < 2:
        raise ValueError("cv_folds debe ser como minimo 2.")

    X_train, y_train = load_training_data(train_path)
    if y_train.value_counts().min() < cv_folds:
        raise ValueError(
            f"Cada clase necesita al menos {cv_folds} registros para "
            "la validacion cruzada seleccionada."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CRISP-DM | Fase 4: Modelado ===")
    print(f"Registros de entrenamiento: {len(X_train)}")
    print(f"Caracteristicas utilizadas: {len(X_train.columns)}")
    print(f"Distribucion de clases: {y_train.value_counts().to_dict()}")
    print(f"Validacion cruzada: {cv_folds} particiones estratificadas")
    print("El archivo test.csv no se carga ni se utiliza en esta fase.\n")

    searches = build_searches(cv_folds=cv_folds, n_jobs=n_jobs)
    results = []

    for name, search in searches.items():
        print(f"\n--- Entrenando: {name} ---")
        search.fit(X_train, y_train)

        best_index = search.best_index_
        cv_results = search.cv_results_
        result = {
            "model": name,
            "cv_f1_mean": float(cv_results["mean_test_f1"][best_index]),
            "cv_f1_std": float(cv_results["std_test_f1"][best_index]),
            "cv_precision_mean": float(
                cv_results["mean_test_precision"][best_index]
            ),
            "cv_recall_mean": float(cv_results["mean_test_recall"][best_index]),
            "cv_roc_auc_mean": float(cv_results["mean_test_roc_auc"][best_index]),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(result)

        model_path = models_dir / f"{name}.joblib"
        joblib.dump(search.best_estimator_, model_path)

        pd.DataFrame(cv_results).to_csv(
            output_dir / f"{name}_search_results.csv",
            index=False,
        )

        print(f"Mejor F1 promedio (CV): {result['cv_f1_mean']:.4f}")
        print(f"Mejores parametros: {search.best_params_}")
        print(f"Modelo guardado en: {model_path}")

    comparison = pd.DataFrame(results).sort_values(
        by="cv_f1_mean",
        ascending=False,
    )
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    metadata = {
        "phase": "CRISP-DM Phase 4 - Modeling",
        "training_file": str(train_path),
        "records": int(len(X_train)),
        "features": list(X_train.columns),
        "label": LABEL_COLUMN,
        "cv_folds": cv_folds,
        "selection_metric": "f1",
        "random_state": RANDOM_STATE,
        "test_set_used": False,
        "models": comparison["model"].tolist(),
    }
    with (output_dir / "modeling_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("\n=== Comparacion de modelos (validacion cruzada) ===")
    print(
        comparison[
            ["model", "cv_f1_mean", "cv_f1_std", "cv_precision_mean",
             "cv_recall_mean", "cv_roc_auc_mean"]
        ].to_string(index=False)
    )
    print(f"\nResultados guardados en: {output_dir}")
    print(
        "Fase 4 completada. La evaluacion final con test.csv corresponde "
        "a la fase 5."
    )

    return comparison


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Entrenamiento y comparacion de modelos para detectar phishing."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=root / "phase3_output" / "train.csv",
        help="Ruta al train.csv generado en la fase 3.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "phase4_output",
        help="Carpeta donde se guardaran modelos y resultados.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Numero de particiones de validacion cruzada (por defecto: 3).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Procesos paralelos de GridSearchCV (por defecto: -1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_modeling(
        train_path=arguments.train,
        output_dir=arguments.output,
        cv_folds=arguments.cv_folds,
        n_jobs=arguments.n_jobs,
    )
