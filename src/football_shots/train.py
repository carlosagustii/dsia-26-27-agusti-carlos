"""Entrenamiento del modelo xG.

Protocolo (ver docs/ARQUITECTURA.md, sección 3.3):
1. Split temporal: train = 2019, test = 2020.
2. Los penaltis se separan: su xG es la tasa de conversión en train (`penalty_xg`).
3. `select_model`: validación cruzada con GroupKFold por `match_id` dentro de 2019
   para elegir entre regresión logística y HistGradientBoosting.
4. `fit_pipeline`: ajuste directo con la configuración elegida.
5. Evaluación única sobre 2020 sin penaltis (`evaluate.py`).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from football_shots import config
from football_shots.evaluate import evaluate, plot_calibration
from football_shots.features import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    build_preprocessor,
    geometry_transformer,
)

logger = logging.getLogger(__name__)

TARGET = "is_goal"
PENALTY = "Penalty"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelConfig:
    """Qué estimador usar y con qué hiperparámetros."""

    name: str
    params: dict = field(default_factory=dict)


# Candidatos que compara `select_model`. La logística es el baseline.
CANDIDATES = [
    ModelConfig("logistic", {"C": 0.1}),
    ModelConfig("logistic", {"C": 1.0}),
    ModelConfig("logistic", {"C": 10.0}),
    ModelConfig("hist_gb", {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15}),
    ModelConfig("hist_gb", {"learning_rate": 0.03, "max_iter": 300, "max_depth": 4}),
]


def make_estimator(model_config: ModelConfig):
    """Crea el clasificador de sklearn a partir de la configuración."""
    if model_config.name == "logistic":
        return LogisticRegression(max_iter=2000, **model_config.params)
    if model_config.name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **model_config.params)
    raise ValueError(f"Modelo desconocido: {model_config.name}")


def build_pipeline(model_config: ModelConfig, min_frequency: int = 20) -> Pipeline:
    """Geometría + preprocesado + modelo, todo en un único Pipeline."""
    return Pipeline(
        steps=[
            ("geometry", geometry_transformer()),
            ("preprocess", build_preprocessor(min_frequency=min_frequency)),
            ("model", make_estimator(model_config)),
        ]
    )


def split_penalties(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (tiros sin penaltis, penaltis)."""
    is_penalty = df["situation"] == PENALTY
    return df[~is_penalty], df[is_penalty]


def penalty_conversion(penalties: pd.DataFrame) -> float:
    """Tasa de conversión de penaltis: es el xG que se asigna a cualquier penalti."""
    if penalties.empty:
        raise ValueError("No hay penaltis para estimar penalty_xg")
    return float(penalties[TARGET].mean())


def fit_pipeline(df: pd.DataFrame, model_config: ModelConfig, min_frequency: int = 20) -> Pipeline:
    """Ajuste directo, sin validación cruzada. `df` no debe contener penaltis."""
    pipeline = build_pipeline(model_config, min_frequency=min_frequency)
    pipeline.fit(df[INPUT_COLUMNS], df[TARGET])
    return pipeline


def select_model(
    df: pd.DataFrame, candidates: list[ModelConfig] = CANDIDATES, n_splits: int = 5
) -> tuple[ModelConfig, list[dict]]:
    """Elige el candidato con menor log loss en GroupKFold por partido.

    Agrupar por `match_id` evita que tiros del mismo partido estén a la vez en
    entrenamiento y validación.
    """
    cv = GroupKFold(n_splits=n_splits)
    results = []
    for candidate in candidates:
        scores = cross_val_score(
            build_pipeline(candidate),
            df[INPUT_COLUMNS],
            df[TARGET],
            groups=df["match_id"],
            cv=cv,
            scoring="neg_log_loss",
        )
        cv_log_loss = float(-scores.mean())
        results.append({**asdict(candidate), "cv_log_loss": round(cv_log_loss, 5)})
        logger.info("train_cv model=%s params=%s log_loss=%.5f", candidate.name, candidate.params, cv_log_loss)

    best = min(results, key=lambda r: r["cv_log_loss"])
    chosen = ModelConfig(best["name"], best["params"])
    logger.info("train_selected model=%s params=%s", chosen.name, chosen.params)
    return chosen, results


def run_training(
    shots_path: Path = config.SHOTS_CSV,
    model_path: Path = config.MODEL_PATH,
    reports_dir: Path = config.REPORTS_DIR,
    candidates: list[ModelConfig] = CANDIDATES,
    n_splits: int = 5,
) -> dict:
    """Entrena, evalúa en 2020 y guarda el modelo (.joblib) y sus metadatos (.json)."""
    shots = pd.read_csv(shots_path)
    train = shots[shots["season"] == config.TRAIN_SEASON]
    test = shots[shots["season"] == config.TEST_SEASON]
    if train.empty or test.empty:
        raise ValueError("Faltan tiros de la temporada de train o de test")

    train_np, train_pen = split_penalties(train)
    test_np, _ = split_penalties(test)
    penalty_xg = penalty_conversion(train_pen)
    logger.info(
        "train_split train=%d test=%d train_penalties=%d penalty_xg=%.4f",
        len(train_np), len(test_np), len(train_pen), penalty_xg,
    )

    chosen, cv_results = select_model(train_np, candidates, n_splits)
    pipeline = fit_pipeline(train_np, chosen)

    model_probs = pipeline.predict_proba(test_np[INPUT_COLUMNS])[:, 1]
    model_metrics = evaluate(test_np[TARGET], model_probs)
    benchmark_metrics = evaluate(test_np[TARGET], test_np["xg_understat"])
    logger.info("train_test_metrics model=%s understat=%s", model_metrics, benchmark_metrics)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(
        test_np[TARGET],
        {"Modelo propio": model_probs, "xG Understat": test_np["xg_understat"].to_numpy()},
        reports_dir / "calibration.png",
    )

    metadata = {
        "model": asdict(chosen),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "input_columns": INPUT_COLUMNS,
        "train_season": config.TRAIN_SEASON,
        "test_season": config.TEST_SEASON,
        "n_train": len(train_np),
        "n_test": len(test_np),
        "penalty_xg": round(penalty_xg, 4),
        "n_penalties_train": len(train_pen),
        "cv_results": cv_results,
        "test_metrics": {"model": model_metrics, "xg_understat": benchmark_metrics},
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_model(pipeline, metadata, model_path)
    return metadata


def metadata_path(model_path: Path) -> Path:
    """Los metadatos van junto al modelo, con extensión .json."""
    return Path(model_path).with_suffix(".json")


def save_model(pipeline: Pipeline, metadata: dict, model_path: Path) -> None:
    """Guarda el Pipeline con joblib y los metadatos en JSON."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path(model_path).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    logger.info("train_saved model=%s", model_path)
