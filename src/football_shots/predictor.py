"""Inferencia: xG de uno o varios tiros con dos rutas.

- Ruta penalti: `situation == "Penalty"` -> `penalty_xg` (tasa de conversión).
- Ruta modelo: el resto -> `predict_proba` del Pipeline entrenado.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from football_shots.features import INPUT_COLUMNS
from football_shots.train import PENALTY, metadata_path

logger = logging.getLogger(__name__)

SOURCE_MODEL = "model"
SOURCE_PENALTY = "penalty_rule"


class ModelNotFoundError(FileNotFoundError):
    """No existe el modelo o sus metadatos en la ruta indicada."""


class XGPredictor:
    """Envuelve el Pipeline y `penalty_xg` para puntuar tiros."""

    def __init__(self, pipeline, penalty_xg: float, metadata: dict | None = None):
        self.pipeline = pipeline
        self.penalty_xg = float(penalty_xg)
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_path: Path) -> "XGPredictor":
        """Carga el .joblib y su .json. Solo deben cargarse modelos propios."""
        model_path = Path(model_path)
        meta_path = metadata_path(model_path)
        if not model_path.exists() or not meta_path.exists():
            raise ModelNotFoundError(
                f"No hay modelo en {model_path}. Ejecuta antes: python -m football_shots.cli train"
            )
        metadata = json.loads(meta_path.read_text())
        return cls(joblib.load(model_path), metadata["penalty_xg"], metadata)

    def predict(self, shots: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame con `xg_model` y `xg_source`, mismo índice que `shots`."""
        missing = [c for c in INPUT_COLUMNS if c not in shots.columns]
        if missing:
            raise ValueError(f"Faltan columnas para predecir: {missing}")

        is_penalty = (shots["situation"] == PENALTY).to_numpy()
        xg = np.full(len(shots), self.penalty_xg, dtype=float)
        if (~is_penalty).any():
            xg[~is_penalty] = self.pipeline.predict_proba(shots.loc[~is_penalty, INPUT_COLUMNS])[:, 1]

        source = np.where(is_penalty, SOURCE_PENALTY, SOURCE_MODEL)
        return pd.DataFrame({"xg_model": xg, "xg_source": source}, index=shots.index)

    def predict_one(self, shot: dict) -> dict:
        """xG de un único tiro dado como diccionario (lo usará la API)."""
        row = {"last_action": None, **shot}
        result = self.predict(pd.DataFrame([row])).iloc[0]
        return {"xg": round(float(result["xg_model"]), 4), "source": result["xg_source"]}


def score_shots(predictor: XGPredictor, shots: pd.DataFrame) -> pd.DataFrame:
    """Añade `xg_model` y `xg_source` a todos los tiros."""
    scored = pd.concat([shots, predictor.predict(shots)], axis=1)
    by_source = scored["xg_source"].value_counts().to_dict()
    logger.info(
        "score_done shots=%d model_route=%d penalty_route=%d",
        len(scored), by_source.get(SOURCE_MODEL, 0), by_source.get(SOURCE_PENALTY, 0),
    )
    return scored
