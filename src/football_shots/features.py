"""Features del tiro: geometría (distancia y ángulo) y preprocesado para sklearn.

Coordenadas de Understat: `x = 1` es la línea de gol rival e `y = 0.5` el centro.
Se asume un campo de 105 x 68 m y una portería de 7,32 m.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

NUMERIC_FEATURES = ["distance", "angle", "minute"]
CATEGORICAL_FEATURES = ["situation", "shot_type", "last_action"]
# Columnas que necesita el modelo como entrada (antes de calcular la geometría).
INPUT_COLUMNS = ["x", "y", "minute", "situation", "shot_type", "last_action"]

NO_LAST_ACTION = "None"


def shot_distance_angle(x, y):
    """Distancia (m) y ángulo de visión de la portería (rad) de un tiro.

    El ángulo es el que forman los dos postes vistos desde el punto de tiro.
    """
    x_m = (1 - np.asarray(x, dtype=float)) * PITCH_LENGTH_M
    y_m = (np.asarray(y, dtype=float) - 0.5) * PITCH_WIDTH_M
    distance = np.sqrt(x_m**2 + y_m**2)
    half_goal = GOAL_WIDTH_M / 2
    angle = np.arctan2(GOAL_WIDTH_M * x_m, x_m**2 + y_m**2 - half_goal**2)
    # Detrás de la línea de fondo el ángulo saldría negativo: se acota a 0.
    angle = np.clip(angle, 0, np.pi)
    return distance, angle


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con `distance`, `angle` y `last_action` sin vacíos."""
    out = df.copy()
    out["distance"], out["angle"] = shot_distance_angle(out["x"], out["y"])
    # Que no haya acción previa es información, no un dato perdido.
    out["last_action"] = out["last_action"].fillna(NO_LAST_ACTION).astype(str)
    out["minute"] = out["minute"].astype(float)
    return out


def build_preprocessor(min_frequency: int = 20) -> ColumnTransformer:
    """Escala las numéricas y hace one-hot de las categóricas.

    `min_frequency` agrupa las categorías raras de `last_action` en una sola;
    `handle_unknown` evita fallos con categorías nuevas en producción.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def geometry_transformer() -> FunctionTransformer:
    """Paso de pipeline que aplica `add_geometry` (así el .joblib lo incluye)."""
    return FunctionTransformer(add_geometry, validate=False)
