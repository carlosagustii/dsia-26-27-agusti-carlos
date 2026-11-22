import pandas as pd
import pytest


@pytest.fixture
def tiro_valido() -> dict:
    """Un tiro correcto. En cada test puedes copiarlo y cambiar un campo."""
    return {
        "shot_id": 1,
        "match_id": 100,
        "date": "2021-05-22 16:00:00",
        "season": 2020,
        "league": "Liga",
        "h_team": "Celta Vigo",
        "a_team": "Real Betis",
        "h_a": "h",
        "player": "Iago Aspas",
        "player_id": 5,
        "minute": 20,
        "x": 0.9,
        "y": 0.5,
        "situation": "OpenPlay",
        "shot_type": "LeftFoot",
        "last_action": "Pass",
        "preferred_foot": "LeftFoot",
        "is_goal": 1,
        "xg_understat": 0.3,
    }


@pytest.fixture
def tiros_df(tiro_valido) -> pd.DataFrame:
    """DataFrame con 3 tiros: 2 válidos y 1 duplicado."""
    otro = {**tiro_valido, "shot_id": 2, "is_goal": 0}
    duplicado = {**tiro_valido}
    return pd.DataFrame([tiro_valido, otro, duplicado])


# --- Proyecto II: fixtures del modelo mini ---

from pathlib import Path  # noqa: E402

from football_shots.predictor import XGPredictor  # noqa: E402
from football_shots.train import (  # noqa: E402
    ModelConfig,
    fit_pipeline,
    penalty_conversion,
    save_model,
    split_penalties,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def shots_mini() -> pd.DataFrame:
    """50 tiros reales (25 por temporada), con 2 penaltis por temporada y 14 goles."""
    return pd.read_csv(FIXTURES_DIR / "shots_mini.csv")


@pytest.fixture(scope="session")
def mini_model_path(tmp_path_factory, shots_mini) -> Path:
    """Entrena un modelo pequeño con `fit_pipeline` y lo guarda en un directorio temporal.

    Los tests no dependen del modelo real de models/, así no se rompen al reentrenar.
    """
    sin_penaltis, penaltis = split_penalties(shots_mini)
    pipeline = fit_pipeline(sin_penaltis, ModelConfig("logistic", {"C": 1.0}), min_frequency=1)
    metadata = {"penalty_xg": penalty_conversion(penaltis), "train_season": 2019}
    path = tmp_path_factory.mktemp("models") / "xg_model.joblib"
    save_model(pipeline, metadata, path)
    return path


@pytest.fixture(scope="session")
def mini_predictor(mini_model_path) -> XGPredictor:
    return XGPredictor.load(mini_model_path)
