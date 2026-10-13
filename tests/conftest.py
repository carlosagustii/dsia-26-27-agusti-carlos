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
