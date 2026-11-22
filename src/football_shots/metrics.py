import pandas as pd


def total_tiros(df: pd.DataFrame) -> int:
    #Cuántos tiros hay.
    return len(df)


def total_goles(df: pd.DataFrame) -> int:
    #Cuántos goles hay (is_goal vale 1 si fue gol y 0 si no).
    return int(df["is_goal"].sum())


def porcentaje_acierto(df: pd.DataFrame) -> float:
    #de cada 100 tiros, cuántos acaban en gol.
    if len(df) == 0:
        return 0.0
    return round(total_goles(df) / total_tiros(df) * 100, 2)


def maximos_goleadores(df: pd.DataFrame, n: int = 5) -> pd.Series:
    # Los n jugadores con más goles.
    goles = df[df["is_goal"] == 1]
    return goles["player"].value_counts().head(n)

# --- Proyecto II: métricas por jugador y temporada a partir de los tiros puntuados ---

PLAYER_METRICS_COLUMNS = [
    "player_id", "player", "season",
    "shots", "np_shots", "goals", "np_goals",
    "xg", "npxg", "xg_understat",
    "goals_minus_xg", "np_goals_minus_npxg", "npxg_per_shot",
    "in_sample",
]


def player_metrics(scored: pd.DataFrame, train_season: int = 2019) -> pd.DataFrame:
    """Agrega los tiros puntuados (`xg_model`) por jugador y temporada.

    - npxG / np_*: sin penaltis (la métrica estándar para valorar a un jugador).
    - goals_minus_xg > 0: el jugador marca más de lo que "deberían" sus ocasiones.
    - in_sample: True en la temporada de entrenamiento (el modelo ya vio esos tiros).
    """
    df = scored.copy()
    no_penalty = df["situation"] != "Penalty"
    df["np_shot"] = no_penalty.astype(int)
    df["np_goal"] = df["is_goal"].where(no_penalty, 0)
    df["np_xg"] = df["xg_model"].where(no_penalty, 0.0)

    grouped = df.groupby(["player_id", "season"]).agg(
        player=("player", "first"),
        shots=("is_goal", "size"),
        np_shots=("np_shot", "sum"),
        goals=("is_goal", "sum"),
        np_goals=("np_goal", "sum"),
        xg=("xg_model", "sum"),
        npxg=("np_xg", "sum"),
        xg_understat=("xg_understat", "sum"),
    ).reset_index()

    grouped["goals_minus_xg"] = grouped["goals"] - grouped["xg"]
    grouped["np_goals_minus_npxg"] = grouped["np_goals"] - grouped["npxg"]
    grouped["npxg_per_shot"] = (grouped["npxg"] / grouped["np_shots"]).where(grouped["np_shots"] > 0, 0.0)
    grouped["in_sample"] = grouped["season"] == train_season

    float_cols = ["xg", "npxg", "xg_understat", "goals_minus_xg", "np_goals_minus_npxg", "npxg_per_shot"]
    grouped[float_cols] = grouped[float_cols].round(3)
    ordered = grouped[PLAYER_METRICS_COLUMNS].sort_values(["season", "npxg"], ascending=[True, False])
    return ordered.reset_index(drop=True)
