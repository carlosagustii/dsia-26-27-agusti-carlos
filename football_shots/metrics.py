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