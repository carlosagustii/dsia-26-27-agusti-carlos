import pandas as pd

SITUATIONS = {"OpenPlay", "FromCorner", "SetPiece", "DirectFreekick", "Penalty"}
SHOT_TYPES = {"RightFoot", "LeftFoot", "Head", "OtherBodyPart"}
MAX_MINUTE = 130


class ValidationError(Exception):
    """Datos que no cumplen reglas de negocio."""


def validar_tiros(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (validos, errores). No modifica el DataFrame original."""
    if frame.empty:
        raise ValidationError("No hay tiros que validar")

    work = frame.copy()
    # Normalización: en los datos aparece "Right" en vez de "RightFoot".
    work["preferred_foot"] = work["preferred_foot"].replace({"Right": "RightFoot"})

    ok = (
        work["x"].between(0, 1)
        & work["y"].between(0, 1)
        & work["xg_understat"].between(0, 1)
        & work["minute"].between(0, MAX_MINUTE)
        & work["is_goal"].isin([0, 1])
        & work["situation"].isin(SITUATIONS)
        & work["shot_type"].isin(SHOT_TYPES)
        & work["player"].notna()
        & ~work["shot_id"].duplicated(keep="first")
    )
    validos = work.loc[ok].copy()
    errores = work.loc[~ok].copy()
    return validos, errores