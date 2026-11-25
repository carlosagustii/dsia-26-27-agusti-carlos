"""Esquemas Pydantic de entrada y salida de la API.

Las categorías de `situation` y `shot_type` se importan de `validator.py`: la
API acepta exactamente las mismas que la ingesta.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from football_shots.validator import ShotType, Situation

Season = Literal[2019, 2020]


class ShotIn(BaseModel):
    """Un tiro a puntuar. Coordenadas de Understat (x = 1 es la línea de gol rival)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    # Tiro a 11 m de la línea de gol, centrado.
                    "x": 0.8952,
                    "y": 0.5,
                    "minute": 45,
                    "situation": "OpenPlay",
                    "shot_type": "RightFoot",
                    "last_action": "Pass",
                }
            ]
        },
    )

    x: float = Field(ge=0, le=1, description="Coordenada horizontal normalizada (1 = línea de gol rival)")
    y: float = Field(ge=0, le=1, description="Coordenada vertical normalizada (0,5 = centro)")
    minute: int = Field(default=45, ge=0, le=130)
    situation: Situation
    shot_type: ShotType
    last_action: str | None = Field(default=None, description="Acción previa (Pass, Cross…). Vacía = 'None'")


class PredictionOut(BaseModel):
    xg: float
    source: Literal["model", "penalty_rule"]
    request_id: str


class PlayerSummary(BaseModel):
    player_id: int
    player: str
    season: int
    shots: int
    goals: int


class PlayerMetricsOut(BaseModel):
    """Una fila de player_metrics.csv (mismas columnas que metrics.PLAYER_METRICS_COLUMNS)."""

    player_id: int
    player: str
    season: int
    shots: int
    np_shots: int
    goals: int
    np_goals: int
    xg: float
    npxg: float
    xg_understat: float
    goals_minus_xg: float
    np_goals_minus_npxg: float
    npxg_per_shot: float
    in_sample: bool


class AnalyzeIn(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"player_id": 2518, "season": 2020}]})

    player_id: int = Field(gt=0)
    season: Season = 2020


class AnalyzeOut(BaseModel):
    player_id: int
    season: int
    report: str
    provider: str
    model: str | None = Field(default=None, description="Modelo del LLM (None con el mock)")
    latency_ms: float
    request_id: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, bool]
    genai_provider: str | None
    genai_model: str | None
    version: str
    details: dict[str, str]


class ErrorOut(BaseModel):
    error: str
    request_id: str


class RouteStatsOut(BaseModel):
    count: int
    errors_4xx: int
    errors_5xx: int
    error_rate: float
    p50_ms: float | None
    p95_ms: float | None


class TotalsOut(BaseModel):
    count: int
    error_rate: float


class GenAIStatsOut(BaseModel):
    provider: str | None
    calls: int
    failures: int
    p50_ms: float | None
    p95_ms: float | None


class RecentRequestOut(BaseModel):
    ts: str
    request_id: str
    method: str
    route: str
    status: int
    latency_ms: float


class ModelInfoOut(BaseModel):
    name: str | None = None
    params: dict = Field(default_factory=dict)
    trained_at: str | None = None
    train_season: int | None = None
    test_season: int | None = None
    penalty_xg: float | None = None
    test_metrics: dict = Field(default_factory=dict)


class MetricsOut(BaseModel):
    uptime_s: float
    started_at: str
    requests: dict[str, RouteStatsOut]
    totals: TotalsOut
    genai: GenAIStatsOut
    recent: list[RecentRequestOut]
    model: ModelInfoOut | None
