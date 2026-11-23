"""API FastAPI del modelo xG (Proyecto III).

La API va encima del núcleo de los Proyectos I y II y no lo reimplementa:
- `XGPredictor` para /predict,
- `player_metrics.csv` (salida de `cli score`) y `cli.find_player` para los jugadores,
- `genai.generate_report` para /analyze.

Arranque:
    uvicorn football_shots.api.main:app --host 0.0.0.0 --port 8000

`create_app` recibe sus dependencias (rutas y cliente de IA) para poder
probarla con un modelo mini y un cliente falso.
"""

import logging
import time
import uuid
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from football_shots import config
from football_shots.api.monitoring import MetricsStore
from football_shots.api.schemas import (
    AnalyzeIn,
    AnalyzeOut,
    ErrorOut,
    HealthOut,
    MetricsOut,
    PlayerMetricsOut,
    PlayerSummary,
    PredictionOut,
    ShotIn,
)
from football_shots.cli import find_player
from football_shots.genai import (
    GenAIClient,
    GenAIConfigError,
    GenAIError,
    GenAITimeoutError,
    generate_report,
    get_client,
)
from football_shots.predictor import ModelNotFoundError, XGPredictor

logger = logging.getLogger("football_shots.api")

PACKAGE_NAME = "dsia-26-27-agusti-carlos"
UNMATCHED_ROUTE = "unmatched"
NOT_RECORDED_ROUTES = {"/metrics"}  # el refresco del dashboard no ensucia las cifras

ERROR_RESPONSES = {
    400: {"model": ErrorOut, "description": "Entrada inválida"},
    404: {"model": ErrorOut, "description": "No encontrado"},
    500: {"model": ErrorOut, "description": "Error interno"},
    503: {"model": ErrorOut, "description": "Servicio no disponible"},
}


def _app_version() -> str:
    try:
        return package_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "unknown"


def _native(value):
    """Convierte tipos numpy a tipos de Python (numpy no serializa a JSON)."""
    return value.item() if hasattr(value, "item") else value


def _error(request: Request, status_code: int, message: str) -> JSONResponse:
    """Cuerpo de error común: {"error", "request_id"}."""
    request_id = getattr(request.state, "request_id", "-")
    return JSONResponse(status_code=status_code, content={"error": message, "request_id": request_id})


def _load_state(app: FastAPI, model_path: Path, player_metrics_path: Path, genai_client: GenAIClient | None):
    """Carga modelo, métricas y cliente de IA sin tumbar la app si algo falla.

    Lo que no se pueda cargar queda a None con su motivo; /health lo reporta.
    """
    state = app.state
    state.details = {}

    try:
        state.predictor = XGPredictor.load(model_path)
    except ModelNotFoundError as exc:
        state.predictor = None
        state.details["model_loaded"] = str(exc)
        logger.error("startup_model_missing error=%s", exc)

    player_metrics_path = Path(player_metrics_path)
    if player_metrics_path.exists():
        state.player_metrics = pd.read_csv(player_metrics_path)
    else:
        state.player_metrics = None
        state.details["player_metrics_loaded"] = (
            f"No existe {player_metrics_path}. Ejecuta: python -m football_shots.cli ingest && "
            "python -m football_shots.cli score"
        )
        logger.error("startup_player_metrics_missing path=%s", player_metrics_path)

    state.genai_error = None
    if genai_client is None:
        try:
            genai_client = get_client()
        except GenAIConfigError as exc:
            state.genai_error = str(exc)
            state.details["genai_configured"] = str(exc)
            logger.error("startup_genai_not_configured error=%s", exc)
    state.genai_client = genai_client

    state.metrics_store = MetricsStore()
    state.version = _app_version()
    logger.info(
        "startup model=%s player_metrics=%s genai=%s version=%s",
        state.predictor is not None,
        state.player_metrics is not None,
        getattr(genai_client, "name", None),
        state.version,
    )


def create_app(
    model_path: Path = config.MODEL_PATH,
    player_metrics_path: Path = config.PLAYER_METRICS_CSV,
    genai_client: GenAIClient | None = None,
) -> FastAPI:
    """Crea la app con sus dependencias. `genai_client=None` usa `get_client()`."""
    config.setup_logging()
    app = FastAPI(
        title="xG LaLiga API",
        description="Modelo de expected goals (xG) e informes de scouting con IA generativa.",
        version=_app_version(),
    )
    _load_state(app, model_path, player_metrics_path, genai_client)

    # ---------- Middleware: request_id, latencia, log y métricas ----------

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Errores no previstos: 500 genérico, sin traza para el cliente.
            # Se capturan aquí (y no con un handler de Exception) para que la
            # respuesta también lleve request_id y quede registrada en /metrics.
            logger.exception("unhandled_error request_id=%s path=%s", request_id, request.url.path)
            response = _error(request, 500, "Error interno")
        latency_ms = (time.perf_counter() - start) * 1000

        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or UNMATCHED_ROUTE
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            request_id, request.method, request.url.path, response.status_code, latency_ms,
        )
        if route_path not in NOT_RECORDED_ROUTES:
            app.state.metrics_store.record(route_path, request.method, response.status_code, latency_ms, request_id)
        return response

    # ---------- Handlers de error: siempre {"error", "request_id"} ----------

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        parts = [str(part) for part in first["loc"] if isinstance(part, str) and part not in ("body", "query", "path")]
        field = ".".join(parts)
        message = f"{field}: {first['msg']}" if field else first["msg"]
        return _error(request, 400, f"Entrada inválida · {message}")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return _error(request, exc.status_code, str(exc.detail))

    # Las subclases van antes que GenAIError (heredan de ella).
    @app.exception_handler(GenAITimeoutError)
    async def genai_timeout(request: Request, exc: GenAITimeoutError):
        logger.error("genai_timeout request_id=%s error=%s", request.state.request_id, exc)
        return _error(request, 504, f"El proveedor de IA no respondió a tiempo: {exc}")

    @app.exception_handler(GenAIConfigError)
    async def genai_not_configured(request: Request, exc: GenAIConfigError):
        logger.error("genai_not_configured request_id=%s error=%s", request.state.request_id, exc)
        return _error(request, 503, f"IA generativa no configurada: {exc}")

    @app.exception_handler(GenAIError)
    async def genai_error(request: Request, exc: GenAIError):
        logger.error("genai_error request_id=%s error=%s", request.state.request_id, exc)
        return _error(request, 502, f"Error del proveedor de IA: {exc}")

    # ---------- Endpoints ----------

    def require_player_metrics() -> pd.DataFrame:
        if app.state.player_metrics is None:
            raise HTTPException(503, app.state.details["player_metrics_loaded"])
        return app.state.player_metrics

    @app.get("/health", response_model=HealthOut, responses={503: {"model": HealthOut}}, tags=["operación"])
    def health():
        """Estado del servicio. 503 si algún check falla. Nunca llama al LLM."""
        state = app.state
        client = state.genai_client
        checks = {
            "model_loaded": state.predictor is not None,
            "player_metrics_loaded": state.player_metrics is not None,
            "genai_configured": client is not None,
        }
        status = "ok" if all(checks.values()) else "degraded"
        body = HealthOut(
            status=status,
            checks=checks,
            genai_provider=client.name if client else config.GENAI_PROVIDER,
            genai_model=getattr(client, "model", None) if client and client.name == "nvidia" else None,
            version=state.version,
            details={k: v for k, v in state.details.items() if not checks.get(k, True)},
        )
        if status == "degraded":
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.post("/predict", response_model=PredictionOut, responses=ERROR_RESPONSES, tags=["producto"])
    def predict(shot: ShotIn, request: Request):
        """xG de un tiro. Los penaltis usan la regla `penalty_xg`; el resto, el modelo."""
        if app.state.predictor is None:
            raise HTTPException(503, f"Modelo no disponible: {app.state.details['model_loaded']}")
        result = app.state.predictor.predict_one(shot.model_dump())
        return PredictionOut(**result, request_id=request.state.request_id)

    @app.get("/players", response_model=list[PlayerSummary], responses=ERROR_RESPONSES, tags=["producto"])
    def players(season: int = Query(2020), min_shots: int = Query(1, ge=0)):
        """Jugadores de una temporada, ordenados por número de tiros."""
        df = require_player_metrics()
        rows = df[(df["season"] == season) & (df["shots"] >= min_shots)]
        rows = rows.sort_values(["shots", "player"], ascending=[False, True])
        columns = ["player_id", "player", "season", "shots", "goals"]
        return [{k: _native(v) for k, v in row.items()} for row in rows[columns].to_dict("records")]

    @app.get(
        "/players/{player_id}/metrics", response_model=PlayerMetricsOut, responses=ERROR_RESPONSES, tags=["producto"]
    )
    def player_metrics(player_id: int, season: int = Query(2020)):
        """Métricas de un jugador en una temporada (goles, xG, npxG, goles − xG…)."""
        df = require_player_metrics()
        try:
            row = find_player(df, season, player_id=player_id)
        except LookupError as exc:
            raise HTTPException(404, f"Jugador {player_id}: {exc}") from exc
        return PlayerMetricsOut(**{k: _native(v) for k, v in row.items()})

    @app.post(
        "/analyze",
        response_model=AnalyzeOut,
        responses={**ERROR_RESPONSES, 502: {"model": ErrorOut}, 504: {"model": ErrorOut}},
        tags=["producto"],
    )
    def analyze(body: AnalyzeIn, request: Request):
        """Informe de scouting con IA generativa a partir de las métricas del jugador."""
        df = require_player_metrics()
        try:
            row = find_player(df, body.season, player_id=body.player_id)
        except LookupError as exc:
            raise HTTPException(404, f"Jugador {body.player_id}: {exc}") from exc

        client = app.state.genai_client
        if client is None:
            raise GenAIConfigError(app.state.genai_error or "sin cliente de IA")

        metrics = {k: _native(v) for k, v in row.items()}
        store = app.state.metrics_store
        start = time.perf_counter()
        try:
            report = generate_report(metrics, client)
        except GenAIError:
            store.record_genai((time.perf_counter() - start) * 1000, ok=False, provider=client.name)
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        store.record_genai(latency_ms, ok=True, provider=client.name)
        logger.info(
            "analyze_ok request_id=%s player_id=%s provider=%s genai_latency_ms=%.1f",
            request.state.request_id, body.player_id, client.name, latency_ms,
        )
        return AnalyzeOut(
            player_id=body.player_id,
            season=body.season,
            report=report,
            provider=client.name,
            latency_ms=round(latency_ms, 1),
            request_id=request.state.request_id,
        )

    @app.get("/metrics", response_model=MetricsOut, tags=["operación"])
    def metrics():
        """Métricas de operación en memoria: peticiones, latencias, IA y modelo."""
        snapshot = app.state.metrics_store.snapshot()
        predictor = app.state.predictor
        model = None
        if predictor is not None:
            meta = predictor.metadata
            model = {
                "name": meta.get("model", {}).get("name"),
                "params": meta.get("model", {}).get("params", {}),
                "trained_at": meta.get("trained_at"),
                "train_season": meta.get("train_season"),
                "test_season": meta.get("test_season"),
                "penalty_xg": meta.get("penalty_xg"),
                "test_metrics": meta.get("test_metrics", {}),
            }
        return {**snapshot, "model": model}

    return app


app = create_app()
