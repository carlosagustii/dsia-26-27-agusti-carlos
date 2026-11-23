"""Tests de la API FastAPI con el modelo mini, métricas temporales y clientes de IA falsos.

Ninguno hace llamadas de red.
"""

import pytest
from fastapi.testclient import TestClient

from football_shots.api import main as api_main
from football_shots.api.main import create_app
from football_shots.api.monitoring import RECENT_SIZE, MetricsStore, percentiles
from football_shots.genai import GenAIConfigError, GenAIError, GenAITimeoutError, MockClient
from football_shots.metrics import PLAYER_METRICS_COLUMNS, player_metrics
from football_shots.predictor import score_shots

TIRO = {"x": 0.9, "y": 0.5, "minute": 30, "situation": "OpenPlay", "shot_type": "RightFoot", "last_action": "Pass"}


@pytest.fixture(scope="module")
def metrics_path(tmp_path_factory, mini_predictor, shots_mini):
    """player_metrics.csv generado con el mismo código que `cli score`."""
    tabla = player_metrics(score_shots(mini_predictor, shots_mini), train_season=2019)
    path = tmp_path_factory.mktemp("processed") / "player_metrics.csv"
    tabla.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def jugador_2020(metrics_path):
    import pandas as pd

    tabla = pd.read_csv(metrics_path)
    return tabla[tabla["season"] == 2020].sort_values("shots", ascending=False).iloc[0].to_dict()


@pytest.fixture
def client(mini_model_path, metrics_path):
    return TestClient(create_app(mini_model_path, metrics_path, genai_client=MockClient()))


class ClienteQueFalla:
    """Cliente de IA falso que lanza el error indicado."""

    name = "fake"

    def __init__(self, error):
        self.error = error

    def generate_report(self, metrics):
        raise self.error


def assert_error(respuesta, codigo):
    cuerpo = respuesta.json()
    assert respuesta.status_code == codigo
    assert set(cuerpo) == {"error", "request_id"}
    assert cuerpo["request_id"] == respuesta.headers["X-Request-ID"]
    return cuerpo


# ---------- /health ----------


def test_health_ok(client):
    respuesta = client.get("/health")

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert cuerpo["status"] == "ok"
    assert all(cuerpo["checks"].values())
    assert cuerpo["genai_provider"] == "mock"
    assert cuerpo["details"] == {}
    assert "X-Request-ID" in respuesta.headers


def test_health_sin_modelo_es_degraded_503(tmp_path, metrics_path):
    app = create_app(tmp_path / "no_existe.joblib", metrics_path, genai_client=MockClient())

    respuesta = TestClient(app).get("/health")

    cuerpo = respuesta.json()
    assert respuesta.status_code == 503
    assert cuerpo["status"] == "degraded"
    assert cuerpo["checks"]["model_loaded"] is False
    assert "train" in cuerpo["details"]["model_loaded"]


def test_health_sin_ia_configurada_es_degraded_503(mini_model_path, metrics_path, monkeypatch):
    def sin_clave():
        raise GenAIConfigError("Falta NVIDIA_API_KEY en el entorno (.env)")

    monkeypatch.setattr(api_main, "get_client", sin_clave)
    app = create_app(mini_model_path, metrics_path, genai_client=None)

    respuesta = TestClient(app).get("/health")

    assert respuesta.status_code == 503
    assert respuesta.json()["checks"]["genai_configured"] is False
    assert "NVIDIA_API_KEY" in respuesta.json()["details"]["genai_configured"]


def test_health_sin_player_metrics_es_degraded(mini_model_path, tmp_path):
    app = create_app(mini_model_path, tmp_path / "no.csv", genai_client=MockClient())

    respuesta = TestClient(app).get("/health")

    assert respuesta.status_code == 503
    assert respuesta.json()["checks"]["player_metrics_loaded"] is False


# ---------- /predict ----------


def test_predict_tiro_de_jugada(client):
    respuesta = client.post("/predict", json=TIRO)

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert 0 < cuerpo["xg"] < 1
    assert cuerpo["source"] == "model"
    assert cuerpo["request_id"] == respuesta.headers["X-Request-ID"]


def test_predict_penalti_usa_la_regla(client, mini_predictor):
    cuerpo = client.post("/predict", json={**TIRO, "situation": "Penalty"}).json()

    assert cuerpo["source"] == "penalty_rule"
    assert cuerpo["xg"] == pytest.approx(mini_predictor.penalty_xg, abs=1e-4)


def test_predict_last_action_y_minute_opcionales(client):
    sin_opcionales = {k: v for k, v in TIRO.items() if k not in ("last_action", "minute")}

    assert client.post("/predict", json=sin_opcionales).status_code == 200


@pytest.mark.parametrize(
    ("cambio", "campo"),
    [
        ({"situation": "Chilena"}, "situation"),
        ({"shot_type": "Mano"}, "shot_type"),
        ({"x": 1.5}, "x"),
        ({"y": -0.1}, "y"),
        ({"minute": 200}, "minute"),
        ({"campo_extra": 1}, "campo_extra"),
    ],
)
def test_predict_entrada_invalida_es_400(client, cambio, campo):
    cuerpo = assert_error(client.post("/predict", json={**TIRO, **cambio}), 400)

    assert campo in cuerpo["error"]


def test_predict_json_mal_formado_es_400(client):
    respuesta = client.post("/predict", content="no es json", headers={"Content-Type": "application/json"})

    assert_error(respuesta, 400)


def test_predict_mas_cerca_mas_xg(client):
    cerca = client.post("/predict", json={**TIRO, "x": 0.95}).json()["xg"]
    lejos = client.post("/predict", json={**TIRO, "x": 0.70}).json()["xg"]

    assert cerca > lejos


def test_predict_sin_modelo_es_503(tmp_path, metrics_path):
    app = create_app(tmp_path / "no.joblib", metrics_path, genai_client=MockClient())

    assert_error(TestClient(app).post("/predict", json=TIRO), 503)


# ---------- /players ----------


def test_players_ordenados_por_tiros(client):
    respuesta = client.get("/players", params={"season": 2020})

    lista = respuesta.json()
    assert respuesta.status_code == 200
    assert lista and all(j["season"] == 2020 for j in lista)
    tiros = [j["shots"] for j in lista]
    assert tiros == sorted(tiros, reverse=True)
    assert set(lista[0]) == {"player_id", "player", "season", "shots", "goals"}


def test_players_min_shots_filtra(client):
    todos = client.get("/players", params={"season": 2019}).json()
    filtrados = client.get("/players", params={"season": 2019, "min_shots": 2}).json()

    assert len(filtrados) <= len(todos)
    assert all(j["shots"] >= 2 for j in filtrados)


def test_players_sin_metricas_es_503(mini_model_path, tmp_path):
    app = create_app(mini_model_path, tmp_path / "no.csv", genai_client=MockClient())

    assert_error(TestClient(app).get("/players"), 503)


def test_player_metrics_devuelve_todas_las_columnas(client, jugador_2020):
    respuesta = client.get(f"/players/{jugador_2020['player_id']}/metrics", params={"season": 2020})

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert set(cuerpo) == set(PLAYER_METRICS_COLUMNS)
    assert cuerpo["player"] == jugador_2020["player"]
    assert isinstance(cuerpo["in_sample"], bool)


def test_player_metrics_inexistente_es_404(client):
    cuerpo = assert_error(client.get("/players/999999/metrics"), 404)

    assert "999999" in cuerpo["error"]


# ---------- /analyze ----------


def test_analyze_con_mock(client, jugador_2020):
    respuesta = client.post("/analyze", json={"player_id": jugador_2020["player_id"], "season": 2020})

    cuerpo = respuesta.json()
    assert respuesta.status_code == 200
    assert cuerpo["provider"] == "mock"
    assert jugador_2020["player"] in cuerpo["report"]
    assert cuerpo["latency_ms"] >= 0
    assert cuerpo["request_id"] == respuesta.headers["X-Request-ID"]


def test_analyze_jugador_inexistente_es_404(client):
    assert_error(client.post("/analyze", json={"player_id": 999999}), 404)


def test_analyze_temporada_no_valida_es_400(client, jugador_2020):
    assert_error(client.post("/analyze", json={"player_id": jugador_2020["player_id"], "season": 2018}), 400)


@pytest.mark.parametrize(
    ("error", "codigo"),
    [
        (GenAITimeoutError("tardó demasiado"), 504),
        (GenAIConfigError("falta la key"), 503),
        (GenAIError("500 del proveedor"), 502),
    ],
)
def test_analyze_errores_de_ia(mini_model_path, metrics_path, jugador_2020, error, codigo):
    app = create_app(mini_model_path, metrics_path, genai_client=ClienteQueFalla(error))
    client = TestClient(app)

    assert_error(client.post("/analyze", json={"player_id": jugador_2020["player_id"]}), codigo)
    genai = client.get("/metrics").json()["genai"]
    assert (genai["calls"], genai["failures"]) == (1, 1)


def test_analyze_sin_cliente_de_ia_es_503(mini_model_path, metrics_path, jugador_2020, monkeypatch):
    def sin_clave():
        raise GenAIConfigError("Falta NVIDIA_API_KEY")

    monkeypatch.setattr(api_main, "get_client", sin_clave)
    client = TestClient(create_app(mini_model_path, metrics_path, genai_client=None))

    cuerpo = assert_error(client.post("/analyze", json={"player_id": jugador_2020["player_id"]}), 503)
    assert "NVIDIA_API_KEY" in cuerpo["error"]


# ---------- /metrics y errores inesperados ----------


def test_metrics_cuenta_por_plantilla_de_ruta(client, jugador_2020):
    client.get("/health")
    client.post("/predict", json=TIRO)
    client.post("/predict", json={**TIRO, "x": 7})
    client.get(f"/players/{jugador_2020['player_id']}/metrics", params={"season": 2020})
    client.get("/players/999999/metrics")
    client.post("/analyze", json={"player_id": jugador_2020["player_id"]})
    client.get("/no-existe")
    client.get("/metrics")

    cuerpo = client.get("/metrics").json()

    peticiones = cuerpo["requests"]
    assert "/metrics" not in peticiones
    assert peticiones["/predict"]["count"] == 2
    assert peticiones["/predict"]["errors_4xx"] == 1
    assert peticiones["/players/{player_id}/metrics"]["count"] == 2
    assert peticiones["unmatched"]["count"] == 1
    assert cuerpo["totals"]["count"] == 7
    assert peticiones["/predict"]["p50_ms"] is not None
    assert cuerpo["genai"]["calls"] == 1
    # El modelo mini de conftest solo guarda penalty_xg y train_season en sus metadatos.
    assert cuerpo["model"]["train_season"] == 2019
    assert cuerpo["model"]["penalty_xg"] is not None
    assert cuerpo["recent"][0]["route"] == "unmatched"  # la más reciente primero
    assert cuerpo["uptime_s"] >= 0


def test_metrics_recent_se_limita(client):
    for _ in range(RECENT_SIZE + 5):
        client.get("/health")

    assert len(client.get("/metrics").json()["recent"]) == RECENT_SIZE


def test_error_inesperado_es_500_generico_sin_traza(mini_model_path, metrics_path):
    class PredictorRoto:
        metadata = {}

        def predict_one(self, shot):
            raise RuntimeError("detalle interno que no debe salir")

    app = create_app(mini_model_path, metrics_path, genai_client=MockClient())
    app.state.predictor = PredictorRoto()
    client = TestClient(app)

    cuerpo = assert_error(client.post("/predict", json=TIRO), 500)

    assert cuerpo["error"] == "Error interno"
    assert "detalle interno" not in client.post("/predict", json=TIRO).text
    assert client.get("/metrics").json()["requests"]["/predict"]["errors_5xx"] == 2


# ---------- monitoring ----------


def test_percentiles():
    assert percentiles([]) == (None, None)
    assert percentiles([10.0]) == (10.0, 10.0)
    p50, p95 = percentiles(range(1, 101))
    assert p50 == pytest.approx(50.5)
    assert p95 == pytest.approx(95.05, abs=0.1)


def test_metrics_store_error_rate():
    store = MetricsStore()
    store.record("/predict", "POST", 200, 5.0, "a")
    store.record("/predict", "POST", 500, 7.0, "b")

    snap = store.snapshot()

    assert snap["requests"]["/predict"]["error_rate"] == 0.5
    assert snap["totals"] == {"count": 2, "error_rate": 0.5}
    assert [r["request_id"] for r in snap["recent"]] == ["b", "a"]
