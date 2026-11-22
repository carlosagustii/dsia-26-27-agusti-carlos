"""Tests de genai.py. Nunca llaman a la red: mock y un cliente NVIDIA falso."""

from types import SimpleNamespace

import openai
import pytest

from football_shots.genai import (
    GenAIConfigError,
    GenAIError,
    GenAITimeoutError,
    MockClient,
    NvidiaClient,
    build_prompt,
    generate_report,
    get_client,
)


@pytest.fixture
def metricas() -> dict:
    return {
        "player_id": 2370,
        "player": "Karim Benzema",
        "season": 2020,
        "shots": 123,
        "np_shots": 118,
        "goals": 23,
        "np_goals": 22,
        "xg": 15.93,
        "npxg": 15.09,
        "xg_understat": 19.45,
        "goals_minus_xg": 7.07,
        "np_goals_minus_npxg": 6.91,
        "npxg_per_shot": 0.128,
        "in_sample": False,
    }


def test_mock_es_determinista_y_usa_las_cifras(metricas):
    informe = MockClient().generate_report(metricas)

    assert informe == MockClient().generate_report(metricas)
    assert "Karim Benzema" in informe and "15.09 npxG" in informe
    assert "por encima" in informe
    assert "in-sample" not in informe


@pytest.mark.parametrize(
    ("diferencia", "esperado"),
    [(3.0, "por encima"), (0.0, "en línea"), (-3.0, "por debajo")],
)
def test_mock_veredicto_segun_goles_menos_npxg(metricas, diferencia, esperado):
    informe = MockClient().generate_report({**metricas, "np_goals_minus_npxg": diferencia})

    assert esperado in informe


def test_mock_avisa_si_es_in_sample(metricas):
    assert "in-sample" in MockClient().generate_report({**metricas, "in_sample": True})


def test_prompt_incluye_cifras_y_aviso_in_sample(metricas):
    prompt = build_prompt(metricas)

    assert "2020-21" in prompt
    assert "npxG: 15.09" in prompt
    assert "in_sample: false" in prompt


def test_get_client_por_defecto_es_mock():
    assert get_client("mock").name == "mock"


def test_get_client_proveedor_desconocido():
    with pytest.raises(GenAIConfigError, match="desconocido"):
        get_client("chatgpt")


def test_nvidia_sin_api_key_lanza_error_de_configuracion():
    with pytest.raises(GenAIConfigError, match="NVIDIA_API_KEY"):
        NvidiaClient(api_key="")


class FakeCompletions:
    """Sustituye a `client.chat.completions` del SDK de OpenAI."""

    def __init__(self, respuesta=None, error=None):
        self.respuesta = respuesta
        self.error = error
        self.llamadas = []

    def create(self, **kwargs):
        self.llamadas.append(kwargs)
        if self.error:
            raise self.error
        mensaje = SimpleNamespace(content=self.respuesta)
        return SimpleNamespace(choices=[SimpleNamespace(message=mensaje)])


def peticion_falsa():
    """Request HTTP que exigen las excepciones del SDK de OpenAI (usa httpx2 o httpx)."""
    try:
        import httpx2 as http
    except ImportError:
        import httpx as http
    return http.Request("POST", "https://test")


def nvidia_con(fake: FakeCompletions) -> NvidiaClient:
    cliente = NvidiaClient(api_key="nvapi-test", model="modelo-test")
    cliente._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return cliente


def test_nvidia_envia_system_y_user_y_devuelve_texto(metricas):
    fake = FakeCompletions(respuesta="  Informe del LLM  ")

    informe = generate_report(metricas, nvidia_con(fake))

    assert informe == "Informe del LLM"
    llamada = fake.llamadas[0]
    assert llamada["model"] == "modelo-test"
    assert [m["role"] for m in llamada["messages"]] == ["system", "user"]
    assert "Karim Benzema" in llamada["messages"][1]["content"]


def test_nvidia_timeout_se_convierte_en_genai_timeout(metricas):
    error = openai.APITimeoutError(request=peticion_falsa())

    with pytest.raises(GenAITimeoutError):
        nvidia_con(FakeCompletions(error=error)).generate_report(metricas)


def test_nvidia_error_de_api_se_convierte_en_genai_error(metricas):
    error = openai.APIConnectionError(request=peticion_falsa())

    with pytest.raises(GenAIError):
        nvidia_con(FakeCompletions(error=error)).generate_report(metricas)


def test_nvidia_respuesta_vacia_es_error(metricas):
    with pytest.raises(GenAIError, match="vacía"):
        nvidia_con(FakeCompletions(respuesta="")).generate_report(metricas)
