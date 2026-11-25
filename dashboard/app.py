"""Dashboard Streamlit del modelo xG de LaLiga.

Solo consume la API por HTTP (api_client.py, variable API_URL). No importa
football_shots ni lee data/ ni models/.

Arranque:
    uv run streamlit run dashboard/app.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Arc, Rectangle  # noqa: E402

import api_client  # noqa: E402
from api_client import ApiError  # noqa: E402

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

SITUATIONS = ["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick", "Penalty"]
SHOT_TYPES = ["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]
LAST_ACTIONS = [
    "None", "Pass", "Cross", "Aerial", "Standard", "Chipped", "TakeOn", "Rebound", "BallRecovery", "Throughball",
]

# Presets de la demo: distancia (m), lateral (m), situación, parte del cuerpo, acción previa.
PRESETS = {
    "A 6 m": (6.0, 0.0, "OpenPlay", "RightFoot", "Pass"),
    "A 30 m": (30.0, 0.0, "OpenPlay", "RightFoot", "Pass"),
    "Penalti": (11.0, 0.0, "Penalty", "RightFoot", "None"),
}

DEFAULT_PLAYER_ID = 2518  # Marcos Llorente, guion de la demo
DEFAULT_SEASON = 2020

st.set_page_config(page_title="xG LaLiga", page_icon="⚽", layout="wide")


# ---------- utilidades ----------


def show_error(exc: ApiError) -> None:
    st.error(str(exc))


def to_understat(distance_m: float, lateral_m: float) -> tuple[float, float]:
    """Metros -> coordenadas normalizadas de Understat, acotadas a [0, 1]."""
    x = 1 - distance_m / PITCH_LENGTH_M
    y = 0.5 + lateral_m / PITCH_WIDTH_M
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def draw_half_pitch(distance_m: float, lateral_m: float, xg: float | None):
    """Medio campo visto desde arriba, con la portería abajo y el punto del tiro."""
    fig, ax = plt.subplots(figsize=(5, 4))
    green, line = "#2e7d32", "white"
    ax.add_patch(Rectangle((-34, 0), 68, 52.5, color=green, zorder=0))
    ax.plot([-34, 34, 34, -34, -34], [0, 0, 52.5, 52.5, 0], color=line, lw=1.5)
    ax.plot([-20.16, -20.16, 20.16, 20.16], [0, 16.5, 16.5, 0], color=line, lw=1.5)  # área grande
    ax.plot([-9.16, -9.16, 9.16, 9.16], [0, 5.5, 5.5, 0], color=line, lw=1.5)  # área pequeña
    ax.plot([-3.66, 3.66], [0, 0], color="white", lw=5)  # portería
    ax.scatter([0], [11], color=line, s=8)  # punto de penalti
    ax.add_patch(Arc((0, 11), 18.3, 18.3, theta1=37, theta2=143, color=line, lw=1.5))
    ax.add_patch(Arc((0, 52.5), 18.3, 18.3, theta1=180, theta2=360, color=line, lw=1.5))
    ax.plot([-3.66, lateral_m, 3.66], [0, distance_m, 0], color="#ffeb3b", lw=1, alpha=0.6)  # ángulo
    ax.scatter([lateral_m], [distance_m], color="#ffeb3b", edgecolor="black", s=140, zorder=5)
    if xg is not None:
        ax.annotate(f"xG {xg:.0%}", (lateral_m, distance_m), xytext=(8, 8), textcoords="offset points",
                    color="white", fontweight="bold")
    ax.set_xlim(-35, 35)
    ax.set_ylim(-2, 53.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def apply_preset(name: str) -> None:
    distance, lateral, situation, shot_type, last_action = PRESETS[name]
    st.session_state.update(
        distance=distance, lateral=lateral, situation=situation, shot_type=shot_type, last_action=last_action
    )
    st.session_state.pop("prediction", None)


@st.cache_data(ttl=60, show_spinner=False)
def cached_players(season: int) -> list[dict]:
    return api_client.players(season)


# ---------- barra lateral: estado de la API ----------

with st.sidebar:
    st.title("⚽ xG LaLiga")
    try:
        health = api_client.health()
        light = "🟢 ok" if health["status"] == "ok" else "🟠 degraded"
        st.subheader(f"API {light}")
        provider = health.get("genai_provider") or "-"
        model = health.get("genai_model")
        st.write(f"**IA:** {provider}" + (f" · `{model}`" if model else ""))
        st.write(f"**Versión:** {health.get('version', '-')}")
        for check, reason in health.get("details", {}).items():
            st.caption(f"❌ {check}: {reason}")
    except ApiError as exc:
        health = None
        st.subheader("API 🔴 sin conexión")
        st.caption(str(exc))
    st.write(f"**API_URL:** `{api_client.API_URL}`")
    st.markdown(f"[Documentación de la API]({api_client.API_URL}/docs)")

tab_shot, tab_player, tab_ops = st.tabs(["🎯 Predice un tiro", "🧑 Jugador + informe IA", "🛠️ Operación"])


# ---------- pestaña 1: predice un tiro ----------

with tab_shot:
    st.header("Predice un tiro")
    st.caption("¿Qué probabilidad tiene de acabar en gol? Ese número es el xG.")

    for key, value in {"distance": 11.0, "lateral": 0.0, "situation": "OpenPlay",
                       "shot_type": "RightFoot", "last_action": "Pass", "minute": 45}.items():
        st.session_state.setdefault(key, value)

    preset_cols = st.columns(len(PRESETS))
    for col, name in zip(preset_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    inputs, result = st.columns([1, 1])
    with inputs:
        distance = st.slider("Distancia a la línea de gol (m)", 0.0, 50.0, step=0.5, key="distance")
        lateral = st.slider("Desplazamiento lateral respecto al centro (m)", -34.0, 34.0, step=0.5, key="lateral",
                            help="Negativo = izquierda, positivo = derecha (mirando a portería)")
        situation = st.selectbox("Situación", SITUATIONS, key="situation")
        shot_type = st.selectbox("Parte del cuerpo", SHOT_TYPES, key="shot_type")
        last_action = st.selectbox("Acción previa", LAST_ACTIONS, key="last_action")
        minute = st.slider("Minuto", 0, 95, key="minute")

        if st.button("Calcular xG", type="primary", width="stretch"):
            x, y = to_understat(distance, lateral)
            shot = {"x": round(x, 4), "y": round(y, 4), "minute": minute, "situation": situation,
                    "shot_type": shot_type, "last_action": None if last_action == "None" else last_action}
            try:
                st.session_state["prediction"] = api_client.predict(shot)
            except ApiError as exc:
                st.session_state.pop("prediction", None)
                show_error(exc)

    with result:
        prediction = st.session_state.get("prediction")
        if prediction:
            route = "regla de penalti" if prediction["source"] == "penalty_rule" else "modelo"
            st.metric("xG", f"{prediction['xg']:.0%}", help=f"{prediction['xg']:.4f}")
            st.write(f"Ruta usada: **{route}**")
            st.caption(f"request_id {prediction['request_id']}")
        st.pyplot(draw_half_pitch(distance, lateral, prediction["xg"] if prediction else None))


# ---------- pestaña 2: jugador + informe IA ----------

with tab_player:
    st.header("Jugador + informe IA")
    season = st.radio("Temporada", [2020, 2019], horizontal=True, format_func=lambda s: f"{s}-{str(s + 1)[-2:]}")

    try:
        player_list = cached_players(season)
    except ApiError as exc:
        player_list = []
        show_error(exc)

    if player_list:
        ids = [p["player_id"] for p in player_list]
        names = {p["player_id"]: f"{p['player']} ({p['shots']} tiros)" for p in player_list}
        default_index = ids.index(DEFAULT_PLAYER_ID) if DEFAULT_PLAYER_ID in ids else 0
        player_id = st.selectbox("Jugador", ids, index=default_index, format_func=names.get)

        try:
            m = api_client.player_metrics(player_id, season)
        except ApiError as exc:
            m = None
            show_error(exc)

        if m:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Tiros", m["shots"])
            c2.metric("Goles", m["goals"])
            c3.metric("xG", f"{m['xg']:.2f}")
            c4.metric("npxG", f"{m['npxg']:.2f}")
            diff = m["np_goals_minus_npxg"]
            c5.metric("Goles sin penalti − npxG", f"{m['np_goals']} − {m['npxg']:.2f}", delta=f"{diff:+.2f}")
            c6.metric("npxG por tiro", f"{m['npxg_per_shot']:.3f}")
            if m["in_sample"]:
                st.warning("Temporada in-sample: el modelo se entrenó con estos tiros, las cifras son optimistas.")

            report_key = f"report_{player_id}_{season}"
            if st.button("Generar informe con IA", type="primary"):
                with st.spinner("Generando informe…"):
                    try:
                        st.session_state[report_key] = api_client.analyze(player_id, season)
                    except ApiError as exc:
                        st.session_state.pop(report_key, None)
                        show_error(exc)

            report = st.session_state.get(report_key)
            if report:
                st.info(report["report"])
                model_name = f" ({report['model']})" if report.get("model") else ""
                st.caption(
                    f"Proveedor {report['provider']}{model_name} · {report['latency_ms']:.0f} ms · "
                    f"request_id {report['request_id']}"
                )


# ---------- pestaña 3: operación ----------

with tab_ops:
    st.header("Operación")
    st.button("Actualizar")

    if health:
        st.subheader("Checks de /health")
        for check, ok in health["checks"].items():
            st.write(f"{'✅' if ok else '❌'} {check}")

    try:
        ops = api_client.metrics()
    except ApiError as exc:
        ops = None
        show_error(exc)

    if ops:
        uptime_min = ops["uptime_s"] / 60
        st.caption(f"Uptime: {uptime_min:.1f} min · en marcha desde {ops['started_at']}")

        model = ops.get("model")
        if model:
            st.subheader("Modelo")
            st.write(f"**{model.get('name') or '-'}** {model.get('params') or ''} · entrenado {model.get('trained_at')}"
                     f" · train {model.get('train_season')} / test {model.get('test_season')}"
                     f" · penalty_xg {model.get('penalty_xg')}")
            test_metrics = model.get("test_metrics") or {}
            if test_metrics:
                table = pd.DataFrame(test_metrics).rename(columns={"model": "Modelo", "xg_understat": "Understat"})
                table.index = table.index.map({"log_loss": "Log loss", "brier": "Brier", "roc_auc": "ROC-AUC"})
                st.dataframe(table, width="content")

        st.subheader("Peticiones por ruta")
        totals = ops["totals"]
        st.caption(f"Total: {totals['count']} peticiones · {totals['error_rate']:.1%} de errores")
        if ops["requests"]:
            rows = [
                {"ruta": route, "peticiones": s["count"], "% errores": f"{s['error_rate']:.1%}",
                 "p50 (ms)": s["p50_ms"], "p95 (ms)": s["p95_ms"]}
                for route, s in ops["requests"].items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.subheader("IA generativa")
        genai = ops["genai"]
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Llamadas", genai["calls"])
        g2.metric("Fallos", genai["failures"])
        g3.metric("p50 (ms)", genai["p50_ms"] if genai["p50_ms"] is not None else "-")
        g4.metric("p95 (ms)", genai["p95_ms"] if genai["p95_ms"] is not None else "-")

        st.subheader("Últimas peticiones")
        if ops["recent"]:
            recent = pd.DataFrame(ops["recent"])
            recent["hora"] = pd.to_datetime(recent["ts"]).dt.strftime("%H:%M:%S")
            st.dataframe(
                recent[["hora", "request_id", "method", "route", "status", "latency_ms"]].rename(
                    columns={"method": "método", "route": "ruta", "status": "estado", "latency_ms": "ms"}
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Todavía no hay peticiones.")
