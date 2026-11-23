# Uso de IA en el desarrollo

Este documento cita las herramientas de IA generativa usadas para **desarrollar** el proyecto, según la política de la asignatura. No describe la IA que forma parte del producto (el informe de scouting de `genai.py`), que está documentada en el README.

## Herramientas

| Herramienta | Proveedor | Para qué |
|---|---|---|
| Claude Code | Anthropic | Asistente de programación en los Proyectos II y III |

## Proyecto II (`v0.2.0`)

### Qué hice yo

- Elegir el dominio, el dataset y el problema: un modelo xG propio comparado con el de Understat.
- Diseñar la arquitectura en `docs/ARQUITECTURA.md`: etapas, features, protocolo de evaluación (split temporal, `GroupKFold` por partido, penaltis aparte), métricas por jugador, contrato de la futura API, decisiones de diseño, limitaciones y plan de entregas.
- Elegir el proveedor de IA generativa (LLM de NVIDIA, recomendado en clase) en lugar del Gemini que tenía en el diseño inicial, y el modelo `nvidia/nemotron-3.5-bolt-e4b` tras probarlo.
- Revisar el código, los resultados del modelo y la documentación generados.

### Qué generó Claude Code

A partir de `docs/ARQUITECTURA.md` y del código del Proyecto I (`loader.py`, `validator.py`, `metrics.py`, `cli.py` y sus tests), Claude Code generó:

- Los módulos `config.py`, `ingesta.py`, `features.py`, `train.py`, `evaluate.py`, `predictor.py` y `genai.py`; la función `player_metrics` de `metrics.py`, y los comandos nuevos de `cli.py`.
- Los tests nuevos (`test_ingesta.py`, `test_features.py`, `test_train.py`, `test_predictor.py`, `test_genai.py` y las ampliaciones de `test_metrics.py` y `test_cli.py`), la fixture del modelo mini en `conftest.py` y `tests/fixtures/shots_mini.csv`.
- El README del Proyecto II, `.env.example`, los cambios en el CI y la sustitución de Gemini por NVIDIA en `docs/ARQUITECTURA.md`.
- La ejecución del pipeline completo, que produjo el modelo de `models/` y la curva de `reports/calibration.png`.

### Cómo se verificó

- 111 tests con pytest y un 99 % de cobertura. El comando del CI se ejecutó en un entorno limpio instalado desde `requirements.txt`.
- Pipeline ejecutado de extremo a extremo con los datos reales. Las métricas del README se obtuvieron de esa ejecución (`models/xg_model.json`).
- Comprobaciones de sentido: el xG baja al alejarse de la portería, la curva de calibración sigue la diagonal y el modelo mejora al baseline sin quedar por encima de Understat.

### Limitaciones del uso de IA

- El código generado sigue las decisiones del documento de arquitectura. Donde la IA propuso detalles concretos (hiperparámetros candidatos, `min_frequency = 20`, el texto del *system prompt*), los revisé antes de aceptarlos.
- Los informes del proveedor `nvidia` los genera un LLM y pueden contener errores. El *system prompt* le obliga a usar solo las cifras recibidas.

## Proyecto III (`v1.0.0-rc` / `v1.0.0`)

### Qué hice yo

- Escribir el plan de desarrollo del Proyecto III con todas las decisiones de diseño cerradas:
  - Streamlit en una única página con tres pestañas, que solo habla con la API por HTTP.
  - FastAPI síncrona con `/health`, `/predict`, `/players`, `/players/{id}/metrics`, `/analyze` y `/metrics`.
  - NVIDIA como proveedor real, el mock solo como doble de pruebas y sin fallback automático.
  - Errores trazables con `request_id` (502, 503 y 504 para la IA).
  - `/health` con checks y 503 si está degradado.
  - Métricas en memoria.
  - Docker, docker-compose y Render, y el job de Docker en el CI.
  - El guion de la demo.
- Probar el proveedor real de NVIDIA con mi API key.
- Revisar y mergear el PR, desplegar en Render y crear los tags.

### Qué generó Claude Code

A partir de ese plan, Claude Code generó:

- La API (`src/football_shots/api/`: `main.py`, `schemas.py`, `monitoring.py`) y sus tests (`tests/test_api.py`), reutilizando el núcleo de los Proyectos I y II sin cambiarlo. El único cambio en el núcleo es extraer los `Literal` de `validator.py` a alias de tipo, para que la API acepte las mismas categorías.
- El dashboard (`dashboard/app.py`, `dashboard/api_client.py`).
- `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `render.yaml`, `scripts/smoke_test.sh` y el job `docker` del CI.
- Los cambios de configuración (`config.py`, `.env.example`), la sección del Proyecto III del README y la actualización de `docs/ARQUITECTURA.md`.

### Cómo se verificó

- Tests de pytest de toda la API, sin red, con el `MockClient` y clientes falsos que simulan timeouts y errores del proveedor.
- Imagen Docker construida y arrancada en modo mock, con `scripts/smoke_test.sh` en verde (lo mismo que hace el CI).
- El guion de la demo recorrido contra la API local.
