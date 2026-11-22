# Uso de IA en el desarrollo

Este documento cita las herramientas de IA generativa usadas para **desarrollar** el proyecto, según la política de la asignatura. No describe la IA que forma parte del producto (el informe de scouting de `genai.py`), que está documentada en el README.

## Herramientas

| Herramienta | Proveedor | Para qué |
|---|---|---|
| Claude Code | Anthropic | Asistente de programación en el Proyecto II |

## Proyecto II (`v0.2.0`)

### Qué hice yo

- Elegir el dominio, el dataset y el problema: un modelo xG propio comparado con el de Understat.
- Diseñar la arquitectura en `docs/ARQUITECTURA.md`: etapas, features, protocolo de evaluación (split temporal, `GroupKFold` por partido, penaltis aparte), métricas por jugador, contrato de la futura API, decisiones de diseño, limitaciones y plan de entregas.
- Elegir el proveedor de IA generativa (LLM de NVIDIA, recomendado en clase) en lugar del Gemini que tenía en el diseño inicial.
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
