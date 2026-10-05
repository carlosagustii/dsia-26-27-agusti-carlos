# Proyecto I — Football Shots (xG)

## Dataset

Para este proyecto se utiliza el dataset **Football Shots Dataset: Top 5 European Leagues**, publicado en Kaggle.

- **URL de origen:** https://www.kaggle.com/datasets/mat126/shots-dataset-for-footballsoccer
- **Fuente original de los datos:** Understat (https://understat.com)
- **Fichero utilizado:** `shots_dataset_cleaned.csv` (completo, no se sube al repo) y `Datos/shots_sample.csv` (muestra)
- **Licencia:** Database Contents License (DbCL) v1.0
- **Condiciones de uso:** uso educativo y no comercial; los datos proceden de Understat.

## Descripción

El dataset contiene los tiros realizados en las cinco grandes ligas europeas (Premier League, LaLiga, Serie A, Bundesliga y Ligue 1) entre las temporadas 2014/15 y 2020/21. Cada fila representa un tiro e incluye la posición desde la que se realizó (coordenadas X e Y normalizadas entre 0 y 1), el resultado del tiro y el valor de xG (*expected goals*) calculado por Understat.

El objetivo del proyecto es validar la calidad de estos datos y calcular métricas sobre la eficacia de los tiros. En proyectos posteriores servirá de base para entrenar un modelo propio de xG.

## Especificaciones

- **Número de registros (completo):** ~400.000
- **Número de registros (muestra):** 20832
- **Número de columnas:** 19
- **Valores nulos en el fichero:** 2436
- **Variable objetivo (proyectos II/III):** si el tiro acaba en gol

## Muestra de datos

El CSV completo es demasiado grande para GitHub, así que en el repo solo se incluye una muestra aleatoria reproducible:

```bash
# 1. Descargar shots_dataset_cleaned.csv de Kaggle y guardarlo en Datos/
# 2. Generar la muestra
python scripts/crear_muestra.py --input Datos/shots_dataset_cleaned.csv
```

## Variables

| Variable | Descripción |
|---|---|
| `shot_id` | Identificador único del tiro |
| `match_id` | Identificador del partido |
| `date` | Fecha y hora del partido |
| `season` | Temporada (año de inicio): 2019 o 2020 |
| `league` | Liga (siempre `Liga`, LaLiga) |
| `h_team` | Equipo local |
| `a_team` | Equipo visitante |
| `h_a` | Si el tirador juega en casa (`h`) o fuera (`a`) |
| `player` | Nombre del jugador que tira |
| `player_id` | Identificador del jugador |
| `minute` | Minuto del partido en que se produce el tiro (0–100) |
| `x` | Coordenada horizontal del tiro, normalizada 0–1 (1 = línea de gol rival) |
| `y` | Coordenada vertical del tiro, normalizada 0–1 (ancho del campo) |
| `situation` | Contexto de la jugada: `OpenPlay`, `FromCorner`, `SetPiece`, `DirectFreekick`, `Penalty` |
| `shot_type` | Parte del cuerpo con la que se tira: `RightFoot`, `LeftFoot`, `Head`, `OtherBodyPart` |
| `last_action` | Acción previa al tiro (pase, centro, regate…); 28 categorías, con nulos |
| `preferred_foot` | Pie dominante del jugador |
| `xg_understat` | Goles esperados (xG) que asigna Understat al tiro (0–1) |
| `is_goal` | **Variable objetivo**: 1 si el tiro acaba en gol, 0 si no |


## Fuente

Football Shots Dataset: Top 5 European Leagues. Kaggle (mat126). Datos de Understat.
