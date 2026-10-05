# Proyecto I — Football Shots (xG)

## Dataset

Para este proyecto se utiliza el dataset **Football Shots Dataset: Top 5 European Leagues**, publicado en Kaggle.

- **URL de origen:** https://www.kaggle.com/datasets/mat126/shots-dataset-for-footballsoccer
- **Fuente original de los datos:** Understat (https://understat.com)
- **Fecha de descarga:** <!-- TODO: dd/mm/2026 -->
- **Fichero utilizado:** `shots_dataset_cleaned.csv` (completo, no se sube al repo) y `Datos/shots_sample.csv` (muestra)
- **Licencia:** <!-- TODO: copiar la licencia que indica la página de Kaggle -->
- **Condiciones de uso:** uso educativo y no comercial; los datos proceden de Understat.

## Descripción

El dataset contiene los tiros realizados en las cinco grandes ligas europeas (Premier League, LaLiga, Serie A, Bundesliga y Ligue 1) entre las temporadas 2014/15 y 2020/21. Cada fila representa un tiro e incluye la posición desde la que se realizó (coordenadas X e Y normalizadas entre 0 y 1), el resultado del tiro y el valor de xG (*expected goals*) calculado por Understat.

El objetivo del proyecto es validar la calidad de estos datos y calcular métricas sobre la eficacia de los tiros. En proyectos posteriores servirá de base para entrenar un modelo propio de xG.

## Especificaciones

- **Número de registros (completo):** ~400.000
- **Número de registros (muestra):** 20.000
- **Número de columnas:** <!-- TODO -->
- **Valores nulos en el fichero:** <!-- TODO -->
- **Variable objetivo (proyectos II/III):** si el tiro acaba en gol

## Muestra de datos

El CSV completo es demasiado grande para GitHub, así que en el repo solo se incluye una muestra aleatoria reproducible:

```bash
# 1. Descargar shots_dataset_cleaned.csv de Kaggle y guardarlo en Datos/
# 2. Generar la muestra
python scripts/crear_muestra.py --input Datos/shots_dataset_cleaned.csv
```

## Variables

<!-- TODO: completar con las columnas reales del CSV -->

| Variable | Descripción |
|---|---|
| | |

## Fuente

Football Shots Dataset: Top 5 European Leagues. Kaggle (mat126). Datos de Understat.
