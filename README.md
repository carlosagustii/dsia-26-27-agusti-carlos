# Proyecto I — Online Shoppers Purchasing Intention

## Dataset

Para este proyecto se utiliza el dataset **Online Shoppers Purchasing Intention Dataset**, disponible públicamente en el **UCI Machine Learning Repository**.

- **URL de origen:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Fecha de descarga:** 28/09/2026
- **Fichero utilizado:** `online_shoppers_intention.csv`
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Condiciones de uso:** se permite copiar, redistribuir y adaptar el dataset, siempre que se atribuya correctamente la fuente.
- **DOI:** https://doi.org/10.24432/C5F88Q

## Descripción

El dataset contiene información sobre sesiones de usuarios en una tienda online. Cada fila representa una sesión de navegación y recoge datos sobre las páginas visitadas, tiempo de navegación, tipo de visitante, tráfico y otras variables relacionadas con el comportamiento del usuario.

La variable `Revenue` indica si la sesión terminó en una compra.

## Especificaciones

- **Número de registros:** 12.330
- **Número de columnas:** 18
- **Valores nulos en el fichero:** 0
- **Tipo de problema indicado por UCI:** clasificación / clustering
- **Variable objetivo:** `Revenue`

## Variables

| Variable | Descripción |
|---|---|
| `Administrative` | Número de páginas administrativas visitadas |
| `Administrative_Duration` | Tiempo empleado en páginas administrativas |
| `Informational` | Número de páginas informativas visitadas |
| `Informational_Duration` | Tiempo empleado en páginas informativas |
| `ProductRelated` | Número de páginas de producto visitadas |
| `ProductRelated_Duration` | Tiempo empleado en páginas de producto |
| `BounceRates` | Tasa de rebote |
| `ExitRates` | Tasa de salida |
| `PageValues` | Valor de las páginas visitadas |
| `SpecialDay` | Cercanía de la visita a una fecha especial |
| `Month` | Mes de la sesión |
| `OperatingSystems` | Sistema operativo |
| `Browser` | Navegador |
| `Region` | Región |
| `TrafficType` | Tipo de tráfico |
| `VisitorType` | Tipo de visitante |
| `Weekend` | Indica si la sesión ocurrió en fin de semana |
| `Revenue` | Indica si la sesión terminó en compra |

## Fuente

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository.
