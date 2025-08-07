# 📂 evaluation

Este directorio contiene los resultados de las evaluaciones de los modelos entrenados directamente sobre las representaciones latentes (`X_latent`) para distintas longitudes de segmento.

## Estructura

Cada subcarpeta corresponde a una longitud de segmento específica:

- `eval_len150/`
- `eval_len200/`
- `eval_len240/`
- `eval_len295/`
- `eval_len350/`

## Contenido por carpeta

Cada subcarpeta incluye:

- **Matrices de confusión** por fold (`fold1` a `fold5`) en formato `.png`.
- **Resultados agregados** (`.csv`) con las métricas de clasificación: F1-score, accuracy, precision, recall, etc.

## Objetivo

Esta evaluación permite comparar el rendimiento del clasificador basado en representaciones latentes en función de la longitud del segmento.
