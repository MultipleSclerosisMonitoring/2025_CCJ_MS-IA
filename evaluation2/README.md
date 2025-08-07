# 📂 evaluation2

Este directorio contiene los resultados de los experimentos realizados sobre representaciones latentes extraídas con distintos modelos de autoencoder Transformer.

## Estructura

Cada subcarpeta `eval_<MODELO>` contiene los resultados de un modelo distinto entrenado con segmentos de longitud 295.

Por ejemplo:
- `eval_A/`
- `eval_B/`
- ...
- `eval_M/`

## Contenido por carpeta

- **Matrices de confusión** por fold (`fold1` a `fold5`) en formato `.png`.
- **Fichero `.csv`** con las métricas de evaluación globales del modelo.

## Objetivo

Esta evaluación permite comparar la capacidad de separación de clases (walk / no_walk) que logra cada modelo entrenado con representaciones latentes distintas.

Las diferencias entre modelos se deben exclusivamente a su arquitectura, ya que todos usan el mismo dataset de entrada (`len295`).
