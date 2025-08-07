# 📂 latent_images_pca

Este directorio contiene las visualizaciones de las representaciones latentes generadas por los modelos `A` a `D`, proyectadas a dos dimensiones mediante PCA (Análisis de Componentes Principales).

## Estructura de los archivos

Cada imagen tiene el siguiente nombre:

`latent_pca_295_<ID>_fold<FOLD>.png`

- `<ID>` representa el modelo (A, B, C, D...)
- `<FOLD>` indica el fold de validación correspondiente (1 a 5)

## Ejemplo

- `latent_pca_295_A_fold3.png`: representa la proyección PCA 2D de las latentes del modelo A para el fold 3.

## Objetivo

Estas visualizaciones permiten evaluar la separabilidad entre clases (`walk` / `no_walk`) en el espacio latente generado por cada modelo, empleando una técnica lineal como PCA. Sirven como complemento a las visualizaciones con UMAP y al análisis numérico con métricas como el Silhouette Score.
