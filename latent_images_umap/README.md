# 📂 latent_images_umap

Este directorio contiene visualizaciones bidimensionales de las representaciones latentes generadas por los distintos modelos (`A` a `F`), proyectadas con la técnica UMAP (Uniform Manifold Approximation and Projection).

## Estructura de los archivos

Cada imagen tiene el siguiente formato:

`latent_umap_295_<ID>_fold<FOLD>.png`

- `<ID>` representa el modelo (A, B, C, D, E, F...)
- `<FOLD>` indica el fold de validación correspondiente (1 a 5)

## Ejemplo

- `latent_umap_295_C_fold3.png`: corresponde a la proyección UMAP 2D de las representaciones latentes del modelo C en el fold 3.

## Objetivo

Estas visualizaciones permiten evaluar visualmente la separabilidad entre clases (`walk` / `no_walk`) en el espacio latente. UMAP es una técnica no lineal, ideal para identificar agrupaciones o patrones complejos que pueden no ser evidentes con métodos lineales como PCA.

## Comparación con PCA

A diferencia de las imágenes contenidas en `latent_images_pca`, estas proyecciones UMAP pueden capturar relaciones no lineales entre las representaciones, y por tanto revelar una mejor o diferente separabilidad entre clases.
