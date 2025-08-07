# 📂 latent_data

Este directorio contiene las representaciones latentes (`X_latent`) generadas por los distintos modelos de autoencoder Transformer tras el entrenamiento con segmentos de longitud 295.

## Estructura

Cada subcarpeta tiene la siguiente forma:

- `latent_len295_<ID>/`

donde `<ID>` puede ser A, B, C, ..., M, y representa un modelo concreto con una arquitectura específica.

## Contenido de cada subcarpeta

- `fold1/X_latent_data.npz`
- `fold2/X_latent_data.npz`
- ...
- `fold5/X_latent_data.npz`

Cada archivo `.npz` contiene los vectores latentes generados por el encoder para los datos de validación de cada `fold`.

## Uso

Estas representaciones se utilizan como entrada para clasificadores tradicionales (como LDA, QDA, Naive Bayes...) y también para análisis de separabilidad visual mediante técnicas como UMAP o PCA.
