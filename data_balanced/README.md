# 📂 data_balanced

Este directorio contiene los datasets equilibrados (misma proporción de clases `walk` y `no_walk`) utilizados durante el entrenamiento y validación de los modelos.

## Contenido

Cada archivo `.hdf5` representa un conjunto de segmentos temporales de distinta longitud, todos muestreados a 50 Hz.

- `dataset_balanced_len150_50Hz.hdf5`
- `dataset_balanced_len200_50Hz.hdf5`
- `dataset_balanced_len240_50Hz.hdf5`
- `dataset_balanced_len295_50Hz.hdf5`
- `dataset_balanced_len350_50Hz.hdf5`

## Formato

Cada archivo contiene:
- `X`: matriz de características de forma `(n_segmentos, n_tiempos, n_features)`
- `y`: etiquetas binarias (`0` = no_walk, `1` = walk)

Estos archivos fueron generados tras el preprocesamiento y balanceo de los datos brutos recogidos por los calcetines Sensoria Smart Socks
