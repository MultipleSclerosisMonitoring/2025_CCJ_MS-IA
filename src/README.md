# 📂 src

Este directorio contiene todo el código fuente del proyecto, organizado por funcionalidades clave.

## Subcarpetas

- `InfluxDBms/`: scripts de conexión y gestión de datos desde bases de datos InfluxDB (no se usan en este experimento, pero están preparados).
- `plotting/`: scripts para análisis visual de resultados y proyecciones latentes.

## Scripts principales

- `analyze_models.py`: compara métricas entre modelos y genera resúmenes visuales o en CSV.
- `encode_latent_transformer.py`: codifica los datos de entrada usando los modelos entrenados (obtiene las representaciones latentes).
- `evaluate_latent_classifier.py`: entrena y evalúa un clasificador binario (`walk` vs `no_walk`) a partir de las representaciones latentes.
- `export_balanced_chunks.py`: exporta segmentos temporales balanceados en formato `.hdf5`, listos para entrenamiento.
- `plot_latents.py`: genera las proyecciones PCA y UMAP de los espacios latentes.
- `run_length_experiment.py`: ejecuta el pipeline completo para comparar longitudes de segmento.
- `train_autoencoder.py`: script de entrenamiento del autoencoder-transformer con validación cruzada.

## Notas

Todos los scripts están preparados para ser ejecutados desde línea de comandos con argumentos configurables. La mayoría de ellos genera carpetas y archivos estructurados automáticamente en función del experimento.
