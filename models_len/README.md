# 📂 models_len

Este directorio contiene todos los modelos entrenados, organizados por longitud del segmento de entrada y configuración del experimento.

## Estructura

Cada subcarpeta representa una configuración distinta:

- `models_len150`, `models_len200`, `models_len240`, `models_len295`, `models_len350`: modelos base entrenados con distintas longitudes de segmento.
- `models_len295_A` a `models_len295_M`: modelos entrenados con distintas configuraciones del transformer (variando hiperparámetros como número de bloques, tamaño de cabeza, dimensiones del feedforward, etc.).

## Archivos adicionales

- `comparison_plot.png`: gráfico comparativo entre configuraciones de modelos.
- `summary_all_models.csv`: resumen global con las métricas principales de todos los modelos evaluados.
- `summary_lengths.txt`: texto resumen de los resultados por longitud de segmento.
- `summary_plot.png`: visualización complementaria de rendimiento por longitud.

## Contenido de las carpetas de modelos

Dentro de cada carpeta encontrarás:
- Los ficheros `.keras` del encoder.
- El scaler `.pkl` utilizado para estandarizar los datos.
- (Opcional) Otros scripts o metadatos asociados al entrenamiento.

## Propósito

Estos modelos se utilizan para codificar los datos en un espacio latente y posteriormente clasificarlos como `walk` o `no_walk`.
