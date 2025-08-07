# 📂 src

Este directorio contiene todo el código fuente del proyecto, organizado por funcionalidades clave.

---

## ▶️ Orden recomendado de ejecución

Para reproducir completamente el flujo de trabajo del proyecto, se recomienda ejecutar los siguientes scripts en **este orden**:

1. **`export_balanced_chunks.py`**  
   Segmenta los datos brutos y crea conjuntos balanceados (`walk` / `no_walk`) en formato `.hdf5`, listos para el entrenamiento.

2. **`train_autoencoder.py`**  
   Entrena el modelo autoencoder-transformer usando validación cruzada. Se generan los modelos por fold y se guardan métricas.

3. **`analyze_models.py`**  
   Analiza los resultados del entrenamiento de los autoencoders, comparando métricas como loss, silhouette score y tiempos por fold.

4. **`evaluate_latent_classifier.py`**  
   Utiliza las representaciones latentes generadas por los encoders para entrenar y evaluar un clasificador binario (`walk` vs `no_walk`), con métricas de validación.

5. **`encode_latent_transformer.py`**  
   Aplica el encoder entrenado para obtener las representaciones latentes de nuevos segmentos. Estas representaciones se usarán para análisis o clasificación.

6. **`plot_latents.py`**  
   Visualiza los espacios latentes usando técnicas de reducción de dimensionalidad como PCA y UMAP. Permite explorar si las clases son separables en el espacio embebido.

---

## 🏁 Fase de producción (modelo final)

Estos scripts están diseñados para la **fase final de despliegue** del modelo, una vez que se ha completado la experimentación:

7. **`train_logistic_classifier.py`**  
   Entrena un clasificador logístico (`LogisticRegression`) sobre las representaciones latentes extraídas por el encoder. Este modelo binario es ligero y rápido, ideal para inferencia en producción.

8. **`inference.py`**  
   Permite ejecutar inferencias sobre nuevos datos no etiquetados. El script realiza el preprocesamiento (segmentación + escalado), obtiene las representaciones latentes con el encoder y predice `walk / no_walk` con el clasificador logístico. Guarda los resultados en un archivo `.csv`.

---

## 📁 Subcarpetas

- **`InfluxDBms/`**: Scripts para conectar y extraer datos desde bases de datos InfluxDB. No se usaron en este experimento concreto, pero están disponibles para futuras extensiones.
- **`plotting/`**: Scripts complementarios para generar gráficos, comparar modelos, o visualizar transformaciones.

---

## ⚙️ Scripts principales

| Script                          | Función principal                                                                 |
|---------------------------------|-----------------------------------------------------------------------------------|
| `export_balanced_chunks.py`     | Exporta segmentos balanceados de series temporales (`walk` / `no_walk`).         |
| `train_autoencoder.py`          | Entrena autoencoders con validación cruzada sobre los datos segmentados.         |
| `analyze_models.py`             | Compara resultados entre modelos entrenados (per fold o por arquitectura).       |
| `evaluate_latent_classifier.py` | Evalúa clasificadores binarios sobre espacios latentes generados por el encoder. |
| `encode_latent_transformer.py`  | Codifica datos con el encoder entrenado para obtener representaciones latentes.  |
| `plot_latents.py`               | Visualiza proyecciones de los espacios latentes (PCA, UMAP).                     |
| `train_logistic_classifier.py`  | Entrena un clasificador logístico binario sobre las representaciones latentes.   |
| `inference.py`                  | Realiza predicciones (`walk` / `no_walk`) sobre datos nuevos no etiquetados.     |

---

## 📝 Notas

- Todos los scripts están diseñados para ser ejecutados desde la línea de comandos.
- Aceptan argumentos configurables y generan carpetas de salida automáticamente.
- Los resultados (modelos, métricas, CSVs, visualizaciones) se almacenan en directorios organizados por experimento.
