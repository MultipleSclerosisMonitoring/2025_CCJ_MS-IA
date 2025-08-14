Introducción
============

Este proyecto de fin de grado tiene como objetivo desarrollar un sistema de identificación de patrones de marcha (gait identification) basado en inteligencia artificial. Para ello, se ha trabajado con datos recogidos por los calcetines inteligentes **Sensoria Smart Socks**, y se han almacenado directamente en la base de datos **InfluxDB**.

Los principales objetivos del proyecto son:

- Preprocesar los datos crudos obtenidos de los sensores.
- Extraer representaciones latentes usando modelos **autoencoder**.
- Clasificar los segmentos en categorías como *walk / no walk* mediante modelos supervisados.
- Visualizar los resultados con técnicas de reducción de dimensión como PCA o UMAP.
- Generar documentación técnica con **Sphinx** y **ReadTheDocs**.

---

Estructura del código fuente (`src`)
====================================

Este proyecto está organizado en módulos independientes que cubren todo el ciclo de vida de los datos: desde su adquisición hasta la inferencia con nuevos datos. A continuación se detalla cada etapa del flujo:

1. Obtención, etiquetado y preprocesado de datos
------------------------------------------------

- ``InfluxDBms/``: contiene funciones para conectarse a una base de datos InfluxDB y exportar los datos recogidos por los sensores. Incluye limpieza básica y funciones auxiliares de visualización.

- ``main.py``: ejecuta el flujo completo desde la lectura de los datos crudos hasta la generación de ventanas etiquetadas. Es el punto de entrada principal para el tratamiento inicial del dataset.

- ``export_balanced_chunks.py``: toma los datos crudos ya etiquetados y los divide en segmentos balanceados de la misma longitud, listos para ser usados en el entrenamiento. Garantiza que `walk` y `no_walk` estén igualmente representados.

2. Entrenamiento de modelos
---------------------------

- ``train_autoencoder.py``: carga los segmentos preprocesados desde un archivo ``.hdf5``, aplica escalado con ``StandardScaler`` y entrena un **autoencoder Transformer** configurable (número de bloques, cabezas, tamaño de representación...). Guarda tanto los pesos del modelo como los logs del entrenamiento.

3. Evaluación de modelos
------------------------

- ``evaluate_latent_classifier.py``: evalúa distintos clasificadores entrenados sobre las representaciones latentes generadas por el autoencoder. Calcula métricas como accuracy, F1-score o matriz de confusión para distintos enfoques y variantes.

4. Codificación y representación de latentes
--------------------------------------------

- ``encode_latent_transformer.py``: transforma los segmentos originales en vectores latentes a través del encoder del autoencoder previamente entrenado. Es clave para alimentar clasificadores posteriores o visualizar la estructura de los datos.

- ``plot_latents.py``: aplica técnicas de reducción de dimensionalidad como **PCA** o **UMAP** sobre las representaciones latentes y genera visualizaciones que permiten explorar la separabilidad entre clases (``walk`` / ``no_walk``).

5. Clasificación e inferencia con nuevos datos ("producción")
--------------------------------------------------------------

- ``train_logistic_classifier.py``: entrena un clasificador logístico simple sobre los vectores latentes, utilizando validación cruzada para comparar distintas configuraciones y seleccionar el modelo final.

- ``inference.py``: este es el script clave para el despliegue. Permite aplicar el modelo ya entrenado sobre datos "frescos", es decir, sin etiquetas. Carga el modelo K (el mejor evaluado), el escalador y el encoder, y devuelve para cada segmento una predicción binaria (``walk`` / ``no_walk``) y su correspondiente representación latente.

Conclusión
----------

Este flujo modular garantiza una **trazabilidad completa del dato**, permitiendo reproducir cada paso de forma aislada o en conjunto, adaptarlo a nuevos sujetos o condiciones, y escalarlo a producción con facilidad.
