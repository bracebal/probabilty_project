# Cálculo de Modelo de Regresión

Este repositorio contiene el código y la documentación del proyecto para el cálculo de un modelo de regresión.

**Estado del Proyecto:**

Durante el desarrollo de este proyecto, se lograron cumplir satisfactoriamente los objetivos específicos relacionados con:

* **Comprensión del problema**
* **Entendimiento de los datos**
* **Preparación y transformación de los datos**
* **Modelado de la Regresión Lineal MCO**

**Objetivos No Alcanzados:**

Es importante destacar que, debido a la complejidad inesperada del proyecto y el desconocimiento en el manejo de grandes datos, no se pudieron completar todos los objetivos inicialmente planteados. 

**Cómo Utilizar:**

Este conjunto de códigos forma un flujo de trabajo completo para el análisis y modelado de datos. El proceso comienza con `cargaDatos.py`, que se encarga de cargar los datos crudos desde un archivo Excel, realizar una limpieza inicial (manejo de valores numéricos, fechas y metadatos), y generar visualizaciones exploratorias. A continuación, `preparaDatos.py` divide los datos en conjuntos de entrenamiento y prueba, elimina filas con valores faltantes en la variable objetivo, filtra columnas no numéricas y aplica escalado estándar a las características. Finalmente, `modelaEvalua.py` entrena y evalúa el modelo de regresión Lineal. Los scripts están diseñados para ejecutarse secuencialmente, compartiendo datos a través de archivos CSV y JSON intermedios, lo que permite un flujo reproducible y modular.

**Licencia:**

None