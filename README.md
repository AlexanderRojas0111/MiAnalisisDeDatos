# Proyecto de Análisis de Datos de Viviendas en California

## 📄 Descripción General del Proyecto

Este proyecto realiza un análisis de datos y modelado de regresión para predecir los valores medianos de las viviendas en los distritos de California, utilizando el [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). El objetivo es explorar los datos, preprocesarlos y comparar el rendimiento de diferentes modelos de Machine Learning, incluyendo técnicas clásicas y redes neuronales.

---

## 🚀 Características Principales

- **Carga de Datos**: Importa el dataset de California Housing desde `scikit-learn`.
- **Análisis Exploratorio de Datos (EDA)**: Visualizaciones (histogramas, mapas de calor, gráficos de dispersión, box plots) y estadísticas descriptivas, todas exportadas automáticamente como archivos `.png` en la carpeta `charts_statistical_data`.
- **Preprocesamiento**: División en conjuntos de entrenamiento/prueba y normalización de características con `StandardScaler`.
- **Modelado de Regresión**: Implementación y comparación de:
    - Regresión Lineal
    - Árbol de Decisión
    - Bosque Aleatorio
    - Red Neuronal (MLP con TensorFlow/Keras)
- **Evaluación**: Métricas MAE, MSE, RMSE y R² para comparar modelos.
- **Exportación de Resultados**: 
    - Todas las gráficas y tablas de resultados se almacenan automáticamente en la carpeta `charts_statistical_data` como archivos `.png` y `.csv`.
    - Comparación gráfica de modelos y tabla resumen exportadas para su uso en informes.
- **Estructura Modular**: Código organizado en módulos para facilitar la mantenibilidad y reutilización.
- **Automatización de Reportes**: Incluye un módulo `report.py` para generar y visualizar reportes de resultados de manera independiente.

---

## 📂 Estructura del Proyecto

```
MiAnalisisDeDatos/
├── main.py                  # Script principal que orquesta todo el flujo.
├── data_loader.py           # Módulo para cargar el dataset.
├── eda.py                   # Módulo para realizar el Análisis Exploratorio de Datos.
├── preprocessor.py          # Módulo para el preprocesamiento de datos (división, normalización).
├── models.py                # Modelos clásicos de ML.
├── evaluator.py             # Módulo para evaluar el rendimiento de los modelos.
├── neural_network_model.py  # Módulo para la red neuronal.
├── report.py                # Automatización de exportación de tablas y gráficas.
├── charts_statistical_data/ # Carpeta donde se guardan todos los .png y .csv generados.
├── requirements.txt         # Lista de dependencias del proyecto.
└── README.md                # Este archivo de descripción del proyecto.
```

---

## 🛠️ Requisitos

Asegúrate de tener Python 3.8+ instalado. Todas las dependencias del proyecto se listan en `requirements.txt` e incluyen:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

---

## 💻 Instalación y Uso

1. **Clona el repositorio**:
    ```sh
    git clone https://github.com/AlexanderRojas0111/MiAnalisisDeDatos.git
    cd MiAnalisisDeDatos
    ```

2. **Crea y activa un entorno virtual**:
    ```sh
    python -m venv .venv
    .venv\Scripts\Activate.ps1   # En PowerShell de Windows
    ```

3. **Instala las dependencias**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Ejecuta el script principal**:
    ```sh
    python main.py
    ```
    Esto generará automáticamente todos los gráficos y tablas en la carpeta `charts_statistical_data`.

5. **Genera o visualiza reportes de resultados** (opcional):
    ```sh
    python report.py
    ```
    Esto exportará la tabla comparativa y la gráfica de comparación de modelos a la carpeta `charts_statistical_data`.

---

## 📊 Resultados y Reportes

- Todas las gráficas (EDA, comparación de modelos, real vs predicción) y tablas de resultados se almacenan en la carpeta `charts_statistical_data`.
- Puedes incluir estos archivos en tus informes o presentaciones.
- El módulo `report.py` permite regenerar la gráfica comparativa y la tabla resumen a partir del archivo `resultados_modelos.csv`.

---

## ⚙️ Personalización (Opcional)

- Puedes modificar la lista de modelos a probar en `main.py`.
- En `preprocessor.py`, puedes optar por seleccionar un subconjunto específico de características (`use_specific_features=True` y `feature_list`) en lugar de usar todas las variables de entrada.
- Puedes agregar nuevos modelos o métricas fácilmente gracias a la estructura modular.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

[Alexander Rojas Diaz](https://www.linkedin.com/in/alexanderrojasdiazenginner/)