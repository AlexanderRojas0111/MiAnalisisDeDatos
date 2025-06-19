# Proyecto de Análisis de Datos de Viviendas en California


## 📄 Descripción General del Proyecto

Este proyecto es un análisis de datos y un sistema de modelado de regresión para predecir los valores medianos de las viviendas en los distritos de California, utilizando el famoso [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). El objetivo principal es explorar los datos, preprocesarlos y comparar el rendimiento de diferentes modelos de Machine Learning para esta tarea de predicción numérica.



## 🚀 Características Principales

- **Carga de Datos**: Importa el dataset de California Housing desde `scikit-learn`.
- **Análisis Exploratorio de Datos (EDA)**: Visualizaciones (histogramas, mapas de calor, gráficos de dispersión, box plots) y estadísticas descriptivas.
- **Preprocesamiento**: División en conjuntos de entrenamiento/prueba y normalización de características con `StandardScaler`.
- **Modelado de Regresión**: Implementación y comparación de:
    - Regresión Lineal
    - Árbol de Decisión
    - Bosque Aleatorio
- **Evaluación**: Métricas MAE, MSE, RMSE y R² para comparar modelos.
- **Estructura Modular** (si aplica): Código organizado en módulos para facilitar la mantenibilidad.



## 📂 Estructura del Proyecto

MiAnalisisDeDatos/
├── main.py             # Script principal que orquesta todo el flujo.
├── data_loader.py      # Módulo para cargar el dataset.
├── eda.py              # Módulo para realizar el Análisis Exploratorio de Datos.
├── preprocessor.py     # Módulo para el preprocesamiento de datos (división, normalización).
├── models.py           # Módulo para definir, entrenar y realizar predicciones con los modelos.
├── evaluator.py        # Módulo para evaluar el rendimiento de los modelos.
└── README.md           # Este archivo de descripción del proyecto.
└── requirements.txt    # Lista de dependencias del proyecto.

## 🛠️ Requisitos

Asegúrate de tener Python 3.8+ instalado. Todas las dependencias del proyecto se listan en `requirements.txt`.



## 💻 Instalación y Uso

1. **Clona el repositorio**:
    ```sh
    git clone https://github.com/AlexanderRojas0111/MiAnalisisDeDatos.git
    cd MiAnalisisDeDatos
    ```

2. **Crea un entorno virtual**:
    ```sh
    python -m venv .venv
    ```

3. **Activa el entorno virtual**:
    - **Windows (PowerShell):**
        ```sh
        .venv\Scripts\Activate.ps1
        ```
    - **Windows (CMD):**
        ```sh
        .venv\Scripts\activate.bat
        ```
    - **macOS / Linux:**
        ```sh
        source .venv/bin/activate
        ```

4. **Instala las dependencias**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Ejecuta el script principal**:
    ```sh
    python main.py
    ```



## ⚙️ Personalización (Opcional)

* Puedes modificar la lista de modelos a probar en `main.py`.
* En `preprocessor.py`, puedes optar por seleccionar un subconjunto específico de características (`use_specific_features=True` y `feature_list`) en lugar de usar todas las variables de entrada.





## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.



## 📧 Contacto

[Alexander Rojas Diaz] - (https://www.linkedin.com/in/alexanderrojasdiazenginner/)