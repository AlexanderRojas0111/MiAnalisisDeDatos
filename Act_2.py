# --- 0. Importar Librerías Necesarias ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Necesario para np.sqrt en el cálculo de RMSE

# Librerías de Scikit-learn para el dataset, división de datos, modelos y métricas
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Para normalización
from sklearn.linear_model import LinearRegression # Modelo de Regresión Lineal
from sklearn.tree import DecisionTreeRegressor # Modelo de Árbol de Decisión
from sklearn.ensemble import RandomForestRegressor # Modelo de Bosque Aleatorio

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Cargar el Dataset de California Housing ---
print("--- 1. Carga del Dataset California Housing ---")
housing = fetch_california_housing(as_frame=True)
df = housing.frame # El DataFrame 'df' ahora contiene todas las variables, incluyendo el target

# Mostrar información básica del DataFrame
print("\nPrimeras 5 filas del DataFrame completo:")
print(df.head())
print("\nInformación general del DataFrame (tipos de datos y nulos):")
df.info()
print("\nEstadísticas descriptivas básicas del DataFrame:")
print(df.describe())

# --- 2. Análisis Exploratorio de Datos (EDA) ---
print("\n--- 2. Análisis Exploratorio de Datos (EDA) ---")

# 2.1 Verificar valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("Este dataset no contiene valores nulos, lo que facilita el preprocesamiento.")

# 2.2 Visualizar la distribución de variables (Histogramas)
print("\nGenerando histogramas de distribución para todas las variables...")
df.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histogramas de Distribución de Todas las Variables', y=1.02)
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Ajuste para evitar solapamiento de título
plt.show()

# Histograma específico de la variable objetivo (MedHouseVal)
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title('Distribución del Valor Medio de la Vivienda (MedHouseVal)')
plt.xlabel('Valor Medio de la Vivienda (en cientos de miles $)')
plt.ylabel('Frecuencia')
plt.show()
print("Observamos el 'capping' o tope en el valor medio de la vivienda ($500,000).")

# 2.3 Explorar relaciones entre variables (Matriz de Correlación y Heatmap)
print("\nCalculando y visualizando la Matriz de Correlación...")
corr_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlación de Variables')
plt.show()

# Mostrar correlaciones con la variable objetivo ordenadas
print("\nCorrelación de cada variable con MedHouseVal (variable objetivo):")
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
print("El Ingreso Medio (MedInc) es la variable más fuertemente correlacionada positivamente.")

# 2.4 Gráficos de dispersión para las relaciones clave
print("\nGenerando gráficos de dispersión para visualizar relaciones clave...")
plt.figure(figsize=(18, 6)) # Aumentamos un poco la altura para mejor visualización

# Ingreso Medio vs. Valor de Vivienda (correlación positiva fuerte)
plt.subplot(1, 3, 1) # 1 fila, 3 columnas, primer gráfico
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.1) # 'alpha' para transparencia en muchos puntos
plt.title('Ingreso Medio (MedInc) vs. Valor Medio de Vivienda')
plt.xlabel('Ingreso Medio (x10,000 $)')
plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

# Número Promedio de Habitaciones vs. Valor de Vivienda
plt.subplot(1, 3, 2)
sns.scatterplot(x='AveRooms', y='MedHouseVal', data=df, alpha=0.1)
plt.title('Número Promedio de Habitaciones vs. Valor Medio de Vivienda')
plt.xlabel('Número Promedio de Habitaciones')
plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

# Edad Media de la Vivienda vs. Valor de Vivienda
plt.subplot(1, 3, 3)
sns.scatterplot(x='HouseAge', y='MedHouseVal', data=df, alpha=0.1)
plt.title('Edad Media de la Vivienda vs. Valor Medio de Vivienda')
plt.xlabel('Edad Media de la Vivienda')
plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

plt.tight_layout() # Ajusta el diseño para evitar superposiciones
plt.show()

# 2.5 Box Plots para detección de Outliers
print("\nGenerando Box Plots para detección de Outliers en cada variable...")
plt.figure(figsize=(15, 12)) # Ajustamos tamaño para más variables
plt.suptitle('Box Plots para Detección de Outliers en Variables', y=1.02)

# Iteramos sobre todas las columnas para generar un box plot para cada una
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1) # Cuadrícula de 3x3 (8 features + 1 target = 9 plots)
    sns.boxplot(y=df[column])
    plt.title(column)
    plt.ylabel('') # Elimina la etiqueta del eje y si el título ya es claro

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()
print("Variables como 'Population' y 'AveOccup' muestran una considerable cantidad de outliers.")

# --- 3. Preparación de Datos para el Modelado ---
print("\n--- 3. Preparación de Datos para el Modelado ---")

# 3.1 Separar Variables de Entrada (X) y Salida (y)
X = df.drop(columns=['MedHouseVal']) # Todas las columnas excepto MedHouseVal son inputs
y = df['MedHouseVal'] # MedHouseVal es la variable objetivo

# Opcional: Si necesitas seleccionar específicamente 6 variables de entrada:
# columnas_seleccionadas_X = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
# X = df[columnas_seleccionadas_X]
# print(f"\nVariables de entrada seleccionadas (6): {X.columns.tolist()}")


# 3.2 Dividir el Dataset en conjuntos de entrenamiento y prueba
print("\nDividiendo el dataset en conjuntos de entrenamiento (80%) y prueba (20%)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(f"Tamaño del conjunto de entrenamiento (X_train): {X_train.shape}")
print(f"Tamaño del conjunto de prueba (X_test): {X_test.shape}")


# 3.3 Normalizar las variables de entrada (X)
print("\nNormalizando las variables de entrada (X) usando StandardScaler...")
scaler = StandardScaler()

# Ajustar el escalador SÓLO con los datos de entrenamiento y luego transformar X_train
X_train_scaled = scaler.fit_transform(X_train)

# Usar el escalador ajustado (con los parámetros de X_train) para transformar X_test
X_test_scaled = scaler.transform(X_test)
print("Variables de entrada normalizadas.")


# --- 4. Elegir, Entrenar y Evaluar Modelos de Regresión ---
print("\n--- 4. Elegir, Entrenar y Evaluar Modelos de Regresión ---")

# --- Paso 4.1: Elegir el Modelo ---
# Descomenta SOLO UNO de los siguientes bloques de modelo para seleccionarlo.

# Opción 1: Regresión Lineal
# model = LinearRegression()
# model_name = "Regresión Lineal"

# Opción 2: Árbol de Decisión para Regresión
# model = DecisionTreeRegressor(random_state=42)
# model_name = "Árbol de Decisión para Regresión"

# Opción 3: Bosque Aleatorio para Regresión (Recomendado para buen rendimiento)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos los núcleos disponibles
model_name = "Bosque Aleatorio para Regresión"

print(f"\nModelo seleccionado para esta ejecución: {model_name}")

# --- Paso 4.2: Entrenar el Modelo ---
print(f"Entrenando el modelo '{model_name}' con los datos de entrenamiento escalados...")
model.fit(X_train_scaled, y_train) # ¡Importante: usamos los datos escalados aquí!
print(f"Modelo '{model_name}' entrenado exitosamente.")

# --- Paso 4.3: Realizar Predicciones ---
print(f"\nRealizando predicciones en el conjunto de prueba con '{model_name}'...")
y_pred = model.predict(X_test_scaled) # ¡Importante: usamos los datos de prueba escalados aquí!
print("Predicciones generadas.")

# Mostrar algunas de las primeras predicciones junto a los valores reales
print("\nPrimeras 10 Predicciones vs. Valores Reales (MedHouseVal):")
for i in range(10):
    print(f"Predicción: {y_pred[i]:.2f}, Real: {y_test.iloc[i]:.2f}")


# --- Paso 4.4: Evaluar el Modelo ---
print(f"\n--- 5. Evaluación del Modelo '{model_name}' ---")

# Calcular métricas de evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) # Raíz Cuadrada del Error Cuadrático Medio
r2 = r2_score(y_test, y_pred) # Coeficiente de Determinación (R-squared)

print(f"Error Absoluto Medio (MAE): {mae:.3f} (Promedio de error absoluto en $x100,000)")
print(f"Error Cuadrático Medio (MSE): {mse:.3f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.3f} (Error promedio en la misma unidad que el target)")
print(f"Coeficiente de Determinación (R-squared): {r2:.3f} (Proporción de varianza explicada, 0-1)")

# Interpretación general del R-squared
if r2 >= 0.8:
    print("El R-squared indica un rendimiento muy bueno: el modelo explica una gran parte de la variabilidad en los valores de vivienda.")
elif r2 >= 0.6:
    print("El R-squared sugiere un buen rendimiento: el modelo explica una parte sustancial de la variabilidad.")
elif r2 >= 0.4:
    print("El R-squared indica un rendimiento moderado: el modelo explica una parte limitada pero notable de la variabilidad.")
else:
    print("El R-squared sugiere un rendimiento bajo: el modelo explica una pequeña parte de la variabilidad en los valores de vivienda.")

print(f"\n--- Proceso completo para '{model_name}' finalizado ---")