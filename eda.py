import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Importar pandas si no viene de df

def perform_eda(df):
    """
    Realiza un Análisis Exploratorio de Datos (EDA) en el DataFrame proporcionado.

    Args:
        df (pandas.DataFrame): El DataFrame sobre el que se realizará el EDA.
    """
    print("\n--- 2. Análisis Exploratorio de Datos (EDA) ---")

    # 2.1 Verificar valores nulos
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("Este dataset no contiene valores nulos, lo que facilita el preprocesamiento.")

    # 2.2 Visualizar la distribución de variables (Histogramas)
    print("\nGenerando histogramas de distribución para todas las variables...")
    df.hist(figsize=(12, 8))
    plt.suptitle("Histogramas de variables")
    plt.tight_layout()
    plt.savefig("eda_histogramas.png")
    plt.show()

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

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Mapa de calor de correlación")
    plt.tight_layout()
    plt.savefig("eda_correlacion.png")
    plt.show()

    print("\nCorrelación de cada variable con MedHouseVal (variable objetivo):")
    print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
    print("El Ingreso Medio (MedInc) es la variable más fuertemente correlacionada positivamente.")

    # 2.4 Gráficos de dispersión para las relaciones clave
    print("\nGenerando gráficos de dispersión para visualizar relaciones clave...")
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=df, alpha=0.1)
    plt.title('Ingreso Medio (MedInc) vs. Valor Medio de Vivienda')
    plt.xlabel('Ingreso Medio (x10,000 $)')
    plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

    plt.subplot(1, 3, 2)
    sns.scatterplot(x='AveRooms', y='MedHouseVal', data=df, alpha=0.1)
    plt.title('Número Promedio de Habitaciones vs. Valor Medio de Vivienda')
    plt.xlabel('Número Promedio de Habitaciones')
    plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

    plt.subplot(1, 3, 3)
    sns.scatterplot(x='HouseAge', y='MedHouseVal', data=df, alpha=0.1)
    plt.title('Edad Media de la Vivienda vs. Valor Medio de Vivienda')
    plt.xlabel('Edad Media de la Vivienda')
    plt.ylabel('Valor Medio de Vivienda (x100,000 $)')

    plt.tight_layout()
    plt.show()

    # 2.5 Box Plots para detección de Outliers
    print("\nGenerando Box Plots para detección de Outliers en cada variable...")
    plt.figure(figsize=(15, 12))
    plt.suptitle('Box Plots para Detección de Outliers en Variables', y=1.02)

    for i, column in enumerate(df.columns):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(y=df[column])
        plt.title(column)
        plt.ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()
    print("Variables como 'Population' y 'AveOccup' muestran una considerable cantidad de outliers.")

if __name__ == "__main__":
    # Ejemplo de uso si ejecutas eda.py directamente
    from data_loader import load_california_housing_data
    data = load_california_housing_data()
    perform_eda(data)