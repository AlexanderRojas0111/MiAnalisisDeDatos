import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_california_housing_data():
    """
    Carga el dataset de California Housing.

    Returns:
        pandas.DataFrame: El DataFrame completo con todas las características
                          y la variable objetivo.
    """
    print("\n--- 1. Carga del Dataset California Housing ---")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print("Dataset de California Housing cargado exitosamente.")
    return df

if __name__ == "__main__":
    # Si ejecutas este archivo directamente, cargará y mostrará info.
    data = load_california_housing_data()
    print("\nPrimeras 5 filas del DataFrame cargado:")
    print(data.head())
    print("\nInformación general del DataFrame cargado:")
    data.info()
    print("\nEstadísticas descriptivas básicas del DataFrame cargado:")
    print(data.describe())