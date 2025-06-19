from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd # Necesario para df.drop y df[] si X e y no son de Pandas directamente

def prepare_data(df, target_column='MedHouseVal', test_size=0.20, random_state=42, use_specific_features=False, feature_list=None):
    """
    Prepara los datos: separa X e y, divide en conjuntos de entrenamiento/prueba y normaliza.

    Args:
        df (pandas.DataFrame): El DataFrame completo.
        target_column (str): El nombre de la columna objetivo.
        test_size (float): La proporción del dataset a usar para la prueba.
        random_state (int): Semilla para la reproducibilidad de la división.
        use_specific_features (bool): Si es True, usa solo las características en feature_list.
        feature_list (list): Lista de nombres de características a usar si use_specific_features es True.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
    """
    print("\n--- 3. Preparación de Datos para el Modelado ---")

    # Separar Variables de Entrada (X) y Salida (y)
    if use_specific_features and feature_list:
        X = df[feature_list]
        print(f"\nVariables de entrada seleccionadas: {feature_list}")
    else:
        X = df.drop(columns=[target_column])
        print("\nUsando todas las variables de entrada.")

    y = df[target_column]

    # Dividir el Dataset en conjuntos de entrenamiento y prueba
    print(f"\nDividiendo el dataset en conjuntos de entrenamiento ({1-test_size:.0%}) y prueba ({test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Tamaño del conjunto de entrenamiento (X_train): {X_train.shape}")
    print(f"Tamaño del conjunto de prueba (X_test): {X_test.shape}")

    # Normalizar las variables de entrada (X)
    print("\nNormalizando las variables de entrada (X) usando StandardScaler...")
    scaler = StandardScaler()

    # Ajustar el escalador SÓLO con los datos de entrenamiento y luego transformar X_train
    X_train_scaled = scaler.fit_transform(X_train)

    # Usar el escalador ajustado (con los parámetros de X_train) para transformar X_test
    X_test_scaled = scaler.transform(X_test)
    print("Variables de entrada normalizadas.")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Ejemplo de uso si ejecutas preprocessor.py directamente
    from data_loader import load_california_housing_data
    data = load_california_housing_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    print("\nDatos preprocesados y listos.")
    print(f"Forma de X_train_scaled: {X_train.shape}")
    print(f"Forma de X_test_scaled: {X_test.shape}")