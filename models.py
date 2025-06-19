from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_model(model_name, random_state=42):
    """
    Retorna una instancia del modelo de regresión especificado.

    Args:
        model_name (str): El nombre del modelo a obtener ('Linear Regression',
                          'Decision Tree', 'Random Forest').
        random_state (int): Semilla para la reproducibilidad de modelos que la usan.

    Returns:
        object: Una instancia del modelo de scikit-learn.
    """
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=random_state)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}. Elige entre 'Linear Regression', 'Decision Tree', 'Random Forest'.")
    return model

def train_model(model, model_name, X_train_scaled, y_train):
    """
    Entrena un modelo de regresión.

    Args:
        model (object): La instancia del modelo de scikit-learn.
        model_name (str): El nombre del modelo para mensajes de impresión.
        X_train_scaled (np.ndarray): Conjunto de entrenamiento de características escaladas.
        y_train (pandas.Series): Conjunto de entrenamiento de la variable objetivo.

    Returns:
        object: El modelo entrenado.
    """
    print(f"\nEntrenando el modelo '{model_name}' con los datos de entrenamiento escalados...")
    model.fit(X_train_scaled, y_train)
    print(f"Modelo '{model_name}' entrenado exitosamente.")
    return model

def make_predictions(model, model_name, X_test_scaled):
    """
    Realiza predicciones usando un modelo entrenado.

    Args:
        model (object): El modelo de scikit-learn entrenado.
        model_name (str): El nombre del modelo para mensajes de impresión.
        X_test_scaled (np.ndarray): Conjunto de prueba de características escaladas.

    Returns:
        np.ndarray: Las predicciones del modelo.
    """
    print(f"\nRealizando predicciones en el conjunto de prueba con '{model_name}'...")
    y_pred = model.predict(X_test_scaled)
    print("Predicciones generadas.")

    # Mostrar algunas de las primeras predicciones junto a los valores reales (solo las predicciones aquí)
    # Los valores reales se mostrarán en el evaluador.
    print(f"\nPrimeras 5 predicciones del modelo '{model_name}':")
    print(y_pred[:5])
    return y_pred

if __name__ == "__main__":
    # Ejemplo de uso si ejecutas models.py directamente
    # Requiere preprocessor.py y data_loader.py para funcionar
    from data_loader import load_california_housing_data
    from preprocessor import prepare_data
    data = load_california_housing_data()
    X_train, X_test, y_train, y_test = prepare_data(data)

    model_name_to_test = "Random Forest"
    selected_model = get_model(model_name_to_test)
    trained_model = train_model(selected_model, model_name_to_test, X_train, y_train)
    predictions = make_predictions(trained_model, model_name_to_test, X_test)
    print(f"\nPredicciones generadas para {model_name_to_test}.")