from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_model(model_name, random_state=42):
    """
    Retorna una instancia del modelo de regresión especificado.
    """
    if model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Decision Tree":
        return DecisionTreeRegressor(random_state=random_state)
    elif model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}. Elige entre 'Linear Regression', 'Decision Tree', 'Random Forest'.")

def train_model(model, model_name, X_train_scaled, y_train):
    """
    Entrena un modelo de regresión.
    """
    print(f"\nEntrenando el modelo '{model_name}' con los datos de entrenamiento escalados...")
    model.fit(X_train_scaled, y_train)
    print(f"Modelo '{model_name}' entrenado exitosamente.")
    return model

def make_predictions(model, model_name, X_test_scaled):
    """
    Realiza predicciones usando un modelo entrenado.
    """
    print(f"\nRealizando predicciones en el conjunto de prueba con '{model_name}'...")
    y_pred = model.predict(X_test_scaled)
    print("Predicciones generadas.")

    print(f"\nPrimeras 5 predicciones del modelo '{model_name}':")
    print(y_pred[:5])
    return y_pred

if __name__ == "__main__":
    # Ejemplo de uso si ejecutas models.py directamente
    # Requiere preprocessor.py y data_loader.py para funcionar
    from data_loader import load_california_housing_data
    from preprocessor import prepare_data

    modelos_disponibles = ["Linear Regression", "Decision Tree", "Random Forest"]
    print("\nModelos disponibles para evaluar:")
    for idx, modelo in enumerate(modelos_disponibles, 1):
        print(f"{idx}. {modelo}")
    seleccion = input("Ingrese el número del modelo a probar: ")
    try:
        idx = int(seleccion.strip()) - 1
        if 0 <= idx < len(modelos_disponibles):
            model_name_to_test = modelos_disponibles[idx]
        else:
            print("Selección inválida. Usando 'Linear Regression' por defecto.")
            model_name_to_test = "Linear Regression"
    except Exception:
        print("Entrada inválida. Usando 'Linear Regression' por defecto.")
        model_name_to_test = "Linear Regression"

    data = load_california_housing_data()
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(data)

    selected_model = get_model(model_name_to_test)
    trained_model = train_model(selected_model, model_name_to_test, X_train_scaled, y_train)
    predictions = make_predictions(trained_model, model_name_to_test, X_test_scaled)
    print(f"\nPredicciones generadas para {model_name_to_test}.")