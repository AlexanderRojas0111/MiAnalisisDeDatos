from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np # Para np.sqrt

def evaluate_model(y_test, y_pred, model_name):
    """
    Evalúa el rendimiento del modelo de regresión e imprime las métricas.

    Args:
        y_test (pandas.Series): Los valores reales de la variable objetivo del conjunto de prueba.
        y_pred (np.ndarray): Las predicciones del modelo.
        model_name (str): El nombre del modelo que se está evaluando.
    """
    print(f"\n--- 5. Evaluación del Modelo '{model_name}' ---")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Error Absoluto Medio (MAE): {mae:.3f} (Promedio de error absoluto en $x100,000)")
    print(f"Error Cuadrático Medio (MSE): {mse:.3f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.3f} (Error promedio en la misma unidad que el target)")
    print(f"Coeficiente de Determinación (R-squared): {r2:.3f} (Proporción de varianza explicada, 0-1)")

    # Interpretación general del R-squared
    if r2 >= 0.8:
        print("El R-squared sugiere un rendimiento muy bueno: el modelo explica una gran parte de la variabilidad en los valores de vivienda.")
    elif r2 >= 0.6:
        print("El R-squared sugiere un buen rendimiento: el modelo explica una parte sustancial de la variabilidad.")
    elif r2 >= 0.4:
        print("El R-squared indica un rendimiento moderado: el modelo explica una parte limitada pero notable de la variabilidad.")
    else:
        print("El R-squared sugiere un rendimiento bajo: el modelo explica una pequeña parte de la variabilidad en los valores de vivienda.")

    # Mostrar algunas de las primeras predicciones vs. valores reales
    print("\nComparación de las primeras 10 Predicciones vs. Valores Reales:")
    for i in range(10):
        print(f"Predicción: {y_pred[i]:.2f}, Real: {y_test.iloc[i]:.2f}")


if __name__ == "__main__":
    # Ejemplo de uso si ejecutas evaluator.py directamente
    # Requiere models.py, preprocessor.py y data_loader.py
    from data_loader import load_california_housing_data
    from preprocessor import prepare_data
    from models import get_model, train_model, make_predictions

    data = load_california_housing_data()
    X_train, X_test, y_train, y_test = prepare_data(data)

    model_name_to_test = "Linear Regression" # Puedes cambiarlo
    selected_model = get_model(model_name_to_test)
    trained_model = train_model(selected_model, model_name_to_test, X_train, y_train)
    predictions = make_predictions(trained_model, model_name_to_test, X_test)

    evaluate_model(y_test, predictions, model_name_to_test)
    print("\nEvaluación completada.")