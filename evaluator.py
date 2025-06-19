from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred, model_name):
    """
    Evalúa el rendimiento del modelo de regresión e imprime las métricas.

    Args:
        y_true (pandas.Series): Los valores reales de la variable objetivo del conjunto de prueba.
        y_pred (np.ndarray): Las predicciones del modelo.
        model_name (str): El nombre del modelo que se está evaluando.
    """
    print(f"\n--- 5. Evaluación del Modelo '{model_name}' ---")

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Error Absoluto Medio (MAE): {mae:.3f} (Promedio de error absoluto en $x100,000)")
    print(f"Error Cuadrático Medio (MSE): {mse:.3f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.3f} (Error promedio en la misma unidad que el target)")
    print(f"Coeficiente de Determinación (R-squared): {r2:.3f} (Proporción de varianza explicada, 0-1)")

    # Comparación de algunas predicciones vs valores reales
    print("\nComparación de las primeras 10 Predicciones vs. Valores Reales:")
    for pred, real in zip(y_pred[:10], y_true[:10]):
        print(f"Predicción: {pred:.2f}, Real: {real:.2f}")

    # Retornar las métricas para el comparativo
    return {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    }


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