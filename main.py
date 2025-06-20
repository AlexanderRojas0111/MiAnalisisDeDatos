from data_loader import load_california_housing_data
from eda import perform_eda
from preprocessor import prepare_data
from models import get_model, train_model, make_predictions
from evaluator import evaluate_model
from neural_network_model import get_neural_network_model, train_nn_model, predict_nn_model
import pandas as pd

def run_analysis():
    """
    Ejecuta el flujo completo de análisis de datos:
    Carga, EDA, preprocesamiento, entrenamiento y evaluación del modelo.
    """
    print("--- Iniciando el Proyecto de Análisis de Datos ---")

    # 1. Cargar los datos
    df = load_california_housing_data()

    # 2. Realizar Análisis Exploratorio de Datos (EDA)
    perform_eda(df)

    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)

    modelos_disponibles = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Neural Network"
    ]
    resultados = []

    for model_name in modelos_disponibles:
        print(f"\n{'='*50}")
        print(f"Procesando el modelo: {model_name}")
        print(f"{'='*50}")

        if model_name == "Neural Network":
            # Obtener y entrenar el modelo de red neuronal
            model = get_neural_network_model(X_train_scaled.shape[1])
            trained_model = train_nn_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
            y_pred = predict_nn_model(trained_model, X_test_scaled)
        else:
            # Modelos clásicos
            model = get_model(model_name)
            trained_model = train_model(model, model_name, X_train_scaled, y_train)
            y_pred = make_predictions(trained_model, model_name, X_test_scaled)

        metrics = evaluate_model(y_test, y_pred, model_name)
        resultados.append({"Modelo": model_name, **metrics})

    # Mostrar comparativo final
    print("\n\n=== Comparativo de Modelos ===")
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    print("\n--- Proyecto de Análisis de Datos Completado ---")

if __name__ == "__main__":
    run_analysis()