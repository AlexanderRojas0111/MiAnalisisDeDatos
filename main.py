from data_loader import load_california_housing_data
from eda import perform_eda
from preprocessor import prepare_data
from models import get_model, train_model, make_predictions
from evaluator import evaluate_model
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

    modelos_disponibles = ["Linear Regression", "Decision Tree", "Random Forest"]
    resultados = []

    for model_name in modelos_disponibles:
        print(f"\n{'='*50}")
        print(f"Procesando el modelo: {model_name}")
        print(f"{'='*50}")

        # Obtener la instancia del modelo
        model = get_model(model_name)

        # Entrenar el modelo
        trained_model = train_model(model, model_name, X_train_scaled, y_train)

        # Realizar predicciones
        y_pred = make_predictions(trained_model, model_name, X_test_scaled)

        # Evaluar el modelo
        metrics = evaluate_model(y_test, y_pred, model_name)
        resultados.append({"Modelo": model_name, **metrics})

    # Mostrar comparativo final
    print("\n\n=== Comparativo de Modelos ===")
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    print("\n--- Proyecto de Análisis de Datos Completado ---")

if __name__ == "__main__":
    run_analysis()