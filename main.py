from data_loader import load_california_housing_data
from eda import perform_eda
from preprocessor import prepare_data
from models import get_model, train_model, make_predictions
from evaluator import evaluate_model
from neural_network_model import get_neural_network_model, train_nn_model, predict_nn_model
from report import export_results  # <--- Importa la función de exportación
import pandas as pd
import matplotlib.pyplot as plt
import os

CHARTS_DIR = "charts_statistical_data"

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

    # Crear la carpeta si no existe
    os.makedirs(CHARTS_DIR, exist_ok=True)

    for model_name in modelos_disponibles:
        print(f"\n{'='*50}")
        print(f"Procesando el modelo: {model_name}")
        print(f"{'='*50}")

        if model_name == "Neural Network":
            # Obtener y entrenar el modelo de red neuronal
            model = get_neural_network_model(X_train_scaled.shape[1])
            trained_model = train_nn_model(model, X_train_scaled, y_train)
            y_pred = predict_nn_model(trained_model, X_test_scaled)
        else:
            # Modelos clásicos
            model = get_model(model_name)
            trained_model = train_model(model, model_name, X_train_scaled, y_train)
            y_pred = make_predictions(trained_model, model_name, X_test_scaled)

        metrics = evaluate_model(y_test, y_pred, model_name)
        resultados.append({"Modelo": model_name, **metrics})

        # Gráfica de Real vs Predicción
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Valor Real")
        plt.ylabel("Predicción")
        plt.title(f"Real vs Predicción - {model_name}")
        plt.savefig(os.path.join(CHARTS_DIR, f"real_vs_prediccion_{model_name.replace(' ', '_').lower()}.png"))
        plt.close()

    # Mostrar comparativo final
    print("\n\n=== Comparativo de Modelos ===")
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    # Exportar resultados a archivo en la carpeta deseada
    export_results(
        df_resultados,
        csv_path=os.path.join(CHARTS_DIR, "resultados_modelos.csv"),
        png_path=os.path.join(CHARTS_DIR, "comparacion_modelos.png")
    )

    print("\n--- Proyecto de Análisis de Datos Completado ---")

def plot_model_comparison(df_resultados):
    """
    Grafica la comparación de métricas entre modelos.
    """
    metricas = ['MAE', 'MSE', 'RMSE', 'R2']
    modelos = df_resultados['Modelo']

    fig, axs = plt.subplots(1, len(metricas), figsize=(18, 5))
    for i, metrica in enumerate(metricas):
        axs[i].bar(modelos, df_resultados[metrica], color='skyblue')
        axs[i].set_title(metrica)
        axs[i].set_ylabel(metrica)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='right')  # <-- Esta línea reemplaza set_xticklabels
    plt.suptitle('Comparación de Modelos de Machine Learning')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_analysis()