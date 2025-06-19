from data_loader import load_california_housing_data
from eda import perform_eda
from preprocessor import prepare_data
from models import get_model, train_model, make_predictions
from evaluator import evaluate_model

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

    # 3. Preparar los datos (división y normalización)
    # Puedes ajustar 'use_specific_features=True' y 'feature_list'
    # si deseas usar solo 6 características específicas como hablamos antes.
    # Por defecto, usará todas las features de X.
    # Ejemplo para 6 features:
    # selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    # X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df, use_specific_features=True, feature_list=selected_features)
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)


    # 4. Elegir, Entrenar y Evaluar Modelos
    # Lista de modelos a probar. Puedes añadir o quitar de aquí.
    # Para probar un solo modelo, comenta los otros.
    models_to_test = ["Linear Regression", "Decision Tree", "Random Forest"]

    for model_name in models_to_test:
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
        evaluate_model(y_test, y_pred, model_name)

    print("\n--- Proyecto de Análisis de Datos Completado ---")

if __name__ == "__main__":
    run_analysis()