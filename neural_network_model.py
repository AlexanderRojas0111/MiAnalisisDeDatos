# neural_network_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np # Para np.random.seed si es necesario

def get_neural_network_model(input_dim):
    """
    Construye y compila una red neuronal feedforward (MLP) para regresión.

    Args:
        input_dim (int): El número de características de entrada de los datos.

    Returns:
        tf.keras.Model: El modelo de red neuronal compilado.
    """
    print("\n--- Construyendo la Arquitectura de la Red Neuronal ---")

    # Inicializa el modelo secuencial (capas apiladas una tras otra)
    model = Sequential([
        # Capa de entrada y primera capa oculta
        # La primera capa 'Dense' requiere 'input_dim' para saber cuántas entradas esperar.
        # Usamos 64 neuronas y la función de activación ReLU.
        Dense(64, activation='relu', input_shape=(input_dim,)), # input_shape espera una tupla (num_features,)
        
        # Segunda capa oculta (opcional, pero ayuda a aprender patrones más complejos)
        Dense(32, activation='relu'),
        
        # Capa de salida
        # Una sola neurona para la tarea de regresión (predecir un único valor continuo).
        # No se usa función de activación (o se usa 'linear'), lo que permite cualquier valor real como salida.
        Dense(1)
    ])

    # Compilar el modelo
    # 'optimizer': Adam es un optimizador muy eficiente y robusto para la mayoría de los casos.
    # 'loss': 'mse' (Mean Squared Error) es la función de pérdida estándar para problemas de regresión.
    # 'metrics': 'mae' (Mean Absolute Error) es una métrica fácil de interpretar durante el entrenamiento.
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    print("Arquitectura de la Red Neuronal construida y compilada.")
    model.summary() # Muestra un resumen de las capas del modelo
    return model

def train_nn_model(model, X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """
    Entrena el modelo de red neuronal.

    Args:
        model (tf.keras.Model): El modelo de red neuronal compilado.
        X_train_scaled (np.ndarray): Datos de entrenamiento escalados.
        y_train (pandas.Series): Etiquetas de entrenamiento.
        epochs (int): Número de épocas para el entrenamiento.
        batch_size (int): Tamaño del lote para el entrenamiento.
        validation_split (float): Proporción de los datos de entrenamiento a usar como validación.

    Returns:
        tf.keras.Model: El modelo de red neuronal entrenado.
    """
    print(f"\n--- Entrenando la Red Neuronal por {epochs} épocas ---")
    # Asegura que y_train sea un array de NumPy
    if hasattr(y_train, "values"):
        y_train = y_train.values
    model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split, # Usa una parte del entrenamiento para validación interna
        verbose=1 # Muestra el progreso del entrenamiento
    )
    print("Entrenamiento de la Red Neuronal completado.")
    return model

def evaluate_nn_model(model, X_test_scaled, y_test):
    """
    Evalúa el modelo de red neuronal en el conjunto de prueba.

    Args:
        model (tf.keras.Model): El modelo de red neuronal entrenado.
        X_test_scaled (np.ndarray): Datos de prueba escalados.
        y_test (pandas.Series): Etiquetas de prueba.

    Returns:
        list: Lista de valores de métricas (loss, mae) en el conjunto de prueba.
    """
    print("\n--- Evaluando la Red Neuronal en el conjunto de prueba ---")
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Pérdida (MSE) en el conjunto de prueba: {loss:.3f}")
    print(f"Error Absoluto Medio (MAE) en el conjunto de prueba: {mae:.3f}")
    return loss, mae

def predict_nn_model(model, X_test_scaled):
    """
    Realiza predicciones con el modelo de red neuronal.

    Args:
        model (tf.keras.Model): El modelo de red neuronal entrenado.
        X_test_scaled (np.ndarray): Datos para realizar predicciones.

    Returns:
        np.ndarray: Predicciones del modelo.
    """
    print("\n--- Realizando predicciones con la Red Neuronal ---")
    predictions = model.predict(X_test_scaled)
    # Keras devuelve las predicciones como un array 2D [[val1], [val2], ...],
    # lo convertimos a 1D para que sea consistente con y_pred de otros modelos.
    return predictions.flatten()

if __name__ == "__main__":
   

    # Este bloque simula los pasos de carga y preprocesamiento de main.py
    print("--- Ejemplo de Uso del Módulo 'neural_network_model.py' ---")
    from data_loader import load_california_housing_data
    from preprocessor import prepare_data
    
    # 1. Cargar datos
    df = load_california_housing_data()

    # 2. Preparar datos (dividir y escalar)
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_data(df)

    # 3. Construir, entrenar y evaluar la Red Neuronal
    # Obtenemos la dimensión de entrada (número de características) de X_train_scaled
    input_dim = X_train_scaled.shape[1] 
    
    nn_model = get_neural_network_model(input_dim)
    
    # Entrenar el modelo (usando un número pequeño de épocas para la demostración)
    history = train_nn_model(nn_model, X_train_scaled, y_train, epochs=10, batch_size=64)
    
    # Evaluar el modelo
    loss, mae = evaluate_nn_model(nn_model, X_test_scaled, y_test)
    
    # Realizar predicciones
    predictions = predict_nn_model(nn_model, X_test_scaled)
    print("\nPrimeras 5 predicciones del modelo de red neuronal:")
    print(predictions[:5])
    print("--- Ejemplo de Uso Completado ---")