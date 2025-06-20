import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_model_comparison(df_resultados, save_path=None):
    """
    Grafica la comparación de métricas entre modelos y permite guardar la figura.
    """
    metricas = ['MAE', 'MSE', 'RMSE', 'R2']
    modelos = df_resultados['Modelo']

    fig, axs = plt.subplots(1, len(metricas), figsize=(18, 5))
    for i, metrica in enumerate(metricas):
        axs[i].bar(modelos, df_resultados[metrica], color='skyblue')
        axs[i].set_title(metrica)
        axs[i].set_ylabel(metrica)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(axs[i].get_xticklabels(), rotation=45, ha='right')
    plt.suptitle('Comparación de Modelos de Machine Learning')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def export_results(df_resultados, csv_path="resultados_modelos.csv", png_path="comparacion_modelos.png"):
    """
    Exporta la tabla de resultados a CSV y la gráfica a PNG.
    """
    df_resultados.to_csv(csv_path, index=False)
    print(f"Tabla de resultados exportada a {csv_path}")
    plot_model_comparison(df_resultados, save_path=png_path)
    print(f"Gráfica comparativa exportada a {png_path}")

CHARTS_DIR = "charts_statistical_data"

# Ejemplo de uso independiente:
if __name__ == "__main__":
    csv_path = os.path.join(CHARTS_DIR, "resultados_modelos.csv")
    if os.path.exists(csv_path):
        df_resultados = pd.read_csv(csv_path)
        export_results(
            df_resultados,
            csv_path=csv_path,
            png_path=os.path.join(CHARTS_DIR, "comparacion_modelos.png")
        )
    else:
        print(f"El archivo '{csv_path}' no existe. Por favor, genera primero los resultados ejecutando tu análisis principal.")