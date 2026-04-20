"""
Punto de entrada principal para la ejecución del framework de validación.
"""
import os
from src.strategies.moving_average import MovingAverageStrategy
from src.core.orchestrator import ValidationOrchestrator
from src.utils.data_gen import generate_sample_data

def main():
    """
    Función principal que orquesta la ejecución demo del framework.
    """
    # 1. Preparar datos si no existen
    data_path = "sample_data.csv"
    if not os.path.exists(data_path):
        generate_sample_data(data_path, n_rows=10000)

    # 2. Instanciar Estrategia (siguiendo el template del contrato)
    strategy = MovingAverageStrategy()

    # 3. Inicializar Orquestador del Framework
    orchestrator = ValidationOrchestrator(strategy, data_path)

    # 4. Ejecutar Pipeline Completo
    results = orchestrator.run_pipeline()

    if results:
        print("\n" + "="*30)
        print("RESULTADOS FINALES OOS")
        print("="*30)
        for k, v in results.items():
            print(f"{k.upper()}: {v:.4f}")
        print("="*30)

if __name__ == "__main__":
    main()
