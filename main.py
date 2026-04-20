import os
import argparse
from src.strategies.moving_average import MovingAverageStrategy
from src.core.orchestrator import ValidationOrchestrator
from src.core.config_loader import EnvironmentConfig
from src.utils.data_gen import generate_sample_data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """
    Punto de entrada CLI para el Framework de Backtesting.
    """
    parser = argparse.ArgumentParser(description="Framework de Backtesting Cuantitativo HFT")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="MovingAverage",
        help="Nombre de la clase de estrategia a ejecutar (ej. MovingAverage)"
    )
    
    parser.add_argument(
        "--template", 
        type=str, 
        default="default",
        help="Nombre del archivo YAML en /templates/ (sin extensión)"
    )

    args = parser.parse_args()

    # 1. Cargar Configuración desde Template
    try:
        config = EnvironmentConfig(args.template)
    except Exception as e:
        print(f"ERROR DE CONFIGURACIÓN: {str(e)}")
        return

    # 2. Preparar datos si no existen (Basado en el path del template)
    if not os.path.exists(config.dataset_path):
        print(f"Dataset no encontrado en {config.dataset_path}. Generando datos de muestra...")
        generate_sample_data(config.dataset_path, n_rows=10000)

    # 3. Instanciar Estrategia
    # Por ahora soportamos MovingAverage dinámicamente
    if args.strategy == "MovingAverage":
        strategy = MovingAverageStrategy()
    else:
        print(f"Error: Estrategia '{args.strategy}' no reconocida.")
        return

    print(f"--- Iniciando Backtester Framework ---")
    print(f"Estrategia: {strategy.name}")
    print(f"Template: {args.template}")
    print(f"Comisiones: {config.commission_bps} bps")
    print("-" * 40)

    # 4. Inicializar Orquestador del Framework
    orchestrator = ValidationOrchestrator(strategy, config)

    # 5. Ejecutar Pipeline Completo
    results = orchestrator.run_pipeline()

    if results:
        print("\n" + "="*30)
        print("RESULTADOS FINALES OOS (Neto de Comisiones)")
        print("="*30)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k.upper()}: {v:.4f}")
            else:
                print(f"{k.upper()}: {v}")
        print("="*30)

if __name__ == "__main__":
    main()
