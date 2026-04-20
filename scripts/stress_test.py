import os
import time
import numpy as np
import pandas as pd
from src.core.orchestrator import ValidationOrchestrator
from src.core.config_loader import EnvironmentConfig
from src.strategies.moving_average import MovingAverageStrategy
from src.utils.data_gen import generate_sample_data

def run_stress_test(n_rows=1000000, n_permutations=1000):
    """
    Ejecuta una auditoría de estrés masiva sobre el framework.
    Mide latencia, throughput y estabilidad.
    """
    print(f"=== INICIANDO AUDITORÍA DE ESTRÉS (N={n_rows}, Perm={n_permutations}) ===")
    
    # 1. Preparación de Entorno
    data_path = "stress_data.csv"
    if not os.path.exists(data_path):
        print(f"Generando {n_rows} barras de datos sintéticos...")
        start_gen = time.perf_counter()
        generate_sample_data(data_path, n_rows=n_rows)
        print(f"Datos generados en {time.perf_counter() - start_gen:.2f}s")

    # 2. Configuración de Estrategia de Estrés
    # Creamos una clase hija que sobrescribe el grid para el test de estrés
    class StressStrategy(MovingAverageStrategy):
        @property
        def param_grid(self) -> list:
            # Generamos N permutaciones artificiales
            return [{'fast_period': i, 'slow_period': i*2} for i in range(5, 5 + n_permutations)]

    strategy = StressStrategy()
    
    # 3. Carga de Configuración (Template Default)
    # Forzamos los parámetros del entorno para el test
    config = EnvironmentConfig("default")
    config.config['environment']['dataset_path'] = data_path
    config.config['environment']['commission_bps'] = 1.0 # Stress con bajas comisiones

    # 4. Ejecución del Pipeline con Profiling
    orchestrator = ValidationOrchestrator(strategy, config)
    
    print("Ejecutando Pipeline de Punta a Punta...")
    t0 = time.perf_counter()
    
    results = orchestrator.run_pipeline()
    
    total_time = time.perf_counter() - t0
    
    # 5. Reporte de Rendimiento
    print("\n" + "="*40)
    print("REPORTE DE RENDIMIENTO (AUDITORÍA DE ESTRÉS)")
    print("="*40)
    print(f"Tiempo Total: {total_time:.2f} segundos")
    
    if total_time > 0:
        throughput = n_permutations / total_time
        print(f"Throughput: {throughput:.2f} backtests/segundo")
        print(f"Latencia Media: {total_time/n_permutations*1000:.4f} ms por configuración")
    
    print(f"Resultado OOS: {results if results else 'FAILED'}")
    print("="*40)

    # Limpieza (opcional)
    # os.remove(data_path)

if __name__ == "__main__":
    # Ajustamos N para que el test sea representativo pero no eterno en este turno
    run_stress_test(n_rows=500000, n_permutations=500)
