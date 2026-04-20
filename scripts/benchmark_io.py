"""
Módulo de benchmark para comparar el rendimiento de carga entre archivos CSV y Parquet.
"""
import time
import os
import sys
from src.core.data_processor import DataProcessor
from src.utils.data_gen import generate_sample_data

def run_benchmark(n_rows=1000000):
    """
    Ejecuta una comparación de tiempos de carga entre formato CSV y Apache Parquet.
    """
    print(f"=== BENCHMARK I/O: CSV vs PARQUET (N={n_rows}) ===")

    csv_path = "bench.csv"
    parquet_path = "bench.parquet"

    # 1. Generación de Datos
    print("Generando archivos de prueba...")
    generate_sample_data(csv_path, n_rows=n_rows)
    generate_sample_data(parquet_path, n_rows=n_rows)

    # 2. Benchmark CSV
    print("\nMidiendo carga CSV...")
    dp_csv = DataProcessor(csv_path)
    t0 = time.perf_counter()
    dp_csv.load_data()
    t_csv = time.perf_counter() - t0
    print(f"Resultado CSV: {t_csv:.4f} segundos")

    # 3. Benchmark Parquet
    print("\nMidiendo carga PARQUET...")
    dp_pq = DataProcessor(parquet_path)
    t0 = time.perf_counter()
    dp_pq.load_data()
    t_pq = time.perf_counter() - t0
    print(f"Resultado PARQUET: {t_pq:.4f} segundos")

    # 4. Resultados
    improvement = (t_csv - t_pq) / t_csv * 100
    print("\n" + "="*30)
    print("RESUMEN DE OPTIMIZACIÓN")
    print("="*30)
    print(f"Reducción de Tiempo: {improvement:.2f}%")
    print(f"Factor de Velocidad: {t_csv / t_pq:.2f}x")
    print("="*30)

    # Limpieza
    os.remove(csv_path)
    os.remove(parquet_path)

if __name__ == "__main__":
    # Aseguramos el PYTHONPATH
    sys.path.append(os.getcwd())
    run_benchmark(n_rows=1000000)
