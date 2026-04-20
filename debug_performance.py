import numpy as np
import time
from numba import njit

@njit(fastmath=True, error_model='numpy')
def benchmark_engine(data, n_backtests):
    """
    Versión simplificada del motor para medir el impacto de la memoria.
    Calcula un Profit Factor ficticio recorriendo filas.
    """
    n_rows = data.shape[0]
    total_pf = 0.0
    
    # Simular n_backtests
    for i in range(n_backtests):
        gross_profits = 0.0
        gross_losses = 0.0
        
        # Simular indicadores (columnas fijas)
        col_a = 0
        col_b = 1
        
        for j in range(n_rows):
            # Acceso por FILA j (esto es lo que cambia entre C y F)
            val_a = data[j, col_a]
            val_b = data[j, col_b]
            
            # Lógica simple para simular carga
            if val_a > val_b:
                gross_profits += 0.0001
            else:
                gross_losses += 0.0001
        
        pf = gross_profits / gross_losses if gross_losses > 0 else 1.0
        total_pf += pf
        
    return total_pf

def run_test():
    n_rows = 50000
    n_backtests = 1000
    print(f"Iniciando Diagnóstico de Alineación (Filas: {n_rows:,}, Backtests: {n_backtests:,})")
    print("-" * 50)
    
    # 1. TEST CON C-ORDER (Row-Major)
    data_c = np.random.rand(n_rows, 10).astype(np.float64) # Contiguo en filas
    # Warmup
    benchmark_engine(data_c, 1)
    
    t0 = time.perf_counter()
    benchmark_engine(data_c, n_backtests)
    t1 = time.perf_counter()
    speed_c = n_backtests / (t1 - t0)
    print(f"RESULTADO C-ORDER (Row-major): {speed_c:,.2f} BT/s | Tiempo: {t1-t0:.4f}s")
    
    # 2. TEST CON F-ORDER (Column-Major)
    data_f = np.asfortranarray(data_c) # Contiguo en columnas
    # Warmup
    benchmark_engine(data_f, 1)
    
    t0 = time.perf_counter()
    benchmark_engine(data_f, n_backtests)
    t1 = time.perf_counter()
    speed_f = n_backtests / (t1 - t0)
    print(f"RESULTADO F-ORDER (Column-major): {speed_f:,.2f} BT/s | Tiempo: {t1-t0:.4f}s")
    
    # 3. IMPACTO
    impacto = (speed_c / speed_f) if speed_f > 0 else 0
    print("-" * 50)
    print(f"DIAGNÓSTICO: La alineación C-ORDER es {impacto:.2f}x MÁS RÁPIDA que F-ORDER para este motor.")
    
    if impacto > 1.2:
        print("\nCONCLUSIÓN: 'order=F' está causando cache misses críticos en el bucle de barras.")
    else:
        print("\nCONCLUSIÓN: La alineación no parece ser el cuello de botella principal.")

if __name__ == "__main__":
    run_test()
