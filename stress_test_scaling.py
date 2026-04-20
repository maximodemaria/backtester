import time
import multiprocessing as mp
import numpy as np
from numba import njit

# --- MOTOR SIMULADO PARA TEST ---
@njit(fastmath=True, cache=True)
def heavy_motor(data, start, end, sig_buf):
    n_rows = data.shape[0]
    total = 0.0
    for i in range(start, end):
        # Simular carga de trabajo del backtester (2 pasadas por data)
        for j in range(n_rows):
            sig_buf[j] = 1 if data[j, 0] > data[j, 1] else 0
        
        acc = 0.0
        for j in range(n_rows):
            acc += data[j, 1] * sig_buf[j]
        
        if acc > 0: total += 1
    return total

def worker_task(args):
    data, start, end = args
    sig_buf = np.zeros(data.shape[0], dtype=np.int8)
    t0 = time.perf_counter()
    res = heavy_motor(data, start, end, sig_buf)
    t1 = time.perf_counter()
    return t1 - t0

def run_stress_test():
    # Configuración de la carga
    n_rows = 10000
    total_tasks = 100000
    data = np.random.rand(n_rows, 5).astype(np.float64)
    
    cores_to_test = [1, 4, 8, 16, 24, 32]
    
    print(f"ESTRÉS DE ESCALAMIENTO (Rows: {n_rows:,}, Total BTs: {total_tasks:,})")
    print(f"{'Cores':<10} | {'Total Time':<12} | {'BT/s Total':<15} | {'Eficiencia':<10}")
    print("-" * 60)
    
    results = {}
    base_speed = 0
    
    for n in cores_to_test:
        chunk_size = total_tasks // n
        tasks = []
        for i in range(n):
            s = i * chunk_size
            e = s + chunk_size if i < n-1 else total_tasks
            tasks.append((data, s, e))
            
        t_start = time.perf_counter()
        with mp.Pool(n) as pool:
            pool.map(worker_task, tasks)
        t_end = time.perf_counter()
        
        total_time = t_end - t_start
        speed = total_tasks / total_time
        
        if n == 1:
            base_speed = speed
            eficiencia = 1.0
        else:
            eficiencia = speed / (base_speed * n)
            
        print(f"{n:<10} | {total_time:<12.2f} | {speed:<15,.0f} | {eficiencia:<10.2%}")
        results[n] = speed

    print("-" * 60)
    if results[max(cores_to_test)] < results[min(cores_to_test)] * 2:
        print("DIAGNÓSTICO: Existe una saturación CRÍTICA. El sistema no escala con más núcleos.")
    else:
        print("DIAGNÓSTICO: El sistema escala, pero con rendimientos decrecientes.")

if __name__ == "__main__":
    run_stress_test()
