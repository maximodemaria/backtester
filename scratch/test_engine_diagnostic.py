import numpy as np
import time
from numba import njit
from src.core.data_processor import DataProcessor
from src.core.jit_ops import _njit_massive_loop_master, _get_params_jit, _get_signals_by_indices_inplace_jit, _check_single_survival_jit, _compute_metrics_inplace_jit

def test():
    print("Iniciando Test de Warmup...")
    shm_shape = (140000, 500)
    map_matrix = np.zeros((14, 41), dtype=np.int32)
    legal_pairs = np.zeros((820, 2), dtype=np.int32)
    
    # Mock data
    data = np.random.rand(140000, 500).astype(np.float64)
    # We can't connect to SHM if it doesn't exist, so we skip SHM for this test
    # and pass the raw data to see if @njit works.
    
    sig_buf = np.zeros(140000, dtype=np.int8)
    ret_buf = np.zeros(140000, dtype=np.float64)
    
    print("Llamando a _njit_massive_loop_master...")
    start = time.time()
    res = _njit_massive_loop_master(
        0, 10, 
        data, 
        legal_pairs, 
        map_matrix, 
        4, 140000 // 4, 1.1, 0.0006, 
        sig_buf, ret_buf
    )
    end = time.time()
    print(f"Resultado: {res}")
    print(f"Tiempo: {end - start:.4f}s")

if __name__ == "__main__":
    test()
