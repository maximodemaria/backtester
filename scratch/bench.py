import os
import time
import numpy as np
from src.core.orchestrator import ValidationOrchestrator
from src.strategies.quad_ma import QuadMAStrategy

class MockConfig:
    def __init__(self):
        self.dataset_path = "data/GGAL_1m.csv"
        self.commission_bps = 6.05

def run_bench():
    strategy = QuadMAStrategy()
    config = MockConfig()
    orchestrator = ValidationOrchestrator(strategy, config)
    
    # Reducimos total_p para el bench
    print("Iniciando Benchmark...")
    orchestrator.processor.load_data()
    
    ma_types = ['sma']
    periods = [5, 10, 15, 20]
    orchestrator.processor.precompute_indicators(ma_types, periods)
    is_data, _ = orchestrator.processor.split_is_oos(0.7)
    
    # 10,000 backtests
    n_backtests = 10000
    
    # Pre-alocacion de buffers
    n = len(is_data)
    sig_buf = np.zeros(n, dtype=np.int8)
    ret_buf = np.zeros(n, dtype=np.float64)
    
    # Mock indicator matrix
    map_matrix = np.zeros((14, 41), dtype=np.int32)
    # sma is 0. periods are 5 (idx 1), 10 (idx 2)...
    map_matrix[0, 1] = 3
    map_matrix[0, 2] = 4
    map_matrix[0, 3] = 5
    map_matrix[0, 4] = 6
    
    from src.core.orchestrator import _njit_massive_loop_master
    
    start = time.time()
    # Ejecutamos 10k BTs en un solo hilo
    best_pf, best_idx, survivors = _njit_massive_loop_master(
        0, n_backtests,
        is_data,
        strategy.legal_pairs,
        map_matrix,
        4, n // 4, 1.0,
        6.05/10000.0,
        sig_buf,
        ret_buf
    )
    end = time.time()
    
    elapsed = end - start
    speed = n_backtests / elapsed if elapsed > 0 else 0
    print(f"Benchmark finalizado: {speed:.0f} BT/s")
    print(f"Mejor PF: {best_pf}")

if __name__ == "__main__":
    run_bench()
