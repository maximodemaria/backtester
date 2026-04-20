import os
import numpy as np
from src.core.orchestrator import ValidationOrchestrator
from src.strategies.quad_ma import QuadMAStrategy

class MockConfig:
    def __init__(self):
        self.dataset_path = "data/GGAL_1m.csv"
        self.commission_bps = 6.05

def test_minimal():
    strategy = QuadMAStrategy()
    config = MockConfig()
    orchestrator = ValidationOrchestrator(strategy, config)
    
    # Reducimos total_p para el test
    print("Iniciando test minimal...")
    orchestrator.processor.load_data()
    orchestrator.processor.precompute_indicators(['sma'], [1, 5, 10])
    is_data, _ = orchestrator.processor.split_is_oos(0.7)
    
    shm_name, shm_shape = orchestrator.processor.create_shared_buffer()
    print(f"SHM creada: {shm_name} {shm_shape}")
    
    # Test manual de un worker
    from src.core.orchestrator import init_worker, _worker_bt_chunk
    indicator_matrix = np.zeros((14, 41), dtype=np.int32)
    # sma is ma_idx 0, period 5 is p_idx 1, period 10 is p_idx 2
    # indicator_map returned from precompute_indicators in processor
    
    # Simulamos el worker
    init_worker(shm_name, shm_shape, strategy.legal_pairs, indicator_matrix)
    print("Worker iniciado exitosamente.")
    
    res, count = _worker_bt_chunk(((0, 100), strategy.__class__, 6.05))
    print(f"Resultado worker: {res}, procesados: {count}")
    
    orchestrator.processor._shm.close()
    orchestrator.processor._shm.unlink()
    print("Test completado.")

if __name__ == "__main__":
    test_minimal()
