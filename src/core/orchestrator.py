import time
import os
import multiprocessing
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.core.data_processor import DataProcessor
from src.core.backtester import BacktesterEngine
from src.core.jit_ops import (
    _get_params_jit,
    _get_signals_by_indices_inplace_jit,
    _check_single_survival_jit,
    _compute_metrics_inplace_jit
)
from src.core.validation.survival_tester import SurvivalTester
from src.core.validation.permutation_test import PermutationTest
from src.core.validation.validator_oos import ValidatorOOS
from src.utils.logger import AsyncLogger
from src.utils.process_guard import ProcessGuard

# --- INFRAESTRUCTURA DE NÚCLEOS AUTÓNOMOS ---
_worker_indicator_matrix = None
_worker_data = None
_worker_legal_pairs = None

def init_worker(shm_name, shm_shape, legal_pairs, indicator_matrix):
    """Inicializa el núcleo conectándose a la memoria compartida (Zero-Copy)."""
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs, _worker_shm
    from src.core.data_processor import DataProcessor
    from multiprocessing.shared_memory import SharedMemory
    
    _worker_legal_pairs = legal_pairs
    _worker_indicator_matrix = indicator_matrix
    
    # IMPORTANTE: Mantenemos la referencia de _worker_shm para que el buffer no se cierre en Windows
    _worker_shm = SharedMemory(name=shm_name)
    _worker_data = np.ndarray(shm_shape, dtype=np.float64, buffer=_worker_shm.buf)

def _worker_bt_chunk(args):
    """
    Función worker que utiliza el bucle maestro NJIT sobre datos compartidos.
    """
    range_tuple, strategy_class, commission_bps = args
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs
    start_idx, end_idx = range_tuple
    
    try:
        if _worker_data is None or _worker_indicator_matrix is None:
            return "ERROR: Worker no inicializado", 0
            
        # Parámetros de supervivencia
        n_windows = 4
        threshold = 1.0
        n_samples = len(_worker_data)
        window_size = n_samples // n_windows
        comm_factor = commission_bps / 10000.0
        
        # --- OPTIMIZACIÓN: REUSO DE BUFFERS POR WORKER ---
        # Alocamos una vez por chunk lo que necesite el NJIT interno
        # (Aunque lo ideal sería alocarlo en init_worker, para no complicar 
        # la firma de mp.Pool, lo hacemos aquí una sola vez por cada bloque de 100k)
        sig_buffer = np.zeros(n_samples, dtype=np.int8)
        ret_buffer = np.zeros(n_samples, dtype=np.float64)
        
        # --- MOTOR BOLA DE FUEGO ---
        best_pf, best_idx, survivors = _njit_massive_loop_master(
            start_idx, end_idx,
            _worker_data,
            _worker_legal_pairs,
            _worker_indicator_matrix,
            n_windows, window_size, threshold,
            comm_factor,
            sig_buffer,
            ret_buffer
        )
        
        winner_res = []
        if best_idx != -1:
            strategy = strategy_class()
            winner_params = strategy.get_params_by_index(best_idx)
            winner_res.append((winner_params, {"profit_factor": best_pf}))
            
        return winner_res, (end_idx - start_idx)

    except Exception as e:
        return f"ERROR: {str(e)}", 0

@njit(fastmath=True, error_model='numpy')
def _njit_massive_loop_master(start_idx, end_idx, data, legal_pairs, map_matrix, n_w, w_size, thresh, comm, sig_buf, ret_buf):
    """
    Motor Monolítico v2: Todo el pipeline de señales, supervivencia y métricas
    en un solo bloque de código máquina. Cero overhead de llamadas.
    """
    n_bars = data.shape[0]
    n_pairs = legal_pairs.shape[0]
    best_pf = -1.0
    best_idx = -1
    survivors = 0
    
    for i in range(start_idx, end_idx):
        # --- 1. DECODER DE PARÁMETROS (Inline) ---
        ma_type_idx = i // (n_pairs * n_pairs)
        rem = i % (n_pairs * n_pairs)
        entry_idx = rem // n_pairs
        exit_idx = rem % n_pairs
        
        fe_se = legal_pairs[entry_idx]
        fx_sx = legal_pairs[exit_idx]
        
        fe, se = fe_se[0], fe_se[1]
        fx, sx = fx_sx[0], fx_sx[1]

        # --- 2. RESOLVER COLUMNAS (Inline) ---
        fe_p_idx = 0 if fe == 1 else fe // 5
        se_p_idx = 0 if se == 1 else se // 5
        fx_p_idx = 0 if fx == 1 else fx // 5
        sx_p_idx = 0 if sx == 1 else sx // 5
        
        fe_col = map_matrix[ma_type_idx, fe_p_idx]
        se_col = map_matrix[ma_type_idx, se_p_idx]
        fx_col = map_matrix[ma_type_idx, fx_p_idx]
        sx_col = map_matrix[ma_type_idx, sx_p_idx]

        # --- 3. GENERACIÓN DE SEÑALES (Inline + Direct Access) ---
        current_pos = 0
        for j in range(n_bars):
            # Lógica de Salida
            if current_pos == 1:
                if data[j, fx_col] < data[j, sx_col]: current_pos = 0
            elif current_pos == -1:
                if data[j, fx_col] > data[j, sx_col]: current_pos = 0
            
            # Lógica de Entrada
            if current_pos == 0:
                if data[j, fe_col] > data[j, se_col]: current_pos = 1
                elif data[j, fe_col] < data[j, se_col]: current_pos = -1
            
            sig_buf[j] = current_pos

        # --- 4. FILTRO DE SUPERVIVENCIA (Inline) ---
        is_survivor = True
        for w in range(n_w):
            win_start = w * w_size
            win_end = win_start + w_size
            
            w_drawdown = 0.0
            w_peak = 0.0
            w_cum_ret = 0.0
            
            prev_p = 0
            for k in range(win_start, win_end):
                # Retorno de la barra (retorno * posición_anterior)
                r = data[k, 1] * prev_p
                # Comisiones (solo si cambia posición)
                if sig_buf[k] != prev_p:
                    r -= abs(sig_buf[k] - prev_p) * comm
                
                w_cum_ret += r
                if w_cum_ret > w_peak: w_peak = w_cum_ret
                dd = w_peak - w_cum_ret
                if dd > w_drawdown: w_drawdown = dd
                prev_p = sig_buf[k]

            if w_drawdown > thresh:
                is_survivor = False
                break
        
        if not is_survivor:
            continue

        survivors += 1
        # --- 5. CÁLCULO DE MÉTRICAS (Profit Factor Inline) ---
        gross_profits = 0.0
        gross_losses = 0.0
        prev_p = 0
        for j in range(n_bars):
            r = data[j, 1] * prev_p
            if sig_buf[j] != prev_p:
                r -= abs(sig_buf[j] - prev_p) * comm
            
            if r > 0: gross_profits += r
            elif r < 0: gross_losses += abs(r)
            prev_p = sig_buf[j]
            
        pf = gross_profits / gross_losses if gross_losses > 0 else (100.0 if gross_profits > 0 else 0.0)
        
        if pf > 1.0:
            if pf > best_pf:
                best_pf = pf
                best_idx = i
                
    return best_pf, best_idx, survivors

class ValidationOrchestrator:
    """
    Orquestador principal del Framework.
    Gestiona el pipeline completo desde la carga de datos hasta la validación OOS.
    """

    def __init__(self, strategy, config):
        self.strategy = strategy
        self.config = config
        self.processor = DataProcessor(config.dataset_path)
        self.logger = AsyncLogger()
        self.backtester = BacktesterEngine()
        self.survival_tester = SurvivalTester(n_windows=4, threshold_pf=1.1)
        self.permutation_tester = PermutationTest(n_permutations=500)
        self.oos_validator = ValidatorOOS(self.logger)

    def _warmup_jit(self, shm_name, shm_shape, map_matrix):
        """
        Pre-calienta el motor JIT en el hilo principal antes de lanzar el pool.
        Esto evita la 'tormenta de compilación' en los workers.
        """
        self.logger.log("Pre-calentando motor 'Bola de Fuego' (Warming up JIT)...")
        try:
            # Pre-alocación mínima para warmup
            sig_buf = np.zeros(shm_shape[0], dtype=np.int8)
            ret_buf = np.zeros(shm_shape[0], dtype=np.float64)
            
            # Conectar localmente
            from multiprocessing.shared_memory import SharedMemory
            self._warmup_shm = SharedMemory(name=shm_name)
            data = np.ndarray(shm_shape, dtype=np.float64, buffer=self._warmup_shm.buf)
            
            # Ejecutar un pequeño bloque para forzar compilación
            _ = _njit_massive_loop_master(
                0, 100, 
                data, 
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf, ret_buf
            )
            self.logger.log("Motor JIT listo y caliente.")
        except Exception as e:
            self.logger.log(f"ERROR DURANTE EL WARMUP JIT: {str(e)}")
            import traceback
            self.logger.log(traceback.format_exc())

    def _run_diagnostic_benchmark(self, shm_name, shm_shape, map_matrix):
        """
        Ejecuta una muestra ultra-pequeña para diagnosticar la salud del motor.
        """
        self.logger.log("-" * 30)
        self.logger.log("INICIANDO DIAGNÓSTICO DE PERFORMANCE...")
        
        n_test = 1000
        try:
            self.logger.log(f"Configurando buffers para {n_test} backtests...")
            sig_buf = np.zeros(shm_shape[0], dtype=np.int8)
            ret_buf = np.zeros(shm_shape[0], dtype=np.float64)
            
            self.logger.log("Conectando a buffers de memoria compartida...")
            from multiprocessing.shared_memory import SharedMemory
            self._diag_shm = SharedMemory(name=shm_name)
            data = np.ndarray(shm_shape, dtype=np.float64, buffer=self._diag_shm.buf)
            
            self.logger.log("Ejecutando motor de cálculo...")
            t_start = time.perf_counter()
            _ = _njit_massive_loop_master(
                0, n_test, 
                data, 
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf, ret_buf
            )
            t_end = time.perf_counter()
            
            ms_total = (t_end - t_start) * 1000
            ms_per_bt = ms_total / n_test
            bt_per_sec = n_test / (t_end - t_start)
            
            self.logger.log(f"Muestra: {n_test:,} Backtests")
            self.logger.log(f"Tiempo Total: {ms_total:.2f} ms")
            self.logger.log(f"Latencia Media: {ms_per_bt:.4f} ms/BT")
            self.logger.log(f"Velocidad Base (1 Core): {bt_per_sec:,.0f} BT/s")
            self.logger.log("-" * 30)
            
            # Liberar Shm locales del main process para el diagnóstico (no unlink)
            self._diag_shm.close()
            if hasattr(self, '_warmup_shm'):
                self._warmup_shm.close()
        except Exception as diag_err:
            self.logger.log(f"ERROR DURANTE EL DIAGNÓSTICO: {str(diag_err)}")
            import traceback
            self.logger.log(traceback.format_exc())

    def run_pipeline(self):
        """
        Ejecuta el pipeline de validación extremo con soporte para templates.
        """
        start_time = time.time()
        self.logger.log(f"Iniciando Pipeline para estrategia: {self.strategy.name}")
        self.logger.log(f"Configuración cargada: {self.config.dataset_path} | Comm: {self.config.commission_bps} bps")

        try:
            # 1. Carga y Procesamiento
            self.processor.load_data()
            
            # --- FASE CRÍTICA: PRE-CÁLCULO MASIVO (Speed Hack) ---
            ma_types = [
                'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'tma',
                'rma', 'zlema', 'kama', 'alma', 'vwma', 'mcginley', 'vidya'
            ]
            periods = list(range(5, 205, 5))
            periods.insert(0, 1)

            self.logger.log(f"Iniciando Pre-cálculo masivo de {len(ma_types) * len(periods)} indicadores...")
            indicator_map = self.processor.precompute_indicators(ma_types, periods)
            
            # Dividir IS/OOS DESPUÉS del pre-cálculo para que las vistas incluyan los indicadores
            is_data, oos_data = self.processor.split_is_oos(0.7)
            self.logger.log(f"Datos cargados y expandidos. IS: {len(is_data)} | OOS: {len(oos_data)}")

            # Convertimos el indicator_map (dict) a una matriz NP para NJIT
            map_matrix = np.zeros((14, 41), dtype=np.int32)
            for (ma_name, p), col_idx in indicator_map.items():
                ma_idx = ma_types.index(ma_name)
                p_idx = 0 if p == 1 else p // 5
                map_matrix[ma_idx, p_idx] = col_idx
            
            self.logger.log("Matriz de indicadores expandida generada.")

            # 2. Exportar IS expandido a Shared Memory para Workers
            original_full_data = self.processor.processed_data
            self.processor.processed_data = is_data 
            shm_name, shm_shape = self.processor.create_shared_buffer()
            self.processor.processed_data = original_full_data # Restauramos
            
            # 3. Calentamiento JIT para evitar colisiones
            self._warmup_jit(shm_name, shm_shape, map_matrix)
            
            # 4. Diagnóstico de Performance (Solicitado por el usuario)
            self._run_diagnostic_benchmark(shm_name, shm_shape, map_matrix)

            # 5. Optimización / Backtesting Masivo (In-Sample) con Motor "Bola de Fuego"
            total_p = 9413600
            
            # --- USO DE MULTIPROCESSING.POOL ULTRA-BALANCEADO ---
            import multiprocessing as mp
            # Usamos un número bajo de núcleos (4) para asegurar que el sistema respire
            n_cores = 4
            self.logger.log(f"--- MOTOR MULTICORE 'BOLA DE FUEGO' ({n_cores} Cores - Shared Memory + Buffer Reuse) ---")
            
            best_pf = -1
            winner_params = None
            survivors_count = 0
            processed_count = 0
            
            with mp.Pool(
                processes=n_cores,
                initializer=init_worker,
                initargs=(
                    shm_name,
                    shm_shape,
                    self.strategy.legal_pairs,
                    map_matrix
                )
            ) as pool:
                chunk_size = 100000 
                tasks = []
                for i in range(0, total_p, chunk_size):
                    end = min(i + chunk_size, total_p)
                    tasks.append(((i, end), self.strategy.__class__, self.config.commission_bps))
                
                try:
                    for res, count in pool.imap_unordered(_worker_bt_chunk, tasks):
                        if isinstance(res, str) and res.startswith("ERROR"):
                            self.logger.log(res)
                        else:
                            processed_count += count
                            survivors_count += len(res)
                            
                            for p, m in res:
                                if m['profit_factor'] > best_pf:
                                    best_pf = m['profit_factor']
                                    winner_params = p

                            # Log más frecuente para ver el arranque (cada chunk)
                            if processed_count % chunk_size == 0 or processed_count >= total_p:
                                elapsed = time.time() - start_time
                                throughput = processed_count / elapsed
                                pct = (processed_count / total_p) * 100
                                self.logger.log(f"[{pct:5.1f}%] {processed_count:,} BTs | Speed: {throughput:6.0f} BT/s | Best PF: {best_pf:.2f}")
                except Exception as pool_err:
                    self.logger.log(f"FALLO CRÍTICO EN EL POOL: {str(pool_err)}")
                finally:
                    # SIEMPRE cerrar el pool pase lo que pase
                    pool.terminate()
                    pool.join()
                    # Disparar también al guardián para estar 100% seguros
                    ProcessGuard.get_instance().cleanup()

            # Limpieza de Memoria Compartida
            if self.processor._shm:
                try:
                    self.processor._shm.close()
                    self.processor._shm.unlink()
                    self.logger.log("Memoria compartida liberada correctamente.")
                except:
                    pass

            self.logger.log(f"Supervivientes: {survivors_count} de {processed_count} procesados.")
            
            if winner_params is None:
                self.logger.log("CRITICAL: Ninguna estrategia superó el filtro de supervivencia (PF > 1.0).")
                return None

            # El ganador IS requiere recalcular su señal final para los siguientes pasos
            winner_signals_is = self.strategy.generate_signal(is_data, winner_params)

            self.logger.log(f"Ganador IS: {winner_params} | PF: {best_pf:.2f}")

            # 3. Permutation Test (Bar-Shuffling) sobre el Ganador
            self.logger.log("Ejecutando Permutation Test (Monte Carlo)...")
            p_val = self.permutation_tester.run_test(is_data[:, 1], winner_signals_is, best_pf)
            status = "(Robusto)" if p_val < 0.05 else "(Poco Robusto)"
            self.logger.log(f"Quasi P-Value: {p_val:.4f} {status}")

            # 6. Validación Final OOS
            self.logger.log("--- FASE FINAL: VALIDACIÓN OUT-OF-SAMPLE ---")
            winner_signals_oos = self.strategy.generate_signal(oos_data, winner_params)
            oos_results = self.oos_validator.validate(
                oos_data, 
                winner_signals_oos, 
                str(winner_params), 
                is_pf=best_pf, 
                commission_bps=self.config.commission_bps
            )

            total_time = time.time() - start_time
            self.logger.log(f"Pipeline completado en {total_time:.2f} segundos.")
            
            return oos_results

        except Exception as e:
            self.logger.log(f"CRITICAL ERROR en el pipeline: {str(e)}")
            raise e
        
        finally:
            # Detener logger asíncrono siempre para evitar zombies
            time.sleep(1)
            self.logger.stop()
