import time
import os
import multiprocessing
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.core.data_processor import DataProcessor
from src.core.backtester import BacktesterEngine
from src.core.jit_ops import (
    _njit_execution_engine,
    _extract_trades_jit,
    _compute_metrics_inplace_jit,
    _check_single_survival_jit
)
from src.core.validation.survival_tester import SurvivalTester
from src.core.validation.permutation_test import PermutationTest
from src.core.validation.validator_oos import ValidatorOOS
from src.core.db_manager import BacktestDB
from src.utils.logger import AsyncLogger
from src.utils.process_guard import ProcessGuard
from src.core.reporting.reporter import ResultsReporter

import json

# --- INFRAESTRUCTURA DE NÚCLEOS AUTÓNOMOS ---
_worker_indicator_matrix = None
_worker_data = None
_worker_legal_pairs = None
_worker_sig_buf = None
_worker_shared_metrics = None # Obsoleto

def init_worker(shm_name, shm_shape, legal_pairs, indicator_matrix):
    """Inicializa el núcleo conectándose a la memoria compartida (Zero-Copy)."""
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs, _worker_shm, _worker_sig_buf
    from src.core.data_processor import DataProcessor
    from multiprocessing.shared_memory import SharedMemory
    
    try:
        _worker_legal_pairs = legal_pairs
        _worker_indicator_matrix = indicator_matrix
        
        _worker_shm = SharedMemory(name=shm_name)
        _worker_data = np.ndarray(shm_shape, dtype=np.float64, buffer=_worker_shm.buf, order='F')
        _worker_sig_buf = np.zeros(shm_shape[0], dtype=np.int8)
    except Exception as e:
        print(f"[FATAL-WORKER] Error al inicializar proceso: {str(e)}")
        raise e

def _worker_bt_chunk(args):
    """
    Función worker que utiliza el bucle maestro NJIT sobre datos compartidos.
    """
    range_tuple, strategy_class, commission_bps = args
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs, _worker_sig_buf, _worker_shared_metrics
    start_idx, end_idx = range_tuple
    
    try:
        if _worker_data is None or _worker_sig_buf is None:
            return "ERROR: Buffers no inicializados", 0
            
        # Parámetros de supervivencia
        n_windows = 4
        threshold = 1.0
        n_samples = len(_worker_data)
        window_size = n_samples // n_windows
        comm_factor = commission_bps / 10000.0

        # --- MOTOR HFT-ENGINE ---
        bh_ret = np.sum(_worker_data[:, 1])
        
        res = _njit_massive_loop_master(
            start_idx, end_idx,
            _worker_data,
            strategy_class().get_logic_id(),
            _worker_legal_pairs, 
            _worker_indicator_matrix,
            n_windows, window_size, threshold,
            comm_factor,
            _worker_sig_buf,
            bh_ret
        )
        
        return res

    except Exception as e:
        return (0.0, -1, 0, 0, 0)

@njit(fastmath=True, error_model='numpy', cache=True)
def _njit_massive_loop_master(start_idx, end_idx, data, logic_id, strategy_metadata, map_matrix, n_w, w_size, thresh, comm, sig_buf, bh_ret):
    """
    Motor HFT-ENGINE v5: Arquitectura consolidada con Pruning Multinivel.
    Capa 1: Retorno vs Buy & Hold (0.7x).
    Capa 2: Profit Factor > 1.0.
    Capa 3: Filtro de Supervivencia (Drawdown).
    """
    best_pf = -1.0
    best_idx = -1
    survivors = 0
    cap1_survivors = 0
    evaluated_total = 0
    
    n_pairs = strategy_metadata.shape[0] if logic_id == 1 else 0
    n_types = map_matrix.shape[0]
    ret_raw = data[:, 1]
    n_bars = data.shape[0]
    
    for i in range(start_idx, end_idx):
        evaluated_total += 1
        
        # --- 1. DECODER DE PARÁMETROS ---
        params_vec = np.zeros(8, dtype=np.float64) 
        if logic_id == 1:
            # Inline FlexMA Decoder
            idx_pairs = i % (n_pairs * n_pairs)
            rem_types = i // (n_pairs * n_pairs)
            p_entry = strategy_metadata[idx_pairs // n_pairs]
            p_exit = strategy_metadata[idx_pairs % n_pairs]
            t4 = rem_types % n_types
            rem_types //= n_types
            t3 = rem_types % n_types
            rem_types //= n_types
            t2 = rem_types % n_types
            t1 = rem_types // n_types
            
            params_vec[0] = map_matrix[t1, 0 if p_entry[0] == 1 else p_entry[0] // 10]
            params_vec[1] = map_matrix[t2, 0 if p_entry[1] == 1 else p_entry[1] // 10]
            params_vec[2] = map_matrix[t3, 0 if p_exit[0] == 1 else p_exit[0] // 10]
            params_vec[3] = map_matrix[t4, 0 if p_exit[1] == 1 else p_exit[1] // 10]

        # --- 2. EJECUCIÓN DE SEÑALES ---
        _njit_execution_engine(data, logic_id, params_vec, sig_buf)

        # --- 3. PRUNING CAPA 1 (Retorno vs BH 0.7x) ---
        strat_ret = 0.0
        current_pos = 0
        for j in range(n_bars):
            new_pos = sig_buf[j]
            if current_pos != 0: strat_ret += current_pos * ret_raw[j]
            if new_pos != current_pos: strat_ret -= abs(new_pos - current_pos) * comm
            current_pos = new_pos
        if current_pos != 0: strat_ret -= abs(current_pos) * comm

        if strat_ret < (0.7 * bh_ret):
            continue
        cap1_survivors += 1

        # --- 4. MÉTRICAS TRADE-BASED (Agnóstico) ---
        gross_profits = 0.0
        gross_losses = 0.0
        current_trade_ret = 0.0
        current_pos = 0
        
        for j in range(n_bars):
            new_pos = sig_buf[j]
            if current_pos != 0: current_trade_ret += current_pos * ret_raw[j]
            if new_pos != current_pos:
                cost = abs(new_pos - current_pos) * comm
                if current_pos != 0:
                    if new_pos == 0 or (new_pos * current_pos < 0):
                        net_trade_ret = current_trade_ret - cost
                        if net_trade_ret > 0: gross_profits += net_trade_ret
                        else: gross_losses += abs(net_trade_ret)
                        current_trade_ret = 0.0
                    else: current_trade_ret -= cost
                else: current_trade_ret -= cost
            current_pos = new_pos

        if current_pos != 0:
            net_trade_ret = current_trade_ret - abs(current_pos) * comm
            if net_trade_ret > 0: gross_profits += net_trade_ret
            else: gross_losses += abs(net_trade_ret)

        pf = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)
        
        # PRUNING CAPA 2 (Rentabilidad)
        if pf <= 1.0:
            continue

        # --- 5. FILTRO DE SUPERVIVENCIA (Drawdown) ---
        is_survivor = True
        for w in range(n_w):
            win_start = w * w_size
            win_end = win_start + w_size
            w_drawdown = 0.0
            w_peak = 0.0
            w_cum_ret = 0.0
            prev_p = 0
            for k in range(win_start, win_end):
                r = ret_raw[k] * prev_p
                if sig_buf[k] != prev_p: r -= abs(sig_buf[k] - prev_p) * comm
                w_cum_ret += r
                if w_cum_ret > w_peak: w_peak = w_cum_ret
                dd = w_peak - w_cum_ret
                if dd > w_drawdown: w_drawdown = dd
                prev_p = sig_buf[k]
            if w_drawdown > thresh:
                is_survivor = False
                break
        
        if not is_survivor: continue

        survivors += 1
        if pf > best_pf:
            best_pf = pf
            best_idx = i
                
    return best_pf, best_idx, survivors, float(evaluated_total), float(cap1_survivors)

def _monitor_thread_func(metrics_dict, stop_event, run_id):
    """Hilo independiente que reporta telemetría cada 10 segundos."""
    from src.utils.logger import AsyncLogger
    import time
    logger = AsyncLogger(run_id)
    last_audit = 0
    t_last = time.time()
    
    logger.log("[MONITOR] Telemetría asíncrona (Thread-mode) iniciada.")
    
    try:
        while not stop_event.is_set():
            time.sleep(10)
            
            audit = metrics_dict["audit"]
            discard = metrics_dict["discard"]
            survive = metrics_dict["survive"]
            
            t_now = time.time()
            dt = t_now - t_last
            new_audit = audit - last_audit
            speed = new_audit / dt if dt > 0 else 0
            
            discard_pct = (discard / audit * 100) if audit > 0 else 0
            
            logger.log(
                f"[TELEMETRÍA] | Auditadas: {audit:,} | Descartadas C1: {discard:,} "
                f"({discard_pct:.1f}%) | Supervivientes: {survive:,} | "
                f"Velocidad: {speed:,.0f} cfg/s"
            )
            
            last_audit = audit
            t_last = t_now
            
    except Exception as e:
        print(f"[MONITOR-ERROR] {str(e)}")
    finally:
        logger.log("[MONITOR] Telemetría finalizada.")
        logger.stop()

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
        self.db = BacktestDB()
        self.run_id = None
        self.checkpoint_path = "results/checkpoints/flex_checkpoint.json"
        self.log_file_path = "results/logs/flex_bt.log"

    def _save_checkpoint(self, last_idx, best_pf, best_idx):
        """Guarda el progreso y el estado de la optimización masiva."""
        checkpoint = {
            "last_idx": int(last_idx),
            "best_pf": float(best_pf),
            "best_idx": int(best_idx),
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=4)

    def _load_checkpoint(self):
        """Carga el progreso previo si existe."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                return json.load(f)
        return None

    def _log_to_file(self, message):
        """Escribe logs enriquecidos en el archivo dedicado."""
        with open(self.log_file_path, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            # Texto enriquecido simple (ASCII art / decoradores)
            if "SPEED" in message.upper():
                decorator = ">>> "
            elif "CHECKPOINT" in message.upper():
                decorator = "[OK] "
            else:
                decorator = "    "
            f.write(f"[{timestamp}] {decorator}{message}\n")

    def _warmup_jit(self, shm_name, shm_shape, map_matrix):
        """
        Pre-calienta el motor JIT en el hilo principal antes de lanzar el pool.
        Esto evita la 'tormenta de compilación' en los workers.
        """
        self.logger.log("Pre-calentando motor 'Bola de Fuego' (Warming up JIT)...")
        try:
            # Pre-alocación mínima para warmup
            sig_buf = np.zeros(shm_shape[0], dtype=np.int8)
            
            # Conectar localmente
            from multiprocessing.shared_memory import SharedMemory
            self._warmup_shm = SharedMemory(name=shm_name)
            data = np.ndarray(shm_shape, dtype=np.float64, buffer=self._warmup_shm.buf, order='F')
            
            # Ejecutar un pequeño bloque para forzar compilación
            bh_ret = np.sum(data[:, 1])
            _ = _njit_massive_loop_master(
                0, 100, 
                data, 
                self.strategy.get_logic_id(),
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf,
                bh_ret
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
            
            self.logger.log("Conectando a buffers de memoria compartida...")
            from multiprocessing.shared_memory import SharedMemory
            self._diag_shm = SharedMemory(name=shm_name)
            data = np.ndarray(shm_shape, dtype=np.float64, buffer=self._diag_shm.buf, order='F')
            
            self.logger.log("Ejecutando motor de cálculo...")
            t_start = time.perf_counter()
            _ = _njit_massive_loop_master(
                0, n_test, 
                data, 
                self.strategy.get_logic_id(),
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf,
                np.sum(data[:, 1])
            )
            t_end = time.perf_counter()
            
            ms_total = (t_end - t_start) * 1000
            ms_per_bt = ms_total / n_test
            bt_per_sec = n_test / (t_end - t_start)
            
            # --- PROCESAMIENTO DE RESULTADOS DEL BLOQUE ---
            total_evaluated = 0
            total_c1_survivors = 0
            batch_survivors = 0
            
            # Reporte Performance
            t_end = time.perf_counter()
            elapsed = t_end - t_start
            bt_s = n_test / elapsed
            self.logger.log(f"Block Speed: {bt_s:,.0f} BT/s | Bloque finalizado.")
            
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
        
        # 0. Iniciar trazabilidad en base de datos
        self.run_id = self.db.create_run(
            self.strategy.name, 
            os.path.basename(self.config.dataset_path), # O el nombre del template si estuviera disponible
            self.config.commission_bps,
            self.config.dataset_path
        )
        self.logger.log(f"Sesión de auditoría iniciada. ID: {self.run_id} | DB: {self.db.db_path}")

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
            self._run_diagnostic_benchmark            # 5. Optimización / Backtesting Masivo (In-Sample) con HFT-ENGINE v5
            total_combinations = (self.strategy.n_types ** 4) * (self.strategy.n_pairs ** 2)
            
            checkpoint = self._load_checkpoint()
            start_idx = 0
            best_pf = -1.0
            best_idx = -1
            
            if checkpoint:
                start_idx = checkpoint["last_idx"]
                best_pf = checkpoint["best_pf"]
                best_idx = checkpoint["best_idx"]
                self.logger.log(f"REANUDANDO desde Checkpoint: {start_idx:,} BTs completados.")
                self._log_to_file(f"REANUDANDO PROCESO DESDE {start_idx:,}")

            import multiprocessing as mp
            import threading
            self.metrics_dict = {"audit": start_idx, "discard": 0, "survive": 0}
            self.stop_monitor = threading.Event()
            
            # Lanzar Hilo Monitor
            monitor_thread = threading.Thread(
                target=_monitor_thread_func, 
                args=(self.metrics_dict, self.stop_monitor, self.run_id),
                daemon=True
            )
            monitor_thread.start()

            n_cores = 12
            self.logger.log(f"--- MOTOR HFT-ENGINE ({n_cores} Cores - Capa 1 Activa) ---")
            
            super_batch_size = 50_000_000
            processed_total = start_idx
            all_is_results = []
            
            for sb_start in range(start_idx, total_combinations, super_batch_size):
                sb_end = min(sb_start + super_batch_size, total_combinations)
                self.logger.log(f"Iniciando Super-Batch: [{sb_start:,} -> {sb_end:,}]")
                b_start = time.perf_counter()
                
                block_evaluated = 0
                block_c1_survivors = 0
                
                with mp.Pool(
                    processes=n_cores,
                    initializer=init_worker,
                    initargs=(shm_name, shm_shape, self.strategy.legal_pairs, map_matrix)
                ) as pool:
                    chunk_size = 100_000
                    tasks = []
                    for i in range(sb_start, sb_end, chunk_size):
                        end = min(i + chunk_size, sb_end)
                        tasks.append(((i, end), self.strategy.__class__, self.config.commission_bps))
                    
                    for res in pool.imap_unordered(_worker_bt_chunk, tasks):
                        # res: (best_pf, b_idx, survivors, ev, c1)
                        if len(res) == 5:
                            b_pf, b_idx, surv, ev, c1 = res
                            # Actualizar métricas para el Monitor
                            self.metrics_dict["audit"] += ev
                            self.metrics_dict["discard"] += (ev - c1)
                            self.metrics_dict["survive"] += surv
                            
                            if b_idx != -1 and b_pf > best_pf:
                                best_pf = b_pf
                                best_idx = int(b_idx)
                
                # Marcamos checkpoint
                self._save_checkpoint(sb_end, best_pf, best_idx)

            # --- FINALIZACIÓN DE OPTIMIZACIÓN ---
            self.stop_monitor.set()
            monitor_thread.join(timeout=5)
            
            # Recuperar mejor estrategia para fase final si no hay supervivientes acumulados
            if best_idx != -1:
                all_is_results = [(self.strategy.get_params_by_index(best_idx), {"profit_factor": best_pf})]

            self.logger.log(f"Búsqueda masiva completada. Mejor PF encontrado: {best_pf:.2f}")
            if not all_is_results:
                self.logger.log("CRITICAL: Ninguna estrategia superó el filtro IS.")
                return None

            # --- FASE 6: VALIDACIÓN MASIVA Y REPORTING (Todos los supervivientes) ---
            self.logger.log(f"Iniciando Validación Masiva para {len(all_is_results)} supervivientes...")
            reporter = ResultsReporter(self.run_id)
            final_reports = []
            
            # Arrays de retornos y precios para regeneración rápida
            is_returns = is_data[:, 1]
            oos_returns = oos_data[:, 1]
            is_prices = is_data[:, 0] # Asumimos close en col 0 (original)
            # Nota: DataProcessor expande el dataset, debemos ver qué indice es close.
            # En GGAL_1m: datetime,open,high,low,close,volume. Close es indice 4 original.
            # Si se cargó con processor.load_data(), el processor sabe el índice.
            close_col = self.processor.price_col_idx 
            
            is_prices = is_data[:, close_col]
            oos_prices = oos_data[:, close_col]
            
            commission_factor = self.config.commission_bps / 10000.0

            for idx, (params, metrics_is) in enumerate(all_is_results):
                # 6.1 Generar señales completas (IS + OOS)
                sig_is = self.strategy.generate_signal(is_data, params, indicator_map)
                sig_oos = self.strategy.generate_signal(oos_data, params, indicator_map)
                
                # 6.2 Validación OOS
                metrics_oos = self.oos_validator.validate(
                    oos_data, sig_oos, f"Strategy_{idx}", 
                    is_pf=metrics_is['profit_factor'], 
                    commission_bps=self.config.commission_bps
                )
                
                # 6.3 Montecarlo (Permutation Test)
                p_val = self.permutation_tester.run_test(is_returns, sig_is, metrics_is['profit_factor'])
                
                # 6.4 Extracción de Trades (IS)
                trades_is_raw = _extract_trades_jit(sig_is, is_prices, commission_factor)
                trades_json = reporter.format_trades_to_json(trades_is_raw, is_data[:, 0]) # Usamos timestamp si es fecha
                
                # 6.5 Generación de Gráfico de Equity
                # Re-calculamos retornos de estrategia para el plot
                is_strat_ret = np.zeros(len(is_returns))
                oos_strat_ret = np.zeros(len(oos_returns))
                _compute_metrics_inplace_jit(is_returns, sig_is, commission_factor, is_strat_ret)
                _compute_metrics_inplace_jit(oos_returns, sig_oos, commission_factor, oos_strat_ret)
                
                curve_path = reporter.generate_equity_curve(idx, is_strat_ret, oos_strat_ret, params)
                
                # 6.6 Consolidar reporte
                final_reports.append({
                    "id": idx,
                    "params": params,
                    "metrics_is": metrics_is,
                    "metrics_oos": metrics_oos,
                    "montecarlo_p": p_val,
                    "trades": trades_json,
                    "equity_curve_path": curve_path
                })
                
                if (idx + 1) % 100 == 0 or (idx + 1) == len(all_is_results):
                    self.logger.log(f"Reportes generados: {idx+1}/{len(all_is_results)}")

            # 7. Persistencia Masiva y Reportes Finales
            self.logger.log("Guardando reportes detallados en base de datos...")
            self.db.save_strategy_report_batch(self.run_id, final_reports)
            
            metrics_table_path = reporter.save_consolidated_metrics(final_reports)
            self.logger.log(f"Tabla de métricas consolidada generada en: {metrics_table_path}")

            total_time = time.time() - start_time
            self.logger.log(f"Pipeline masivo completado en {total_time:.2f} segundos.")
            
            return final_reports[0]['metrics_oos'] if final_reports else None # Retornamos algo por compatibilidad

        except Exception as e:
            self.logger.log(f"CRITICAL ERROR en el pipeline: {str(e)}")
            raise e
        
        finally:
            # Detener logger asíncrono siempre para evitar zombies
            time.sleep(1)
            self.logger.stop()
