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
from src.core.db_manager import BacktestDB
from src.utils.logger import AsyncLogger
from src.utils.process_guard import ProcessGuard
from src.core.reporting.reporter import ResultsReporter
from src.core.jit_ops import _extract_trades_jit, _compute_metrics_inplace_jit

# --- INFRAESTRUCTURA DE NÚCLEOS AUTÓNOMOS ---
_worker_indicator_matrix = None
_worker_data = None
_worker_legal_pairs = None
_worker_sig_buf = None

def init_worker(shm_name, shm_shape, legal_pairs, indicator_matrix):
    """Inicializa el núcleo conectándose a la memoria compartida (Zero-Copy)."""
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs, _worker_shm, _worker_sig_buf
    from src.core.data_processor import DataProcessor
    from multiprocessing.shared_memory import SharedMemory
    import os
    
    try:
        # Debug para Windows: Confirmar que el proceso hijo ha nacido
        # print(f"[DEBUG-WORKER] Nacido Proceso {os.getpid()} - Conectando a {shm_name}...")
        
        _worker_legal_pairs = legal_pairs
        _worker_indicator_matrix = indicator_matrix
        
        # IMPORTANTE: Mantenemos la referencia de _worker_shm para que el buffer no se cierre en Windows
        _worker_shm = SharedMemory(name=shm_name)
        _worker_data = np.ndarray(shm_shape, dtype=np.float64, buffer=_worker_shm.buf, order='F')
        
        # Asignación de memoria una sola vez por núcleo
        _worker_sig_buf = np.zeros(shm_shape[0], dtype=np.int8)
        
        # print(f"[DEBUG-WORKER] Proceso {os.getpid()} listo y conectado.")
    except Exception as e:
        print(f"[FATAL-WORKER] Error al inicializar proceso {os.getpid()}: {str(e)}")
        raise e

def _worker_bt_chunk(args):
    """
    Función worker que utiliza el bucle maestro NJIT sobre datos compartidos.
    """
    range_tuple, strategy_class, commission_bps = args
    global _worker_indicator_matrix, _worker_data, _worker_legal_pairs, _worker_sig_buf
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
        
        # --- MOTOR BOLA DE FUEGO ---
        best_pf, best_idx, survivors = _njit_massive_loop_master(
            start_idx, end_idx,
            _worker_data,
            _worker_legal_pairs,
            _worker_indicator_matrix,
            n_windows, window_size, threshold,
            comm_factor,
            _worker_sig_buf
        )
        
        winner_res = []
        if best_idx != -1:
            strategy = strategy_class()
            winner_params = strategy.get_params_by_index(best_idx)
            winner_res.append((winner_params, {"profit_factor": best_pf}))
            
        return winner_res, (end_idx - start_idx)

    except Exception as e:
        return f"ERROR: {str(e)}", 0

@njit(fastmath=True, error_model='numpy', cache=True)
def _njit_massive_loop_master(start_idx, end_idx, data, legal_pairs, map_matrix, n_w, w_size, thresh, comm, sig_buf):
    """
    Motor Flex-MA v3: Indización independiente para 1.7B combinaciones.
    Todo inlined para máximo performance.
    """
    n_bars = data.shape[0]
    n_pairs = legal_pairs.shape[0]
    n_types = map_matrix.shape[0]
    best_pf = -1.0
    best_idx = -1
    survivors = 0
    
    for i in range(start_idx, end_idx):
        # --- 1. DECODER FLEX-MA (Inline) ---
        idx_pairs = i % (n_pairs * n_pairs)
        rem_types = i // (n_pairs * n_pairs)
        
        pair_exit_idx = idx_pairs % n_pairs
        pair_entry_idx = idx_pairs // n_pairs
        
        t4 = rem_types % n_types
        rem_types //= n_types
        t3 = rem_types % n_types
        rem_types //= n_types
        t2 = rem_types % n_types
        t1 = rem_types // n_types
        
        p_entry = legal_pairs[pair_entry_idx]
        p_exit = legal_pairs[pair_exit_idx]
        
        # --- 2. RESOLVER COLUMNAS (Inline) ---
        # Resolución step=10
        fe_col = map_matrix[t1, 0 if p_entry[0] == 1 else p_entry[0] // 10]
        se_col = map_matrix[t2, 0 if p_entry[1] == 1 else p_entry[1] // 10]
        fx_col = map_matrix[t3, 0 if p_exit[0] == 1 else p_exit[0] // 10]
        sx_col = map_matrix[t4, 0 if p_exit[1] == 1 else p_exit[1] // 10]

        fe_raw = data[:, fe_col]
        se_raw = data[:, se_col]
        fx_raw = data[:, fx_col]
        sx_raw = data[:, sx_col]
        ret_raw = data[:, 1]
        
        # --- 3. GENERACIÓN DE SEÑALES Y CÁLCULO DE MÉTRICAS (Trade-Based + Pruning) ---
        current_pos = 0
        prev_pos = 0
        gross_profits = 0.0
        gross_losses = 0.0
        current_trade_ret = 0.0
        
        for j in range(n_bars):
            # Lógica de Salida
            new_pos = current_pos
            if current_pos == 1:
                if fx_raw[j] < sx_raw[j]: new_pos = 0
            elif current_pos == -1:
                if fx_raw[j] > sx_raw[j]: new_pos = 0
            
            # Lógica de Entrada
            if new_pos == 0:
                if fe_raw[j] > se_raw[j]: new_pos = 1
                elif fe_raw[j] < se_raw[j]: new_pos = -1
            
            # --- CÁLCULO DE MÉTRICAS TRADE-BASED INLINE ---
            # Acumular retorno de la barra antes de cambiar posición
            if current_pos != 0:
                current_trade_ret += current_pos * ret_raw[j]
            
            # Detectar cambio para aplicar comisiones y cerrar trades
            if new_pos != current_pos:
                cost = abs(new_pos - current_pos) * comm
                if current_pos != 0:
                    # Cerramos trade o Reversal
                    if new_pos == 0 or (new_pos * current_pos < 0):
                        net_trade_ret = current_trade_ret - cost
                        if net_trade_ret > 0: gross_profits += net_trade_ret
                        else: gross_losses += abs(net_trade_ret)
                        current_trade_ret = 0.0
                    else:
                        current_trade_ret -= cost # Aumento de posición
                else:
                    # Entrada desde 0
                    current_trade_ret -= cost
            
            current_pos = new_pos
            sig_buf[j] = current_pos

        # Forzar cierre del último trade
        if current_pos != 0:
            net_trade_ret = current_trade_ret - abs(current_pos) * comm
            if net_trade_ret > 0: gross_profits += net_trade_ret
            else: gross_losses += abs(net_trade_ret)

        # --- 4. ETAPA DE PRUNING TEMPRANO ---
        pf = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)
        
        if pf <= 1.0:
            continue

        # --- 5. FILTRO DE SUPERVIVENCIA (Solo para ganadoras) ---
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
            _ = _njit_massive_loop_master(
                0, 100, 
                data, 
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf
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
                self.strategy.legal_pairs, 
                map_matrix, 
                4, len(data) // 4, 1.1, 0.0006, 
                sig_buf
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
            self._run_diagnostic_benchmark(shm_name, shm_shape, map_matrix)

            # 5. Optimización / Backtesting Masivo (In-Sample) con Motor "Bola de Fuego"
            # 14^4 * 210^2
            total_combinations = (self.strategy.n_types ** 4) * (self.strategy.n_pairs ** 2)
            
            # --- SISTEMA DE CHECKPOINT ---
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

            # --- USO DE MULTIPROCESSING.POOL ULTRA-BALANCEADO ---
            import multiprocessing as mp
            n_cores = 12
            self.logger.log(f"--- MOTOR FLEX-MA ({n_cores} Cores - Checkpointing Activo) ---")
            
            # Super-Batches de 50M para Checkpointing
            super_batch_size = 50_000_000
            processed_total = start_idx
            
            for sb_start in range(start_idx, total_combinations, super_batch_size):
                sb_end = min(sb_start + super_batch_size, total_combinations)
                self.logger.log(f"Iniciando Super-Batch: [{sb_start:,} -> {sb_end:,}]")
                
                all_is_results = []
                with mp.Pool(
                    processes=n_cores,
                    initializer=init_worker,
                    initargs=(shm_name, shm_shape, self.strategy.legal_pairs, map_matrix)
                ) as pool:
                    chunk_size = 100_000 # Chunks más grandes para 1.7B
                    tasks = []
                    for i in range(sb_start, sb_end, chunk_size):
                        end = min(i + chunk_size, sb_end)
                        tasks.append(((i, end), self.strategy.__class__, self.config.commission_bps))
                    
                    batch_processed = 0
                    for res, count in pool.imap_unordered(_worker_bt_chunk, tasks):
                        if not isinstance(res, str):
                            batch_processed += count
                            processed_total += count
                            for p, m in res:
                                if m['profit_factor'] > best_pf:
                                    best_pf = m['profit_factor']
                                    best_idx = self.strategy._params_to_idx(p) # Helpert para recuperar índice si es necesario
                                all_is_results.append((p, m))
                        
                        # Logging masivo cada 100k
                        if processed_total % 100_000 == 0:
                            elapsed = time.time() - start_time
                            throughput = processed_total / elapsed if elapsed > 0 else 0
                            msg = f"SPEED: {processed_total:,} BTs | {throughput:,.0f} BT/s | Best PF: {best_pf:.2f}"
                            self._log_to_file(msg)
                            if processed_total % 1_000_000 == 0: # Feedback visual en consola menos frecuente
                                pct = (processed_total / total_combinations) * 100
                                self.logger.log(f"[{pct:5.2f}%] {processed_total:,} BTs | Best PF: {best_pf:.2f}")

                # Fin de Super-Batch: Checkpoint y Persistencia
                self._save_checkpoint(sb_end, best_pf, best_idx)
                self._log_to_file(f"CHECKPOINT: Guardado en {sb_end:,}")
                if all_is_results:
                    self.db.save_is_batch(self.run_id, all_is_results)

            # 5.5 Persistencia masiva final (fuera del bucle crítico para maximizar throughput)
            if all_is_results:
                self.logger.log(f"Guardando {len(all_is_results):,} estrategias IS en la base de datos de auditoría...")
                self.db.save_is_batch(self.run_id, all_is_results)

            self.logger.log(f"Supervivientes que superaron el filtro IS: {survivors_count} de {processed_count} procesados.")
            
            if not all_is_results:
                self.logger.log("CRITICAL: Ninguna estrategia superó el filtro de supervivencia (PF > 1.0).")
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
