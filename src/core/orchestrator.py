"""
Orquestador maestro que gestiona el pipeline de validación completo.
"""
import time
import numpy as np
from src.core.data_processor import DataProcessor
from src.core.backtester import BacktesterEngine
from src.core.validation.survival_tester import SurvivalTester
from src.core.validation.permutation_test import PermutationTest
from src.core.validation.validator_oos import ValidatorOOS
from src.utils.logger import AsyncLogger

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
            is_data, oos_data = self.processor.split_is_oos(0.7)
            self.logger.log(f"Datos cargados. IS: {len(is_data)} | OOS: {len(oos_data)}")

            # 2. Optimización / Backtesting Masivo (In-Sample) con Streaming
            params_list = self.strategy.param_grid
            self.logger.log(f"Procesando {len(params_list)} permutaciones en modo Streaming...")

            best_pf = -1
            winner_params = None
            winner_signals_is = None
            survivors_count = 0
            
            # Recolectamos resultados solo de los que sobreviven para ahorrar memoria
            is_results = [] 

            for params in params_list:
                signals = self.strategy.generate_signal(is_data, params)
                
                # Test de Supervivencia Inmediato (Streaming)
                if self.survival_tester.check_single_survival(is_data, signals):
                    survivors_count += 1
                    metrics = self.backtester.run(is_data, signals, commission_bps=self.config.commission_bps)
                    is_results.append((params, metrics))
                    
                    # Actualizar ganador
                    if metrics['profit_factor'] > best_pf:
                        best_pf = metrics['profit_factor']
                        winner_params = params
                        winner_signals_is = signals # Solo guardamos la señal del mejor

            self.logger.log(f"Supervivientes: {survivors_count} de {len(params_list)}")

            if survivors_count == 0:
                msg = "FAIL: Ninguna configuración sobrevivió al filtro de robustez temporal."
                self.logger.log(msg)
                return None

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
