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

    def __init__(self, strategy, data_path: str):
        self.strategy = strategy
        self.processor = DataProcessor(data_path)
        self.logger = AsyncLogger()
        self.backtester = BacktesterEngine()
        self.survival_tester = SurvivalTester(n_windows=4, threshold_pf=1.1)
        self.permutation_tester = PermutationTest(n_permutations=500)
        self.oos_validator = ValidatorOOS(self.logger)

    def run_pipeline(self):
        """
        Ejecuta el pipeline de validación extremo.
        """
        start_time = time.time()
        self.logger.log(f"Iniciando Pipeline para estrategia: {self.strategy.name}")

        # 1. Carga y Procesamiento
        self.processor.load_data()
        is_data, oos_data = self.processor.split_is_oos(0.7)
        self.logger.log(f"Datos cargados. IS: {len(is_data)} | OOS: {len(oos_data)}")

        # 2. Optimización / Backtesting Masivo (In-Sample)
        params_list = self.strategy.param_grid
        self.logger.log(f"Procesando {len(params_list)} permutaciones...")

        is_results = []
        is_signals_matrix = []

        for params in params_list:
            signals = self.strategy.generate_signal(is_data, params)
            metrics = self.backtester.run(is_data, signals)
            is_results.append((params, metrics))
            is_signals_matrix.append(signals)

        is_signals_matrix = np.array(is_signals_matrix)

        # 3. Test de Supervivencia (IS)
        self.logger.log("Ejecutando Test de Supervivencia sobre IS...")
        survival_mask = self.survival_tester.compute_survival_matrix(is_data, is_signals_matrix)
        survivors_count = np.sum(survival_mask)
        self.logger.log(f"Supervivientes: {survivors_count} de {len(params_list)}")

        if survivors_count == 0:
            msg = "FAIL: Ninguna configuración sobrevivió al filtro de robustez temporal."
            self.logger.log(msg)
            self.logger.stop()
            return None

        # 4. Selección del Ganador (Basado en PF en IS)
        best_pf = -1
        winner_idx = -1
        for idx, (_, metrics) in enumerate(is_results):
            if survival_mask[idx] and metrics['profit_factor'] > best_pf:
                best_pf = metrics['profit_factor']
                winner_idx = idx

        winner_params, _ = is_results[winner_idx]
        self.logger.log(f"Ganador IS: {winner_params} | PF: {best_pf:.2f}")

        # 5. Permutation Test (Bar-Shuffling) sobre el Ganador
        self.logger.log("Ejecutando Permutation Test (Monte Carlo)...")
        winner_signals_is = is_signals_matrix[winner_idx]
        p_val = self.permutation_tester.run_test(is_data[:, 1], winner_signals_is, best_pf)
        status = "(Robusto)" if p_val < 0.05 else "(Poco Robusto)"
        self.logger.log(f"Quasi P-Value: {p_val:.4f} {status}")

        # 6. Validación Final OOS
        self.logger.log("--- FASE FINAL: VALIDACIÓN OUT-OF-SAMPLE ---")
        winner_signals_oos = self.strategy.generate_signal(oos_data, winner_params)
        oos_results = self.oos_validator.validate(oos_data, winner_signals_oos, str(winner_params))

        total_time = time.time() - start_time
        self.logger.log(f"Pipeline completado en {total_time:.2f} segundos.")

        # Detener logger asíncrono
        time.sleep(1)
        self.logger.stop()

        return oos_results
