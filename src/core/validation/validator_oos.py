"""
Validador final para la ejecución sobre datos fuera de muestra (OOS).
"""
import numpy as np
from src.core.backtester import BacktesterEngine
from src.utils.logger import AsyncLogger

class ValidatorOOS:
    """
    Módulo final de ejecución sobre datos reservados (Out-of-Sample).
    Asegura que el rendimiento se mantiene fuera del set de entrenamiento.
    """

    def __init__(self, logger: AsyncLogger = None):
        self.logger = logger
        self.engine = BacktesterEngine()

    def validate(self, oos_data: np.ndarray, signals: np.ndarray, config_name: str) -> dict:
        """
        Ejecuta la validación final y reporta resultados.
        """
        if self.logger:
            self.logger.log(f"Iniciando Validación OOS para: {config_name}")

        # Cálculo de métricas finales
        results = self.engine.run(oos_data, signals)

        if self.logger:
            msg = f"OOS Completado: PF={results['profit_factor']:.2f} | " \
                  f"Return={results['total_return']*100:.2f}%"
            self.logger.log(msg)

        return results
