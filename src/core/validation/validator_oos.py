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

    def validate(self, oos_data: np.ndarray, signals: np.ndarray, config_name: str, 
                 is_pf: float = 0.0, commission_bps: float = 0.0) -> dict:
        """
        Ejecuta la validación final y reporta resultados.
        Compara contra el rendimiento IS para detectar sobre-ajuste e incluye comisiones.
        """
        if self.logger:
            self.logger.log(f"Iniciando Validación OOS para: {config_name}")

        # Cálculo de métricas finales integrando comisiones
        results = self.engine.run(oos_data, signals, commission_bps=commission_bps)
        oos_pf = results['profit_factor']

        if self.logger:
            msg = f"OOS Completado: PF={oos_pf:.2f} | " \
                  f"Return={results['total_return']*100:.2f}%"
            self.logger.log(msg)
            
            # --- DETECCIÓN DE OVERFITTING ---
            if is_pf > 0:
                drop_off = (is_pf - oos_pf) / is_pf
                if drop_off > 0.30:
                    warn_msg = f"CRITICAL WARNING: Detectado drop-off de {drop_off*100:.1f}% en OOS. " \
                               f"Posible sobre-ajuste (IS PF: {is_pf:.2f} -> OOS PF: {oos_pf:.2f})"
                    self.logger.log(warn_msg)
                else:
                    self.logger.log(f"Integridad OOS confirmada (Drop-off: {drop_off*100:.1f}%)")

        return results
