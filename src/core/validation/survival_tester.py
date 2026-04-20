"""
Módulo para el filtrado de robustez temporal mediante matrices de supervivencia.
"""
import numpy as np
from numba import njit, prange
from src.core.backtester import _compute_metrics_jit

# Fallback para linters estáticos que no reconocen prange como iterable
if False:  # pylint: disable=using-constant-test
    prange = range

class SurvivalTester:
    """
    Implementa la matriz de supervivencia para validar la robustez temporal.
    Asegura que una estrategia no solo funcione en el agregado, sino consistentemente.
    """

    def __init__(self, n_windows: int = 4, threshold_pf: float = 1.0):
        self.n_windows = n_windows
        self.threshold_pf = threshold_pf

    def compute_survival_matrix(self,
                                data: np.ndarray,
                                configurations_signals: np.ndarray) -> np.ndarray:
        """
        Calcula la robustez de múltiples configuraciones sobre ventanas temporales.

        Args:
            data: Buffer de retornos OOS [N x 2].
            configurations_signals: Matriz [Configuraciones x Tiempo].

        Returns:
            np.ndarray: Vector booleano [Configuraciones] donde True indica supervivencia.
        """
        log_returns = data[:, 1]
        n_samples = len(log_returns)
        window_size = n_samples // self.n_windows

        return _check_survival_jit(
            log_returns,
            configurations_signals,
            self.n_windows,
            window_size,
            self.threshold_pf
        )

    def check_single_survival(self, data: np.ndarray, signals: np.ndarray) -> bool:
        """
        Verifica la supervivencia de una única configuración.
        """
        log_returns = data[:, 1]
        n_samples = len(log_returns)
        window_size = n_samples // self.n_windows
        
        return _check_single_survival_jit(
            log_returns, 
            signals, 
            self.n_windows, 
            window_size, 
            self.threshold_pf
        )

@njit(cache=True)
def _check_single_survival_jit(log_returns, signals, n_windows, window_size, threshold):
    """
    Evaluación de supervivencia ultra-rápida (NJIT).
    """
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size
        
        w_returns = log_returns[start:end]
        w_signals = signals[start:end]
        
        # profit_factor atomico desde backtester
        from src.core.backtester import _compute_metrics_jit
        metrics = _compute_metrics_jit(w_returns, w_signals)
        if metrics[0] < threshold:
            return False
    return True

@njit(parallel=True, cache=True)
def _check_survival_jit(log_returns, signals_matrix, n_windows, window_size, threshold):
    """
    Filtro de supervivencia paralelo.
    Cualquier fallo en una ventana descarta la configuración inmediatamente.
    """
    n_configs = signals_matrix.shape[0]
    is_survivor = np.ones(n_configs, dtype=np.bool_)

    for i in prange(n_configs):
        signals = signals_matrix[i]

        for w in range(n_windows):
            start = w * window_size
            end = (w + 1) * window_size

            # Extraemos vistas de la ventana
            w_returns = log_returns[start:end]
            w_signals = signals[start:end]

            # Calculamos PF simplificado para velocidad
            metrics = _compute_metrics_jit(w_returns, w_signals)
            profit_factor = metrics[0]

            if profit_factor < threshold:
                is_survivor[i] = False
                break # Short-circuit: Falló una ventana, descartamos

    return is_survivor
