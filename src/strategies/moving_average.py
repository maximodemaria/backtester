import numpy as np
from numba import njit
from src.core.base_strategy import BaseStrategy
from src.indicators import sma

class MovingAverageStrategy(BaseStrategy):
    """
    Estrategia de Cruce de Medias Móviles optimizada.
    Utiliza la librería centralizada de indicadores para los cálculos.
    """

    def __init__(self):
        super().__init__(name="MovingAverageCross")

    @property
    def param_grid(self) -> list:
        """Genera permutaciones para la optimización."""
        grid = []
        for f in np.arange(5, 50, 10):
            for s in np.arange(20, 100, 20):
                grid.append({'fast_period': f, 'slow_period': s})
        return grid

    def generate_signal(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Genera señales de trading basadas en el cruce de SMAs.
        data[:, 0] es Close.
        """
        fast_p = int(params['fast_period'])
        slow_p = int(params['slow_period'])
        
        close = data[:, 0]
        
        # Obtenemos las series de indicadores desde la librería centralizada
        sma_fast = sma(close, fast_p)
        sma_slow = sma(close, slow_p)
        
        return _compute_cross_signals(sma_fast, sma_slow)

@njit(cache=True)
def _compute_cross_signals(fast_ma: np.ndarray, slow_ma: np.ndarray) -> np.ndarray:
    """
    Lógica de decisión pura. 
    Solo se encarga de comparar las series ya calculadas.
    """
    n = len(fast_ma)
    signals = np.zeros(n, dtype=np.int8)

    for i in range(n):
        # Las comparaciones con NaN (periodos incompletos) devuelven False.
        # Esto evita señales espurias al inicio del dataset.
        if fast_ma[i] > slow_ma[i]:
            signals[i] = 1
        elif fast_ma[i] < slow_ma[i]:
            signals[i] = -1

    return signals
