"""
Estrategia de ejemplo basada en el cruce de medias móviles.
"""
import numpy as np
from numba import njit
from src.core.base_strategy import BaseStrategy

class MovingAverageStrategy(BaseStrategy):
    """
    Estrategia de Cruce de Medias Móviles optimizada.
    Demuestra la integración entre la interfaz abstracta y el motor Numba.
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
        Envoltorio para llamar a la función compilada JIT.
        data[:, 0] es Close.
        """
        fast_p = params['fast_period']
        slow_p = params['slow_period']
        return _compute_ma_signals(data[:, 0], int(fast_p), int(slow_p))

@njit(cache=True)
def _compute_ma_signals(close: np.ndarray, fast_p: int, slow_p: int) -> np.ndarray:
    """
    Motor de señales JIT.
    Calcula medias móviles simples y genera señales de compra (1) y venta (-1).
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)

    # Evitamos procesar antes de tener suficientes datos
    start_idx = max(fast_p, slow_p)

    for i in range(start_idx, n):
        # SMA Rápida
        sum_fast = 0.0
        for j in range(i - fast_p + 1, i + 1):
            sum_fast += close[j]
        sma_fast = sum_fast / fast_p

        # SMA Lenta
        sum_slow = 0.0
        for j in range(i - slow_p + 1, i + 1):
            sum_slow += close[j]
        sma_slow = sum_slow / slow_p

        if sma_fast > sma_slow:
            signals[i] = 1
        elif sma_fast < sma_slow:
            signals[i] = -1

    return signals
