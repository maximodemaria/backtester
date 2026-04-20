import numpy as np
from numba import njit, prange
from .wma import wma

@njit(cache=True)
def hma(data: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average (HMA)."""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    
    diff = 2.0 * wma_half - wma_full
    # HMA es WMA de la diferencia sobre la raíz del periodo
    return wma(diff, sqrt_period)

