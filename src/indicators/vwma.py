import numpy as np
from numba import njit, prange
from .sma import sma

@njit(cache=True)
def vwma(data: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average (VWMA)."""
    pv = data * volume
    return sma(pv, period) / sma(volume, period)

