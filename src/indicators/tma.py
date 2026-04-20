import numpy as np
from numba import njit, prange
from .sma import sma

@njit(cache=True)
def tma(data: np.ndarray, period: int) -> np.ndarray:
    """Triangular Moving Average (TMA)."""
    half = int((period + 1) / 2)
    return sma(sma(data, half), half)

