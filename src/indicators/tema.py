import numpy as np
from numba import njit, prange
from .ema import ema

@njit(cache=True)
def tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average (TEMA)."""
    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * (e1 - e2) + e3

