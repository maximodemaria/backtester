import numpy as np
from numba import njit, prange
from .ema import ema

@njit(cache=True)
def dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average (DEMA)."""
    e1 = ema(data, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2

