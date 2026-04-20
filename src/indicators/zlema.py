import numpy as np
from numba import njit, prange
from .ema import ema

@njit(cache=True)
def zlema(data: np.ndarray, period: int) -> np.ndarray:
    """Zero Lag Exponential Moving Average (ZLEMA)."""
    lag = int((period - 1) / 2)
    n = len(data)
    de_lagged = np.full(n, np.nan, dtype=np.float64)
    for i in range(lag, n):
        de_lagged[i] = 2.0 * data[i] - data[i - lag]
    return ema(de_lagged, period)

