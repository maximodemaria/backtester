import numpy as np
from numba import njit, prange

@njit(cache=True)
def donchian_channels(high: np.ndarray, low: np.ndarray, period: int) -> tuple:
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
    return upper, lower

