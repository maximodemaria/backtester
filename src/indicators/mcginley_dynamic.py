import numpy as np
from numba import njit, prange

@njit(cache=True)
def mcginley_dynamic(data: np.ndarray, period: int) -> np.ndarray:
    """McGinley Dynamic Moving Average."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        # Formula: MD = MD[-1] + (Price - MD[-1]) / (period * (Price / MD[-1])^4)
        denom = period * (data[i] / result[i - 1]) ** 4
        result[i] = result[i - 1] + (data[i] - result[i - 1]) / denom
    return result

