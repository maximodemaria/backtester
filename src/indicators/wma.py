import numpy as np
from numba import njit, prange

@njit(cache=True)
def wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average (WMA) - Lineal."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    weight_sum = period * (period + 1) / 2
    for i in range(period - 1, n):
        current_sum = 0.0
        for j in range(period):
            current_sum += data[i - j] * (period - j)
        result[i] = current_sum / weight_sum
    return result

