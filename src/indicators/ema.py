import numpy as np
from numba import njit, prange

@njit(cache=True)
def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (EMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    alpha = 2.0 / (period + 1)
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * alpha + result[i - 1]
    return result

