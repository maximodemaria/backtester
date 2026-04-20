import numpy as np
from numba import njit, prange

@njit(cache=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (SMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        current_sum = current_sum - data[i - period] + data[i]
        result[i] = current_sum / period
    return result

