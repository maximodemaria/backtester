import numpy as np
from numba import njit, prange

@njit(cache=True)
def rma(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling Moving Average (RMA) - Usada en RSI."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    alpha = 1.0 / period
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

