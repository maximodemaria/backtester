import numpy as np
from numba import njit, prange

@njit(cache=True)
def alma(data: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """Arnaud Legoux Moving Average (ALMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    m = offset * (period - 1)
    s = period / sigma
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    weights /= np.sum(weights)
    
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.sum(window * weights[::-1])
    return result

