import numpy as np
from numba import njit, prange

@njit(cache=True)
def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic Oscillator (%K, %D)."""
    n = len(close)
    k_line = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(k_period - 1, n):
        h_max = np.max(high[i - k_period + 1 : i + 1])
        l_min = np.min(low[i - k_period + 1 : i + 1])
        if h_max != l_min:
            k_line[i] = 100.0 * (close[i] - l_min) / (h_max - l_min)
        else:
            k_line[i] = 50.0
            
    d_line = sma(k_line, d_period)
    return k_line, d_line

