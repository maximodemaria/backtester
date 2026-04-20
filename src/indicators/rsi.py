import numpy as np
from numba import njit, prange

@njit(cache=True)
def rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (RSI)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period: return result
    
    deltas = data[1:] - data[:-1]
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    
    # Smoothing inicial (SMA) y luego RMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0: result[period] = 100.0 if avg_gain != 0 else 50.0
    else: result[period] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss
        if avg_loss == 0: result[i] = 100.0 if avg_gain != 0 else 50.0
        else: result[i] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    return result

