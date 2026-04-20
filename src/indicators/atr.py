import numpy as np
from numba import njit, prange
from .rma import rma

@njit(cache=True)
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (ATR)."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    return rma(tr, period)

