import numpy as np
from numba import njit, prange
from .rma import rma
from .atr import atr

@njit(cache=True)
def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index (ADX)."""
    n = len(close)
    dm_pos = np.zeros(n, dtype=np.float64)
    dm_neg = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0: dm_pos[i] = up_move
        if down_move > up_move and down_move > 0: dm_neg[i] = down_move
        
    tr_smooth = atr(high, low, close, period)
    di_pos = 100.0 * rma(dm_pos, period) / tr_smooth
    di_neg = 100.0 * rma(dm_neg, period) / tr_smooth
    
    dx = 100.0 * np.abs(di_pos - di_neg) / (di_pos + di_neg)
    return rma(dx, period)

