import numpy as np
from numba import njit, prange
from .ema import ema

@njit(cache=True)
def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Moving Average Convergence Divergence (MACD)."""
    macd_line = ema(data, fast) - ema(data, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

