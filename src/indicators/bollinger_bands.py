import numpy as np
from numba import njit, prange
from .sma import sma

@njit(cache=True)
def bollinger_bands(data: np.ndarray, period: int, std_dev: float = 2.0) -> tuple:
    """Bollinger Bands (BBANDS)."""
    middle = sma(data, period)
    n = len(data)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        sigma = np.std(data[i - period + 1 : i + 1])
        upper[i] = middle[i] + (std_dev * sigma)
        lower[i] = middle[i] - (std_dev * sigma)
    return upper, middle, lower

