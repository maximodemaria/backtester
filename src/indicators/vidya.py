import numpy as np
from numba import njit, prange
from .rsi import rsi

@njit(cache=True)
def vidya(data: np.ndarray, period: int, select_period: int = 9) -> np.ndarray:
    """Chande's Variable Index Dynamic Average (VIDYA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    # Chande Momentum Oscillator (CMO) como factor de adaptatividad
    cmo = rsi(data, select_period) # Usamos RSI escalado como proxy de CMO
    cmo_abs = np.abs(cmo - 50.0) / 50.0
    alpha = 2.0 / (period + 1)
    
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        if np.isnan(cmo_abs[i]): continue
        k = alpha * cmo_abs[i]
        result[i] = k * data[i] + (1 - k) * result[i - 1]
    return result

