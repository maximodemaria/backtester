import numpy as np
from numba import njit, prange

@njit(cache=True)
def k_efficiency_ratio(data: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Efficiency Ratio (ER)."""
    n = len(data)
    er = np.full(n, np.nan, dtype=np.float64)
    if n < period: return er
    
    for i in range(period, n):
        change = np.abs(data[i] - data[i - period])
        diffs = data[i-period+1:i+1] - data[i-period:i]
        vol = np.sum(np.abs(diffs))
        er[i] = change / vol if vol != 0 else 0.0
    return er

# =============================================================================
# BLOQUE 2: OSCILADORES (MOMENTUM)

