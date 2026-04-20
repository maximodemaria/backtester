import numpy as np
from numba import njit, prange

@njit(cache=True)
def roc(data: np.ndarray, period: int) -> np.ndarray:
    """Rate of Change (ROC)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    result[period:] = 100.0 * (data[period:] - data[:-period]) / data[:-period]
    return result

# =============================================================================
# BLOQUE 3: VOLATILIDAD Y TENDENCIA
# =============================================================================

