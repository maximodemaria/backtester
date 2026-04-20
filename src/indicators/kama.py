import numpy as np
from numba import njit, prange

@njit(cache=True)
def kama(data: np.ndarray, period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman Adaptive Moving Average (KAMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period: return result
    
    # Efficiency Ratio (ER)
    change = np.abs(data[period:] - data[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    # Eficiencia de Diferencia Absoluta manual para evitar errores de contigüidad
    abs_diff = np.abs(data[1:] - data[:-1])
    for i in range(n - period):
        volatility[i] = np.sum(abs_diff[i:i+period-1])
        
    er = np.zeros(n, dtype=np.float64)
    er[period:] = change / volatility
    er[er == 0] = 0.0001 # Prevenir división por cero
    
    sc_fast = 2.0 / (fast + 1)
    sc_slow = 2.0 / (slow + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        result[i] = result[i - 1] + sc[i] * (data[i] - result[i - 1])
    return result

