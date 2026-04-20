import numpy as np
from numba import njit

@njit(cache=True)
def total_return(strategy_returns: np.ndarray) -> float:
    """
    Calcula el retorno total acumulado a partir de retornos logarítmicos.
    Formula: exp(sum(r)) - 1
    """
    total_log_return = 0.0
    for r in strategy_returns:
        total_log_return += r
        
    return np.exp(total_log_return) - 1
