import numpy as np
from numba import njit

@njit(cache=True)
def profit_factor(strategy_returns: np.ndarray) -> float:
    """
    Calcula el Profit Factor: (Suma de retornos positivos) / |Suma de retornos negativos|.
    """
    pos_returns_sum = 0.0
    neg_returns_sum = 0.0
    
    for r in strategy_returns:
        if r > 0:
            pos_returns_sum += r
        elif r < 0:
            neg_returns_sum += abs(r)
            
    if neg_returns_sum == 0:
        return 999.0 if pos_returns_sum > 0 else 1.0
        
    return pos_returns_sum / neg_returns_sum
