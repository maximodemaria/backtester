import numpy as np
from numba import njit

@njit(cache=True)
def sharpe_ratio(strategy_returns: np.ndarray, annualization_factor: float = 252.0) -> float:
    """
    Calcula el Sharpe Ratio con estabilidad numérica.
    """
    epsilon = 1e-9
    mean_ret = np.mean(strategy_returns)
    std_dev = np.std(strategy_returns)
    
    if std_dev == 0:
        return 0.0
        
    return (mean_ret / (std_dev + epsilon)) * np.sqrt(annualization_factor)
