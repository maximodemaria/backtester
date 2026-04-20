import numpy as np
from numba import njit, prange

@njit(cache=True)
def parabolic_sar(high: np.ndarray, low: np.ndarray, step: float = 0.02, max_step: float = 0.2) -> np.ndarray:
    """Parabolic Stop and Reverse (SAR)."""
    n = len(high)
    sar = np.zeros(n, dtype=np.float64)
    uptrend = True
    af = step
    ep = high[0]
    sar[0] = low[0]
    
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if uptrend:
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                af = step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                af = step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
    return sar
