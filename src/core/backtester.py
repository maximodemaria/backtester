import numpy as np
from numba import njit
from src.core.metrics import profit_factor, total_return, sharpe_ratio

class BacktesterEngine:
    """
    Motor de backtesting de alto rendimiento.
    Calcula el rendimiento de estrategias mediante vectorización pura y compilación JIT.
    """

    @staticmethod
    def run(data: np.ndarray, signals: np.ndarray, commission_bps: float = 0.0) -> dict:
        """
        Ejecuta el cálculo de métricas sobre un set de datos y señales.
        """
        log_returns = data[:, 1]
        
        # --- HARD CHECK: LOOKAHEAD BIAS ---
        if len(signals) > 10:
            with np.errstate(invalid='ignore'):
                correlation_matrix = np.corrcoef(signals, log_returns)
                correlation = correlation_matrix[0, 1]
            
            # Si la correlación es NaN, la señal es constante o inválida.
            # No lanzamos error, simplemente retornamos métricas vacías.
            if np.isnan(correlation):
                return {"profit_factor": 0.0, "total_return": 0.0, "sharpe_ratio": 0.0}

            if abs(correlation) > 0.99:
                return {"profit_factor": 0.0, "total_return": 0.0, "sharpe_ratio": 0.0}

        # Pre-calculamos el factor de comisión (bps a decimal)
        commission_factor = commission_bps / 10000.0
        
        metrics = _compute_metrics_jit(log_returns, signals, commission_factor)

        return {
            "profit_factor": metrics[0],
            "total_return": metrics[1],
            "sharpe_ratio": metrics[2]
        }

@njit(nopython=True, cache=True)
def _compute_metrics_jit(log_returns: np.ndarray, signals: np.ndarray, comm_factor: float = 0.0) -> np.ndarray:
    """
    Motor matemático optimizado.
    Deduce comisiones por cada cambio de posición.
    """
    n = len(log_returns)
    strategy_returns = np.zeros(n, dtype=np.float64)

    # El retorno en i depende de la posición en i-1
    for i in range(1, n):
        # 1. Retorno Bruto
        strategy_returns[i] = signals[i-1] * log_returns[i]
        
        # 2. Aplicar Comisiones por trade (cambio de posición)
        # Comparamos posición actual (i-1) con la anterior (i-2)
        prev_pos = signals[i-2] if i > 1 else 0.0
        if signals[i-1] != prev_pos:
            # Costo simplificado: comm_factor * abs(cambio_posicion)
            # En HFT con señales {-1, 0, 1}, el máximo cambio es 2.
            cost = abs(signals[i-1] - prev_pos) * comm_factor
            strategy_returns[i] -= cost

    # Delegación a módulos atómicos
    pf = profit_factor(strategy_returns)
    ret = total_return(strategy_returns)
    sh = sharpe_ratio(strategy_returns)

    return np.array([pf, ret, sh])
