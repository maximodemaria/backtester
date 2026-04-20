"""
Motor de cálculo de backtesting vectorizado y optimizado con Numba.
"""
from src.core.metrics import profit_factor, total_return, sharpe_ratio

class BacktesterEngine:
    """
    Motor de backtesting de alto rendimiento.
    Calcula el rendimiento de estrategias mediante vectorización pura y compilación JIT.
    """

    @staticmethod
    def run(data: np.ndarray, signals: np.ndarray) -> dict:
        """
        Ejecuta el cálculo de métricas sobre un set de datos y señales.
        Incluye validación de Lookahead Bias.
        """
        log_returns = data[:, 1]
        
        # --- HARD CHECK: LOOKAHEAD BIAS ---
        if len(signals) > 10:
            correlation = np.corrcoef(signals, log_returns)[0, 1]
            assert abs(correlation) < 0.99, (
                f"POTENCIAL LOOKAHEAD BIAS DETECTADO: Correlación signal/return = {correlation:.4f}. "
                "La estrategia parece usar información del futuro."
            )

        metrics = _compute_metrics_jit(log_returns, signals)

        return {
            "profit_factor": metrics[0],
            "total_return": metrics[1],
            "sharpe_ratio": metrics[2]
        }

@njit(nopython=True, cache=True)
def _compute_metrics_jit(log_returns: np.ndarray, signals: np.ndarray) -> np.ndarray:
    """
    Motor matemático optimizado.
    Orquesta llamadas a módulos de métricas individuales.
    Firma mantenida: [pf, total_ret, sharpe]
    """
    n = len(log_returns)
    strategy_returns = np.zeros(n, dtype=np.float64)

    # Desplazamiento obligatorio para evitar Lookahead Bias
    for i in range(1, n):
        strategy_returns[i] = signals[i-1] * log_returns[i]

    # Delegación a módulos atómicos
    pf = profit_factor(strategy_returns)
    ret = total_return(strategy_returns)
    sh = sharpe_ratio(strategy_returns)

    return np.array([pf, ret, sh])
