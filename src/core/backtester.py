"""
Motor de cálculo de backtesting vectorizado y optimizado con Numba.
"""
import numpy as np
from numba import njit

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
        # Verificamos si hay una correlación sospechosa (perfecta) entre la señal
        # y el retorno del mismo periodo antes del desplazamiento.
        # Si corr(signals, log_returns) == 1.0 o -1.0, es casi seguro que 
        # la señal usa información del precio de cierre actual para posicionarse.
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
    Calcula: [Profit Factor, Total Return, Sharpe Ratio]
    """
    # Desplazamos las señales una barra para evitar look-ahead bias
    # El retorno de la estrategia en t es Posicion en t-1 * Retorno en t

    n = len(log_returns)
    strategy_returns = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        strategy_returns[i] = signals[i-1] * log_returns[i]

    # Cálculo de métricas
    pos_returns_sum = 0.0
    neg_returns_sum = 0.0
    total_log_return = 0.0

    for i in range(n):
        total_log_return += strategy_returns[i]
        if strategy_returns[i] > 0:
            pos_returns_sum += strategy_returns[i]
        elif strategy_returns[i] < 0:
            neg_returns_sum += abs(strategy_returns[i])

    # Profit Factor
    profit_factor = pos_returns_sum / neg_returns_sum if neg_returns_sum != 0 else 1.0

    # Sharpe Ratio simplificado
    std_dev = np.std(strategy_returns)
    mean_ret = np.mean(strategy_returns)
    # Asumiendo diario para demo (anualización)
    sharpe = (mean_ret / std_dev) * np.sqrt(252) if std_dev != 0 else 0.0

    return np.array([profit_factor, np.exp(total_log_return) - 1, sharpe])
