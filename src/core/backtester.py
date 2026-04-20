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
    Motor matemático Trade-based.
    Calcula el PF basado en resultados de operaciones cerradas.
    """
    n = len(log_returns)
    gross_profits = 0.0
    gross_losses = 0.0
    total_ret = 0.0
    
    current_trade_ret = 0.0
    prev_pos = 0.0
    
    for i in range(1, n):
        curr_pos = signals[i-1]
        
        # 1. Acumular retorno de la barra si estamos en posición
        if curr_pos != 0:
            current_trade_ret += curr_pos * log_returns[i]
        
        # 2. Detectar salto de posición (Entrada, Cierre o Reversal)
        if curr_pos != prev_pos:
            # Si veníamos de una posición, cerramos el trade anterior
            if prev_pos != 0:
                # El costo de cierre es comm_factor * abs(prev_pos)
                # Pero el usuario quiere cobrar comisión al entrar también.
                # Lógica: Cualquier cambio de posicion paga abs(curr - prev) * comm.
                cost = abs(curr_pos - prev_pos) * comm_factor
                
                # Para el Trade-based PF, usualmente el trade se evalúa al cerrar.
                # Sumamos el acumulado y restamos el costo total del trade (entrada + salida).
                # Pero para mantenerlo simple y exacto segun el pedido:
                # El "Trade Result" neto es el acumulado menos los costos incurridos.
                
                # IMPORTANTE: Para PF por trade, necesitamos saber cuándo empezó el trade.
                # Simplificación robusta: Acumulamos retornos y solo cuando volvemos a 0 
                # o cambiamos de signo, cerramos la métrica del trade.
                
                # Si es un cierre (curr_pos == 0) o un reversal (signos opuestos)
                if curr_pos == 0 or (curr_pos * prev_pos < 0):
                    net_trade_ret = current_trade_ret - cost
                    if net_trade_ret > 0:
                        gross_profits += net_trade_ret
                    else:
                        gross_losses += abs(net_trade_ret)
                    
                    # Reiniciamos acumulador
                    current_trade_ret = 0.0
                else:
                    # Es un aumento de posición (no debería ocurrir en este motor HFT de -1,0,1, 
                    # pero lo manejamos restando el costo al acumulado actual)
                    current_trade_ret -= cost
            else:
                # Es una ENTRADA desde 0
                cost = abs(curr_pos - prev_pos) * comm_factor
                current_trade_ret -= cost
                
            total_ret -= abs(curr_pos - prev_pos) * comm_factor

        total_ret += curr_pos * log_returns[i]
        prev_pos = curr_pos

    # Forzar cierre del último trade si quedó abierto
    if prev_pos != 0:
        cost = abs(prev_pos) * comm_factor # Costo de salida final
        net_trade_ret = current_trade_ret - cost
        if net_trade_ret > 0:
            gross_profits += net_trade_ret
        else:
            gross_losses += abs(net_trade_ret)
        total_ret -= cost

    pf = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 1.0)
    
    # Cálculo de Sharpe simplificado (basado en barra para estabilidad)
    # El Sharpe trade-based es menos común en optimizadores rápidos.
    # Usamos el total_ret y una aproximación o dejamos el de barras para riesgo.
    # El usuario solo pidió PF trade-based.
    
    # Recalculamos Sharpe de barras para mantener consistencia de riesgo
    sh = total_ret / (np.std(log_returns * signals) * np.sqrt(252 * 60)) if np.std(log_returns * signals) > 0 else 0.0

    return np.array([pf, total_ret, sh])
