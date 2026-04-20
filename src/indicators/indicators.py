"""
Librería de Indicadores Técnicos - Suite Cuantitativa HFT
Optimizado con Numba (@njit) para ejecución de alto rendimiento.
Categorías: Medias Móviles, Osciladores, Volatilidad y Tendencia.
"""
import numpy as np
from numba import njit, prange

# =============================================================================
# BLOQUE 1: MEDIAS MÓVILES (TREND FOLLOWING)
# =============================================================================

@njit(cache=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average (SMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        current_sum = current_sum - data[i - period] + data[i]
        result[i] = current_sum / period
    return result

@njit(cache=True)
def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average (EMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    alpha = 2.0 / (period + 1)
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * alpha + result[i - 1]
    return result

@njit(cache=True)
def wma(data: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average (WMA) - Lineal."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    weight_sum = period * (period + 1) / 2
    for i in range(period - 1, n):
        current_sum = 0.0
        for j in range(period):
            current_sum += data[i - j] * (period - j)
        result[i] = current_sum / weight_sum
    return result

@njit(cache=True)
def hma(data: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average (HMA)."""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    
    diff = 2.0 * wma_half - wma_full
    # HMA es WMA de la diferencia sobre la raíz del periodo
    return wma(diff, sqrt_period)

@njit(cache=True)
def dema(data: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average (DEMA)."""
    e1 = ema(data, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2

@njit(cache=True)
def tema(data: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average (TEMA)."""
    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * (e1 - e2) + e3

@njit(cache=True)
def tma(data: np.ndarray, period: int) -> np.ndarray:
    """Triangular Moving Average (TMA)."""
    half = int((period + 1) / 2)
    return sma(sma(data, half), half)

@njit(cache=True)
def rma(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling Moving Average (RMA) - Usada en RSI."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    alpha = 1.0 / period
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    result[period - 1] = current_sum / period
    
    for i in range(period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

@njit(cache=True)
def zlema(data: np.ndarray, period: int) -> np.ndarray:
    """Zero Lag Exponential Moving Average (ZLEMA)."""
    lag = int((period - 1) / 2)
    n = len(data)
    de_lagged = np.full(n, np.nan, dtype=np.float64)
    for i in range(lag, n):
        de_lagged[i] = 2.0 * data[i] - data[i - lag]
    return ema(de_lagged, period)

@njit(cache=True)
def kama(data: np.ndarray, period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman Adaptive Moving Average (KAMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period: return result
    
    # Efficiency Ratio (ER)
    change = np.abs(data[period:] - data[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    
    abs_diff = np.abs(np.diff(data))
    for i in range(n - period):
        volatility[i] = np.sum(abs_diff[i:i+period])
        
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

@njit(cache=True)
def alma(data: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """Arnaud Legoux Moving Average (ALMA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    m = offset * (period - 1)
    s = period / sigma
    weights = np.exp(-((np.arange(period) - m) ** 2) / (2 * s * s))
    weights /= np.sum(weights)
    
    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = np.sum(window * weights[::-1])
    return result

@njit(cache=True)
def vwma(data: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    """Volume Weighted Moving Average (VWMA)."""
    pv = data * volume
    return sma(pv, period) / sma(volume, period)

@njit(cache=True)
def mcginley_dynamic(data: np.ndarray, period: int) -> np.ndarray:
    """McGinley Dynamic Moving Average."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        # Formula: MD = MD[-1] + (Price - MD[-1]) / (period * (Price / MD[-1])^4)
        denom = period * (data[i] / result[i - 1]) ** 4
        result[i] = result[i - 1] + (data[i] - result[i - 1]) / denom
    return result

@njit(cache=True)
def vidya(data: np.ndarray, period: int, select_period: int = 9) -> np.ndarray:
    """Chande's Variable Index Dynamic Average (VIDYA)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    
    # Chande Momentum Oscillator (CMO) como factor de adaptatividad
    cmo = rsi(data, select_period) # Usamos RSI escalado como proxy de CMO
    cmo_abs = np.abs(cmo - 50.0) / 50.0
    alpha = 2.0 / (period + 1)
    
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        if np.isnan(cmo_abs[i]): continue
        k = alpha * cmo_abs[i]
        result[i] = k * data[i] + (1 - k) * result[i - 1]
    return result

@njit(cache=True)
def k_efficiency_ratio(data: np.ndarray, period: int) -> np.ndarray:
    """Kaufman Efficiency Ratio (ER)."""
    n = len(data)
    er = np.full(n, np.nan, dtype=np.float64)
    if n < period: return er
    
    for i in range(period, n):
        path = abs(data[i] - data[i - period])
        vol = np.sum(np.abs(np.diff(data[i - period:i + 1])))
        er[i] = path / vol if vol != 0 else 0.0
    return er

# =============================================================================
# BLOQUE 2: OSCILADORES (MOMENTUM)
# =============================================================================

@njit(cache=True)
def rsi(data: np.ndarray, period: int) -> np.ndarray:
    """Relative Strength Index (RSI)."""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period: return result
    
    deltas = np.diff(data)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
    
    # Smoothing inicial (SMA) y luego RMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0: result[period] = 100.0 if avg_gain != 0 else 50.0
    else: result[period] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss
        if avg_loss == 0: result[i] = 100.0 if avg_gain != 0 else 50.0
        else: result[i] = 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
    return result

@njit(cache=True)
def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Moving Average Convergence Divergence (MACD)."""
    macd_line = ema(data, fast) - ema(data, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

@njit(cache=True)
def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """Stochastic Oscillator (%K, %D)."""
    n = len(close)
    k_line = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(k_period - 1, n):
        h_max = np.max(high[i - k_period + 1 : i + 1])
        l_min = np.min(low[i - k_period + 1 : i + 1])
        if h_max != l_min:
            k_line[i] = 100.0 * (close[i] - l_min) / (h_max - l_min)
        else:
            k_line[i] = 50.0
            
    d_line = sma(k_line, d_period)
    return k_line, d_line

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

@njit(cache=True)
def bollinger_bands(data: np.ndarray, period: int, std_dev: float = 2.0) -> tuple:
    """Bollinger Bands (BBANDS)."""
    middle = sma(data, period)
    n = len(data)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        sigma = np.std(data[i - period + 1 : i + 1])
        upper[i] = middle[i] + (std_dev * sigma)
        lower[i] = middle[i] - (std_dev * sigma)
    return upper, middle, lower

@njit(cache=True)
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range (ATR)."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    return rma(tr, period)

@njit(cache=True)
def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index (ADX)."""
    n = len(close)
    dm_pos = np.zeros(n, dtype=np.float64)
    dm_neg = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0: dm_pos[i] = up_move
        if down_move > up_move and down_move > 0: dm_neg[i] = down_move
        
    tr_smooth = atr(high, low, close, period)
    di_pos = 100.0 * rma(dm_pos, period) / tr_smooth
    di_neg = 100.0 * rma(dm_neg, period) / tr_smooth
    
    dx = 100.0 * np.abs(di_pos - di_neg) / (di_pos + di_neg)
    return rma(dx, period)

@njit(cache=True)
def donchian_channels(high: np.ndarray, low: np.ndarray, period: int) -> tuple:
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
    return upper, lower

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
