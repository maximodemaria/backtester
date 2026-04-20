import numpy as np
import pandas as pd
from numba import njit, prange
import sqlite3
import random
import time
import os

# -----------------------------------------------------------------------------
# 1. INDICADORES (NUMBA JIT)
# -----------------------------------------------------------------------------

@njit(cache=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
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
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if period <= 1: return data.copy()
    weight_sum = period * (period + 1) / 2
    for i in range(period - 1, n):
        current_sum = 0.0
        for j in range(period):
            current_sum += data[i - j] * (period - j)
        result[i] = current_sum / weight_sum
    return result

@njit(cache=True)
def hma(data: np.ndarray, period: int) -> np.ndarray:
    if period <= 1: return data.copy()
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma_half = wma(data, half_period)
    wma_full = wma(data, period)
    diff = 2.0 * wma_half - wma_full
    return wma(diff, sqrt_period)

@njit(cache=True)
def dema(data: np.ndarray, period: int) -> np.ndarray:
    e1 = ema(data, period)
    e2 = ema(e1, period)
    return 2.0 * e1 - e2

@njit(cache=True)
def tema(data: np.ndarray, period: int) -> np.ndarray:
    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * (e1 - e2) + e3

@njit(cache=True)
def tma(data: np.ndarray, period: int) -> np.ndarray:
    half = int((period + 1) / 2)
    return sma(sma(data, half), half)

@njit(cache=True)
def rma(data: np.ndarray, period: int) -> np.ndarray:
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
def rsi(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n-1 <= period: return result
    deltas = data[1:] - data[:-1]
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)
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
def zlema(data: np.ndarray, period: int) -> np.ndarray:
    lag = int((period - 1) / 2)
    n = len(data)
    de_lagged = np.full(n, np.nan, dtype=np.float64)
    for i in range(lag, n):
        de_lagged[i] = 2.0 * data[i] - data[i - lag]
    return ema(de_lagged, period)

@njit(cache=True)
def kama(data: np.ndarray, period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period: return result
    change = np.abs(data[period:] - data[:-period])
    volatility = np.zeros(n - period, dtype=np.float64)
    abs_diff = np.abs(data[1:] - data[:-1])
    for i in range(n - period):
        volatility[i] = np.sum(abs_diff[i:i+period-1])
    er = np.zeros(n, dtype=np.float64)
    er[period:] = change / volatility
    er[er == 0] = 0.0001
    sc_fast = 2.0 / (fast + 1)
    sc_slow = 2.0 / (slow + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        result[i] = result[i - 1] + sc[i] * (data[i] - result[i - 1])
    return result

@njit(cache=True)
def alma(data: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
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
    pv = data * volume
    return sma(pv, period) / sma(volume, period)

@njit(cache=True)
def mcginley_dynamic(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        denom = period * (data[i] / result[i - 1]) ** 4
        result[i] = result[i - 1] + (data[i] - result[i - 1]) / denom
    return result

@njit(cache=True)
def vidya(data: np.ndarray, period: int, select_period: int = 9) -> np.ndarray:
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period: return result
    cmo = rsi(data, select_period)
    cmo_abs = np.abs(cmo - 50.0) / 50.0
    alpha = 2.0 / (period + 1)
    result[period - 1] = data[period - 1]
    for i in range(period, n):
        if np.isnan(cmo_abs[i]): continue
        k = alpha * cmo_abs[i]
        result[i] = k * data[i] + (1 - k) * result[i - 1]
    return result

# -----------------------------------------------------------------------------
# 2. KERNELES DE ESTRATEGIA Y BACKTEST
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _compute_quad_ma_signals_inplace(fe_ma, se_ma, fx_ma, sx_ma, out_signals):
    n = len(fe_ma)
    current_pos = 0 
    for i in range(1, n):
        if current_pos == 1:
            if fx_ma[i] < sx_ma[i]: current_pos = 0
        elif current_pos == -1:
            if fx_ma[i] > sx_ma[i]: current_pos = 0
        
        if current_pos == 0:
            if fe_ma[i] > se_ma[i]: current_pos = 1
            elif fe_ma[i] < se_ma[i]: current_pos = -1
        out_signals[i] = current_pos

@njit(cache=True, fastmath=True)
def _compute_metrics_jit(log_returns, signals, comm_factor):
    n = len(log_returns)
    wins = 0.0
    losses = 0.0
    total_ret = 0.0
    
    for i in range(1, n):
        ret = signals[i-1] * log_returns[i]
        prev_pos = signals[i-2] if i > 1 else 0.0
        if signals[i-1] != prev_pos:
            ret -= abs(signals[i-1] - prev_pos) * comm_factor
        
        total_ret += ret
        if ret > 0: wins += ret
        elif ret < 0: losses += abs(ret)
    
    pf = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
    return pf, total_ret

# -----------------------------------------------------------------------------
# 3. Lógica Principal de Optimización
# -----------------------------------------------------------------------------

def run_optimization():
    print("Cargando datos de GGAL 1m...", flush=True)
    df = pd.read_csv("data/GGAL_1m.csv")
    close = df['close'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
    log_returns = np.zeros_like(close)
    log_returns[1:] = np.log(close[1:] / close[:-1])
    
    ma_types = [
        'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'tma',
        'rma', 'zlema', 'kama', 'alma', 'vwma', 'mcginley', 'vidya'
    ]
    
    periods = [1] + list(range(5, 205, 5))
    legal_pairs = []
    for fe in periods:
        for se in periods:
            if fe < se:
                legal_pairs.append((fe, se))
    
    print(f"Pre-calculando {len(ma_types) * len(periods)} indicadores...", flush=True)
    # Cache: matrix [ma_type_idx, period_idx, price_data]
    # Usaremos un diccionario para facilitar, pero para Numba pasaremos una matriz si es necesario.
    # Como el script es monolítico y queremos velocidad, pre-calcularemos todo.
    
    indicator_cache = {}
    for mt_idx, mt in enumerate(ma_types):
        for p in periods:
            if mt == 'sma': res = sma(close, p)
            elif mt == 'ema': res = ema(close, p)
            elif mt == 'wma': res = wma(close, p)
            elif mt == 'hma': res = hma(close, p)
            elif mt == 'dema': res = dema(close, p)
            elif mt == 'tema': res = tema(close, p)
            elif mt == 'tma': res = tma(close, p)
            elif mt == 'rma': res = rma(close, p)
            elif mt == 'zlema': res = zlema(close, p)
            elif mt == 'kama': res = kama(close, p)
            elif mt == 'alma': res = alma(close, p)
            elif mt == 'vwma': res = vwma(close, volume, p)
            elif mt == 'mcginley': res = mcginley_dynamic(close, p)
            elif mt == 'vidya': res = vidya(close, p)
            indicator_cache[(mt, p)] = res

    # 100,000 combinaciones aleatorias
    # (ma_type, fe, se, fx, sx)
    print("Generando combinaciones aleatorias...", flush=True)
    total_legal_pairs = len(legal_pairs)
    
    random_configs = []
    for _ in range(100000):
        mt = random.choice(ma_types)
        entry_pair = random.choice(legal_pairs)
        exit_pair = random.choice(legal_pairs)
        random_configs.append((mt, entry_pair[0], entry_pair[1], exit_pair[0], exit_pair[1]))

    # Resultados
    results = []
    # I'll use 0.0 based on the last user change to templates/ggal_hft.yaml
    comm_factor = 0.0

    print("Iniciando optimización de 100,000 combinaciones...", flush=True)
    start_time = time.time()
    
    out_signals = np.zeros(len(close), dtype=np.int8)
    
    for i, cfg in enumerate(random_configs):
        mt, fe, se, fx, sx = cfg
        
        fe_ma = indicator_cache[(mt, fe)]
        se_ma = indicator_cache[(mt, se)]
        fx_ma = indicator_cache[(mt, fx)]
        sx_ma = indicator_cache[(mt, sx)]
        
        # Limpiar señales
        out_signals[:] = 0
        
        # Generar señales
        _compute_quad_ma_signals_inplace(fe_ma, se_ma, fx_ma, sx_ma, out_signals)
        
        # Backtest
        pf, ret = _compute_metrics_jit(log_returns, out_signals, comm_factor)
        
        if ret > 0:
            results.append({
                'ma_type': mt,
                'fe': fe,
                'se': se,
                'fx': fx,
                'sx': sx,
                'pf': pf,
                'return': ret
            })
            
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Procesadas {i+1} combinaciones... {elapsed:.2f}s", flush=True)

    # Guardar en DB
    print(f"Guardando {len(results)} resultados rentables en SQLite...", flush=True)
    conn = sqlite3.connect("ggal_quadma_results.db")
    df_res = pd.DataFrame(results)
    df_res.to_sql("results", conn, if_exists="replace", index=False)
    conn.close()
    
    total_time = time.time() - start_time
    print(f"Optimización completada en {total_time:.2f}s", flush=True)
    print("Base de datos 'ggal_quadma_results.db' creada.", flush=True)

if __name__ == "__main__":
    run_optimization()
