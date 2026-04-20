"""
Librería centralizada de indicadores técnicos optimizada con Numba.
Todas las funciones son puras y operan sobre arrays de numpy.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (SMA).
    Retorna np.nan para índices menores al periodo.
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return result

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
    """
    Exponential Moving Average (EMA).
    Utiliza el SMA inicial como primer valor de EMA.
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n < period:
        return result
    
    alpha = 2.0 / (period + 1)
    
    # El primer valor es un SMA
    current_sum = 0.0
    for i in range(period):
        current_sum += data[i]
    
    result[period - 1] = current_sum / period
    
    # Cálculo recursivo
    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * alpha + result[i - 1]
        
    return result

@njit(cache=True)
def rsi(data: np.ndarray, period: int) -> np.ndarray:
    """
    Relative Strength Index (RSI).
    Implementación estándar de Wilder.
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n <= period:
        return result
    
    deltas = np.diff(data)
    gains = np.zeros(n - 1, dtype=np.float64)
    losses = np.zeros(n - 1, dtype=np.float64)
    
    for i in range(n - 1):
        if deltas[i] > 0:
            gains[i] = deltas[i]
        else:
            losses[i] = abs(deltas[i])
            
    # Promedio inicial
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
        
    # Smoothed Moving Average (Wilder)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return result

@njit(cache=True)
def bollinger_bands(data: np.ndarray, period: int, std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands (BBANDS).
    Retorna una tupla (upper_band, middle_band, lower_band).
    """
    middle_band = sma(data, period)
    upper_band = np.full_like(middle_band, np.nan)
    lower_band = np.full_like(middle_band, np.nan)
    
    for i in range(period - 1, len(data)):
        window = data[i - period + 1 : i + 1]
        sigma = np.std(window)
        upper_band[i] = middle_band[i] + (std_dev * sigma)
        lower_band[i] = middle_band[i] - (std_dev * sigma)
        
    return upper_band, middle_band, lower_band
