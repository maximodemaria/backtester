import numpy as np
from src.indicators import (
    sma, ema, wma, hma, dema, tema, tma, rma, zlema, kama, alma, vwma,
    mcginley_dynamic, vidya
)

def get_indicator(data: np.ndarray, ma_type: str, period: int) -> np.ndarray:
    """
    Despachador central de indicadores para evitar circularidades.
    """
    close = data[:, 0]
    volume = data[:, 2] if data.shape[1] > 2 else np.ones_like(close)

    if ma_type == 'sma': return sma(close, period)
    if ma_type == 'ema': return ema(close, period)
    if ma_type == 'wma': return wma(close, period)
    if ma_type == 'hma': return hma(close, period)
    if ma_type == 'dema': return dema(close, period)
    if ma_type == 'tema': return tema(close, period)
    if ma_type == 'tma': return tma(close, period)
    if ma_type == 'rma': return rma(close, period)
    if ma_type == 'zlema': return zlema(close, period)
    if ma_type == 'kama': return kama(close, period)
    if ma_type == 'alma': return alma(close, period)
    if ma_type == 'vwma': return vwma(close, volume, period)
    if ma_type == 'mcginley': return mcginley_dynamic(close, period)
    if ma_type == 'vidya': return vidya(close, period)
    
    return sma(close, period)
