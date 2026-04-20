"""
Estrategia de 4 medias móviles con sistema de pre-cálculo masivo.
Soporta 14 tipos de indicadores y optimización de malla 4D.
"""
import numpy as np
from src.core.base_strategy import BaseStrategy
from src.core.jit_ops import (
    _get_params_jit, 
    _get_signals_by_indices_jit, 
    _compute_quad_ma_signals
)
from src.indicators import (
    sma, ema, wma, hma, dema, tema, tma, rma, zlema, kama, alma, vwma,
    mcginley_dynamic, vidya
)

class QuadMAStrategy(BaseStrategy):
    """
    Estrategia de 4 Medias Móviles con soporte para 14 tipos de indicadores.
    Optimizada para GGAL con pre-cálculo de indicadores (Caching).
    """

    def __init__(self):
        super().__init__(name="QuadMA")
        # Cache para evitar re-calcular indicadores en cada backtest del grid
        self._ma_cache = {}
        
        # Pre-calculamos los 820 pares legales de periodos (fe < se)
        self.periods = np.arange(5, 205, 5).astype(np.int32).tolist()
        self.periods.insert(0, 1)
        
        legal_list = []
        for fe in self.periods:
            for se in self.periods:
                if fe < se:
                    legal_list.append((fe, se))
        
        self.legal_pairs = np.array(legal_list, dtype=np.int32)
        
        self.ma_types = [
            'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'tma',
            'rma', 'zlema', 'kama', 'alma', 'vwma', 'mcginley', 'vidya'
        ]

    def get_params_by_index(self, global_idx: int) -> dict:
        """
        Mapeo matemático: global_idx -> dict(params)
        """
        params_array = _get_params_jit(global_idx, len(self.legal_pairs), self.legal_pairs)
        ma_idx = params_array[0]
        fe, se = params_array[1], params_array[2]
        fx, sx = params_array[3], params_array[4]
        
        return {
            'ma_type': self.ma_types[ma_idx],
            'fast_entry': int(fe),
            'slow_entry': int(se),
            'fast_exit': int(fx),
            'slow_exit': int(sx)
        }

    @property
    def param_grid(self):
        """
        Retorna un objeto LazyGrid que encapsula el generador de 9.4M de combinaciones.
        """
        from src.core.base_strategy import LazyGrid
        total_combinations = len(self.ma_types) * len(self.legal_pairs) * len(self.legal_pairs)
        return LazyGrid(self._generate_combinations, total_combinations)

    def _generate_combinations(self):
        """Generador interno simplificado usando el mapeo de pares."""
        for ma in self.ma_types:
            for fe, se in self.legal_pairs:
                for fx, sx in self.legal_pairs:
                    yield {
                        'ma_type': ma,
                        'fast_entry': int(fe),
                        'slow_entry': int(se),
                        'fast_exit': int(fx),
                        'slow_exit': int(sx)
                    }

    def generate_signal(self, data: np.ndarray, params: dict, indicator_map: dict = None) -> np.ndarray:
        """
        Genera señales combinando 4 medias.
        """
        ma_type = params['ma_type']
        fe = params['fast_entry']
        se = params['slow_entry']
        fx = params['fast_exit']
        sx = params['slow_exit']

        if indicator_map:
            fe_idx = indicator_map[(ma_type, fe)]
            se_idx = indicator_map[(ma_type, se)]
            fx_idx = indicator_map[(ma_type, fx)]
            sx_idx = indicator_map[(ma_type, sx)]
            
            return _compute_quad_ma_signals(
                data[:, fe_idx], data[:, se_idx],
                data[:, fx_idx], data[:, sx_idx]
            )

        ma_fast_entry = self._get_ma(data, ma_type, fe)
        ma_slow_entry = self._get_ma(data, ma_type, se)
        ma_fast_exit = self._get_ma(data, ma_type, fx)
        ma_slow_exit = self._get_ma(data, ma_type, sx)

        return _compute_quad_ma_signals(
            ma_fast_entry, ma_slow_entry,
            ma_fast_exit, ma_slow_exit
        )

    def _get_ma(self, data: np.ndarray, ma_type: str, period: int) -> np.ndarray:
        """Selector de media con caching."""
        key = (ma_type, period)
        data_id = id(data)
        if data_id not in self._ma_cache:
            self._ma_cache[data_id] = {}
        cache = self._ma_cache[data_id]

        if key in cache:
            return cache[key]

        from src.core.indicator_factory import get_indicator
        res = get_indicator(data, ma_type, period)

        cache[key] = res
        return res
