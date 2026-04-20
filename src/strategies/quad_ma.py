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

    def _params_to_idx(self, p: dict) -> int:
        """
        Inversa de get_params_by_index: convierte params en un global_idx.
        """
        t1 = self.ma_types.index(p['type_fe'])
        t2 = self.ma_types.index(p['type_se'])
        t3 = self.ma_types.index(p['type_fx'])
        t4 = self.ma_types.index(p['type_sx'])
        
        # Encontrar índices de pares
        p_entry = (p['fast_entry'], p['slow_entry'])
        p_exit = (p['fast_exit'], p['slow_exit'])
        
        # Búsqueda de índice de par (lineal por ser solo ~210 elementos)
        pair_entry_idx = -1
        pair_exit_idx = -1
        for i in range(self.n_pairs):
            if self.legal_pairs[i, 0] == p_entry[0] and self.legal_pairs[i, 1] == p_entry[1]:
                pair_entry_idx = i
            if self.legal_pairs[i, 0] == p_exit[0] and self.legal_pairs[i, 1] == p_exit[1]:
                pair_exit_idx = i
        
        idx_pairs = pair_entry_idx * self.n_pairs + pair_exit_idx
        idx_types = (((t1 * self.n_types + t2) * self.n_types + t3) * self.n_types + t4)
        
        return idx_types * (self.n_pairs ** 2) + idx_pairs

class QuadMAStrategy(BaseStrategy):
    """
    Estrategia Flex-MA: 4 Medias Móviles con tipos INDEPENDIENTES. 
    Espacio de búsqueda escalado a ~1.7B de combinaciones.
    """

    def __init__(self):
        super().__init__(name="FlexMA")
        self._ma_cache = {}
        
        # Resolución ajustada para escala masiva (step=10)
        # Periodos: [1, 10, 20, ..., 200] -> 21 periodos
        self.periods = np.arange(10, 210, 10).astype(np.int32).tolist()
        self.periods.insert(0, 1) # Mantenemos el 1 como "naked price"
        
        legal_list = []
        for fe in self.periods:
            for se in self.periods:
                if fe < se:
                    legal_list.append((fe, se))
        
        self.legal_pairs = np.array(legal_list, dtype=np.int32)
        self.n_pairs = len(self.legal_pairs) # Debería ser ~210
        
        self.ma_types = [
            'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'tma',
            'rma', 'zlema', 'kama', 'alma', 'vwma', 'mcginley', 'vidya'
        ]
        self.n_types = len(self.ma_types)

    def get_params_by_index(self, global_idx: int) -> dict:
        """
        Decodificador Flex-MA para 1.7B combinaciones.
        Mapeo: global_idx -> {Type1, Type2, Type3, Type4, PairEntry, PairExit}
        """
        # 1. Extraer índices de pares de periodos
        idx_pairs = global_idx % (self.n_pairs * self.n_pairs)
        rem_types = global_idx // (self.n_pairs * self.n_pairs)
        
        pair_exit_idx = idx_pairs % self.n_pairs
        pair_entry_idx = idx_pairs // self.n_pairs
        
        # 2. Extraer los 4 tipos de media
        t4_idx = rem_types % self.n_types
        rem_types //= self.n_types
        t3_idx = rem_types % self.n_types
        rem_types //= self.n_types
        t2_idx = rem_types % self.n_types
        t1_idx = rem_types // self.n_types
        
        # 3. Resolver valores reales
        p_entry = self.legal_pairs[pair_entry_idx]
        p_exit = self.legal_pairs[pair_exit_idx]
        
        return {
            'type_fe': self.ma_types[t1_idx],
            'type_se': self.ma_types[t2_idx],
            'type_fx': self.ma_types[t3_idx],
            'type_sx': self.ma_types[t4_idx],
            'fast_entry': int(p_entry[0]),
            'slow_entry': int(p_entry[1]),
            'fast_exit': int(p_exit[0]),
            'slow_exit': int(p_exit[1])
        }

    @property
    def param_grid(self):
        """
        Retorna el LazyGrid con el nuevo espacio de búsqueda de 1.7B.
        """
        from src.core.base_strategy import LazyGrid
        # 14^4 * 210^2 = 38416 * 44100 = 1,694,145,600
        total_combinations = (self.n_types ** 4) * (self.n_pairs ** 2)
        return LazyGrid(self._generate_combinations, total_combinations)

    def _generate_combinations(self):
        """Generador compatible para Flex-MA (No recomendado para 1.7B, usar índices)."""
        for t1 in self.ma_types:
            for t2 in self.ma_types:
                for t3 in self.ma_types:
                    for t4 in self.ma_types:
                        for entry in self.legal_pairs:
                            for exit_p in self.legal_pairs:
                                yield {
                                    'type_fe': t1, 'type_se': t2, 'type_fx': t3, 'type_sx': t4,
                                    'fast_entry': int(entry[0]), 'slow_entry': int(entry[1]),
                                    'fast_exit': int(exit_p[0]), 'slow_exit': int(exit_p[1])
                                }

    def generate_signal(self, data: np.ndarray, params: dict, indicator_map: dict = None) -> np.ndarray:
        """
        Genera señales combinando 4 medias con tipos independientes.
        """
        t_fe, t_se = params['type_fe'], params['type_se']
        t_fx, t_sx = params['type_fx'], params['type_sx']
        fe, se = params['fast_entry'], params['slow_entry']
        fx, sx = params['fast_exit'], params['slow_exit']

        if indicator_map:
            fe_idx = indicator_map[(t_fe, fe)]
            se_idx = indicator_map[(t_se, se)]
            fx_idx = indicator_map[(t_fx, fx)]
            sx_idx = indicator_map[(t_sx, sx)]
            
            return _compute_quad_ma_signals(
                data[:, fe_idx], data[:, se_idx],
                data[:, fx_idx], data[:, sx_idx]
            )

        ma_fe = self._get_ma(data, t_fe, fe)
        ma_se = self._get_ma(data, t_se, se)
        ma_fx = self._get_ma(data, t_fx, fx)
        ma_sx = self._get_ma(data, t_sx, sx)

        return _compute_quad_ma_signals(ma_fe, ma_se, ma_fx, ma_sx)

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
