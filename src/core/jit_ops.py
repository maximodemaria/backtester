import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _compute_quad_ma_signals_inplace(fe_ma, se_ma, fx_ma, sx_ma, out_signals):
    """Lógica de cruce de 4 medias optimizada con buffer pre-alocado."""
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
def _compute_quad_ma_signals(fe_ma, se_ma, fx_ma, sx_ma):
    """Wrapper compatible que aloca (USO NO MASIVO)."""
    out = np.zeros(len(fe_ma), dtype=np.int8)
    _compute_quad_ma_signals_inplace(fe_ma, se_ma, fx_ma, sx_ma, out)
    return out

@njit(cache=True, fastmath=True)
def _get_params_jit(global_idx, n_pairs, legal_pairs):
    """Decodificador de índice a parámetros."""
    ma_idx = global_idx // (n_pairs * n_pairs)
    remainder = global_idx % (n_pairs * n_pairs)
    entry_idx = remainder // n_pairs
    exit_idx = remainder % n_pairs
    fe_se = legal_pairs[entry_idx]
    fx_sx = legal_pairs[exit_idx]
    return np.array([ma_idx, fe_se[0], fe_se[1], fx_sx[0], fx_sx[1]], dtype=np.int32)

@njit(cache=True, fastmath=True)
def _get_signals_by_indices_inplace_jit(data, ma_type_idx, fe, se, fx, sx, indicator_map_matrix, out_signals):
    """Genera señales in-place a partir de los índices pre-calculados."""
    fe_p_idx = 0 if fe == 1 else fe // 5
    se_p_idx = 0 if se == 1 else se // 5
    fx_p_idx = 0 if fx == 1 else fx // 5
    sx_p_idx = 0 if sx == 1 else sx // 5
    
    fe_col = indicator_map_matrix[ma_type_idx, fe_p_idx]
    se_col = indicator_map_matrix[ma_type_idx, se_p_idx]
    fx_col = indicator_map_matrix[ma_type_idx, fx_p_idx]
    sx_col = indicator_map_matrix[ma_type_idx, sx_p_idx]
    
    _compute_quad_ma_signals_inplace(
        data[:, fe_col], data[:, se_col],
        data[:, fx_col], data[:, sx_col],
        out_signals
    )

@njit(cache=True, fastmath=True)
def _get_signals_by_indices_jit(data, ma_type_idx, fe, se, fx, sx, indicator_map_matrix):
    """Wrapper compatible que aloca (USO NO MASIVO)."""
    out = np.zeros(len(data), dtype=np.int8)
    _get_signals_by_indices_inplace_jit(data, ma_type_idx, fe, se, fx, sx, indicator_map_matrix, out)
    return out

@njit(cache=True, fastmath=True)
def _check_single_survival_jit(log_returns, signals, n_windows, window_size, threshold):
    """Evaluación de supervivencia ultra-rápida."""
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size
        
        # Cálculo de PF inlined para evitar slicing de arrays si es posible
        wins = 0.0
        losses = 0.0
        for i in range(start + 1, end):
            ret = signals[i-1] * log_returns[i]
            if ret > 0: wins += ret
            elif ret < 0: losses += abs(ret)
        
        pf = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
        if pf < threshold:
            return False
    return True

@njit(cache=True, fastmath=True)
def _compute_metrics_inplace_jit(log_returns, signals, comm_factor, out_strategy_returns):
    """Motor matemático optimizado con buffer de retornos pre-alocado."""
    n = len(log_returns)
    # out_strategy_returns ya viene con ceros o se limpia externamente si es necesario
    # Aquí lo procesamos directamente
    wins = 0.0
    losses = 0.0
    total_ret = 0.0
    
    for i in range(1, n):
        ret = signals[i-1] * log_returns[i]
        prev_pos = signals[i-2] if i > 1 else 0.0
        if signals[i-1] != prev_pos:
            ret -= abs(signals[i-1] - prev_pos) * comm_factor
        
        out_strategy_returns[i] = ret
        total_ret += ret
        if ret > 0: wins += ret
        elif ret < 0: losses += abs(ret)
    
    pf = wins / losses if losses > 0 else (wins if wins > 0 else 0.0)
    return pf, total_ret
