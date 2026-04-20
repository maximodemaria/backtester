import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _njit_execution_engine(data, logic_id, params, signals_buf):
    """
    Despachador centralizado de lógicas de trading (HFT-ENGINE).
    logic_id: Selector de kernel.
    params: Vector plano de parámetros e índices de columnas.
    """
    if logic_id == 1:
        # Lógica QuadMA basada en 4 columnas
        _logic_quad_ma(data, params, signals_buf)
    # elif logic_id == 2: ... (Nuevas lógicas aquí)

@njit(cache=True, fastmath=True)
def _logic_quad_ma(data, params, out_signals):
    """
    Kernel de ejecución para la estrategia QuadMA / FlexMA.
    params: [fe_col, se_col, fx_col, sx_col, comm_factor, ...]
    """
    n = data.shape[0]
    fe_col = int(params[0])
    se_col = int(params[1])
    fx_col = int(params[2])
    sx_col = int(params[3])
    
    fe_raw = data[:, fe_col]
    se_raw = data[:, se_col]
    fx_raw = data[:, fx_col]
    sx_raw = data[:, sx_col]
    
    current_pos = 0
    for i in range(1, n):
        # Lógica de Salida
        new_pos = current_pos
        if current_pos == 1:
            if fx_raw[i] < sx_raw[i]: new_pos = 0
        elif current_pos == -1:
            if fx_raw[i] > sx_raw[i]: new_pos = 0
        
        # Lógica de Entrada
        if new_pos == 0:
            if fe_raw[i] > se_raw[i]: new_pos = 1
            elif fe_raw[i] < se_raw[i]: new_pos = -1
            
        current_pos = new_pos
        out_signals[i] = current_pos

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
def _get_flex_params_jit(global_idx, n_types, n_pairs, legal_pairs):
    """
    Decodificador Flex-MA NJIT para 1.7B combinaciones.
    Estructura optimizada para evitar overhead de objetos.
    """
    # 1. Extraer pares de periodos (n_pairs^2)
    idx_pairs = global_idx % (n_pairs * n_pairs)
    rem_types = global_idx // (n_pairs * n_pairs)
    
    pair_exit_idx = idx_pairs % n_pairs
    pair_entry_idx = idx_pairs // n_pairs
    
    # 2. Extraer los 4 tipos de media (n_types^4)
    t4 = rem_types % n_types
    rem_types //= n_types
    t3 = rem_types % n_types
    rem_types //= n_types
    t2 = rem_types % n_types
    t1 = rem_types // n_types
    
    p_entry = legal_pairs[pair_entry_idx]
    p_exit = legal_pairs[pair_exit_idx]
    
    # Retornamos vector plano: [t1, t2, t3, t4, fe, se, fx, sx]
    return np.array([t1, t2, t3, t4, p_entry[0], p_entry[1], p_exit[0], p_exit[1]], dtype=np.int32)

@njit(cache=True, fastmath=True)
def _get_signals_by_indices_inplace_jit(data, t_idx, fe, se, fx, sx, indicator_map_matrix, out_signals):
    """
    Genera señales in-place con soporte para 4 tipos de media independientes.
    t_idx: array [t_fe, t_se, t_fx, t_sx]
    """
    fe_p_idx = 0 if fe == 1 else fe // 10
    se_p_idx = 0 if se == 1 else se // 10
    fx_p_idx = 0 if fx == 1 else fx // 10
    sx_p_idx = 0 if sx == 1 else sx // 10
    
    fe_col = indicator_map_matrix[t_idx[0], fe_p_idx]
    se_col = indicator_map_matrix[t_idx[1], se_p_idx]
    fx_col = indicator_map_matrix[t_idx[2], fx_p_idx]
    sx_col = indicator_map_matrix[t_idx[3], sx_p_idx]
    
    _compute_quad_ma_signals_inplace(
        data[:, fe_col], data[:, se_col],
        data[:, fx_col], data[:, sx_col],
        out_signals
    )

@njit(cache=True, fastmath=True)
def _get_signals_by_indices_jit(data, t_idx, fe, se, fx, sx, indicator_map_matrix):
    """Wrapper compatible que aloca (USO NO MASIVO)."""
    out = np.zeros(len(data), dtype=np.int8)
    _get_signals_by_indices_inplace_jit(data, t_idx, fe, se, fx, sx, indicator_map_matrix, out)
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

@njit(cache=True, fastmath=True)
def _extract_trades_jit(signals, prices, comm_factor):
    """
    Extrae la lista de trades de una serie de señales.
    Retorna un array 2D: [[entry_idx, exit_idx, entry_price, exit_price, net_ret], ...]
    """
    n = len(signals)
    # Estimación conservadora de número de trades para alocar memoria
    max_trades = n // 2
    trades = np.zeros((max_trades, 5), dtype=np.float64)
    trade_count = 0
    
    current_pos = 0
    entry_idx = 0
    entry_price = 0.0
    accumulated_ret = 0.0
    
    # Manejamos el cambio de log_returns de forma interna para precisión
    # o simplemente usamos los precios si están disponibles.
    
    for i in range(1, n):
        prev_pos = current_pos
        current_pos = signals[i-1]
        
        # Detectar Cambio de Posición
        if current_pos != prev_pos:
            # 1. Si veníamos de una posición, cerramos el trade anterior
            if prev_pos != 0:
                exit_idx = i - 1
                exit_price = prices[exit_idx]
                
                # Calculamos retorno neto (simplificado para este motor)
                # En un sistema real usaríamos la serie acumulada, aquí aproximamos
                # o calculamos el log-return total.
                raw_ret = prev_pos * (np.log(exit_price / entry_price))
                net_ret = raw_ret - (abs(current_pos - prev_pos) * comm_factor)
                
                trades[trade_count, 0] = entry_idx
                trades[trade_count, 1] = exit_idx
                trades[trade_count, 2] = entry_price
                trades[trade_count, 3] = exit_price
                trades[trade_count, 4] = net_ret
                trade_count += 1
            
            # 2. Si entramos en una nueva posición (o Reversal)
            if current_pos != 0:
                entry_idx = i - 1
                entry_price = prices[entry_idx]
                # Incurre en comisión de entrada
                # (Aproximación: restamos medio round-trip o el costo de entrada)
    
    return trades[:trade_count]
