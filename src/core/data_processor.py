from multiprocessing import shared_memory
import os
import numpy as np
import pandas as pd
from numba import njit

class DataProcessor:
    """
    Motor de pre-procesamiento optimizado para HFT con soporte para Shared Memory.
    Maneja la carga, transformación y partición de datos con enfoque Zero-Copy.
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None  # Buffer numpy float64
        self._shm = None  # Referencia para persistencia en Windows

    def load_data(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Carga datos desde un DataFrame o archivo (CSV/Parquet).
        Se asume que el input tiene columnas ['timestamp', 'close', 'volume'].
        """
        if df is None and self.data_path:
            ext = os.path.splitext(self.data_path)[1].lower()
            if ext == '.parquet':
                df = pd.read_parquet(self.data_path)
            else:
                # Soportamos tanto 'timestamp' como 'datetime' como nombre de columna
                df = pd.read_csv(self.data_path)
                if 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'timestamp'})

        # Convertir a numpy asegurando memoria contigua
        close_prices = df['close'].values.astype(np.float64)
        if 'volume' in df.columns:
            volume = df['volume'].values.astype(np.float64)
        else:
            volume = np.ones(len(df))

        # Validar estabilidad numérica (prevenir log(0) o log(negativo))
        if np.any(close_prices <= 0):
            print("WARNING: Detectados precios <= 0. Ajustando a epsilon para estabilidad.")
            close_prices = np.where(close_prices <= 0, 1e-9, close_prices)

        # Calcular retornos logarítmicos usando Numba
        log_returns = self._calculate_log_returns(close_prices)

        # Estructura final: [Close, LogReturns, Volume]
        self.processed_data = np.ascontiguousarray(
            np.column_stack((close_prices, log_returns, volume))
        )

        return self.processed_data

    def load_from_array(self, data: np.ndarray):
        """Carga directa desde un array numpy (usado por workers)."""
        self.processed_data = data

    def precompute_indicators(self, ma_types, periods) -> dict:
        """
        Calcula masivamente todas las medias posibles.
        Retorna la matriz expandida y el mapa de índices.
        """
        base_data = self.processed_data
        close = base_data[:, 0]
        volume = base_data[:, 2]
        
        # Mapa: (ma_type, period) -> index_columna
        indicator_map = {}
        columns = [base_data]
        curr_idx = base_data.shape[1]
        
        from src.core.indicator_factory import get_indicator
        
        for ma in ma_types:
            for p in periods:
                res = get_indicator(base_data, ma, p)
                columns.append(res.reshape(-1, 1))
                indicator_map[(ma, p)] = curr_idx
                curr_idx += 1
                
        # Ensamblar matriz en orden Fortran (Col-Major) de forma eficiente
        n_rows = base_data.shape[0]
        # Descomponemos base_data y agregamos los indicadores
        final_columns = []
        for j in range(base_data.shape[1]):
            final_columns.append(base_data[:, j])
        
        for col in columns[1:]: # El primer elemento era base_data (ya procesado)
            final_columns.append(col.flatten())

        n_cols = len(final_columns)
        self.processed_data = np.empty((n_rows, n_cols), order='F')
        for i, col in enumerate(final_columns):
            self.processed_data[:, i] = col
            
        return indicator_map

    @staticmethod
    @njit(cache=True)
    def _calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """
        Cálculo vectorizado de retornos logarítmicos r = ln(Pt / Pt-1).
        Optimizado con JIT para velocidad extrema.
        """
        n = len(prices)
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            returns[i] = np.log(prices[i] / prices[i-1])
        return returns

    def create_shared_buffer(self) -> tuple:
        """
        Exporta los datos procesados a un segmento de memoria compartida.
        Retorna (nombre_segmento, shape).
        """
        if self.processed_data is None:
            raise ValueError("No hay datos procesados para compartir.")

        # Aseguramos contigüidad FORTRAN
        data = np.asarray(self.processed_data, order='F')
        
        # Limpiar si ya existe
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
            except:
                pass
                
        # Crear segmento y mantener referencia de instancia
        self._shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        
        # Mapear numpy array al segmento
        shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shm.buf, order='F')
        shared_array[:] = data[:]
        
        return self._shm.name, data.shape

    @staticmethod
    def connect_shared_buffer(shm_name: str, shape: tuple) -> np.ndarray:
        """
        Conecta un proceso worker a un segmento de memoria compartida existente.
        """
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        return np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)

    def split_is_oos(self, train_ratio: float = 0.7):
        """
        Divide los datos en In-Sample (IS) y Out-of-Sample (OOS).
        Utiliza vistas de memoria para evitar duplicación (Zero-Copy).
        """
        if self.processed_data is None:
            raise ValueError("Los datos no han sido cargados. Llame a load_data() primero.")

        split_idx = int(len(self.processed_data) * train_ratio)

        # Retornamos vistas (vía slicing de numpy)
        self.data_is = self.processed_data[:split_idx]
        self.data_oos = self.processed_data[split_idx:]

        return self.data_is, self.data_oos
