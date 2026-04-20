"""
Motor de pre-procesamiento optimizado para HFT.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from numba import njit

class DataProcessor:
    """
    Motor de pre-procesamiento optimizado para HFT.
    Maneja la carga, transformación y partición de datos con enfoque Zero-Copy.
    """

    def __init__(self, data_path: str = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None  # Buffer numpy float64

    def load_data(self, df: pd.DataFrame = None) -> np.ndarray:
        """
        Carga datos desde un DataFrame o archivo.
        Se asume que el input tiene columnas ['timestamp', 'close'].
        """
        if df is None and self.data_path:
            df = pd.read_csv(self.data_path)

        # Convertir a numpy asegurando memoria contigua
        close_prices = df['close'].values.astype(np.float64)

        # Calcular retornos logarítmicos usando Numba
        log_returns = self._calculate_log_returns(close_prices)

        # Estructura final: [Close, LogReturns]
        self.processed_data = np.ascontiguousarray(
            np.column_stack((close_prices, log_returns))
        )
        return self.processed_data

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

    def split_is_oos(self, train_ratio: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Divide los datos en In-Sample (IS) y Out-of-Sample (OOS).
        Utiliza vistas de memoria para evitar duplicación (Zero-Copy).
        """
        if self.processed_data is None:
            raise ValueError("Los datos no han sido cargados. Llame a load_data() primero.")

        split_idx = int(len(self.processed_data) * train_ratio)

        # Retornamos vistas (vía slicing de numpy)
        is_data = self.processed_data[:split_idx]
        oos_data = self.processed_data[split_idx:]

        return is_data, oos_data
