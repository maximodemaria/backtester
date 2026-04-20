"""
Definición de la interfaz base para estrategias de trading.
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias en el framework.
    Define el contrato que garantiza la compatibilidad con el motor de validación.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def param_grid(self) -> list:
        """
        Retorna una lista de diccionarios con las combinaciones a testear.
        Ejemplo: return [{'fast': 10, 'slow': 20}, {'fast': 20, 'slow': 50}]
        """

    @abstractmethod
    def generate_signal(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Genera el vector de posiciones (-1, 0, 1) basado en los datos y parámetros.
        DEBE ser vectorizado y llamar a una función decorada con @njit.
        """
