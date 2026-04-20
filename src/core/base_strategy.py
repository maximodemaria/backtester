from abc import ABC, abstractmethod
import numpy as np

class LazyGrid:
    """
    Wrapper para generadores que permite reportar un tamaño total.
    Ideal para optimización de memoria en grids masivos.
    """
    def __init__(self, generator_func, total_count):
        self.generator_func = generator_func
        self.total_count = total_count

    def __iter__(self):
        return self.generator_func()

    def __len__(self):
        return self.total_count

class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias en el framework.
    Define el contrato que garantiza la compatibilidad con el motor de validación.
    """

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def param_grid(self):
        """
        Retorna un iterable (lista o LazyGrid) de diccionarios con las combinaciones.
        """

    @abstractmethod
    def generate_signal(self, data: np.ndarray, params: dict) -> np.ndarray:
        """
        Genera el vector de posiciones (-1, 0, 1) basado en los datos y parámetros.
        DEBE ser vectorizado y llamar a una función decorada con @njit.
        """
