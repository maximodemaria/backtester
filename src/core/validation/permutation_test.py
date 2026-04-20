"""
Módulo para la ejecución de tests de permutación estadísticos.
"""
import numpy as np
from numba import njit, prange
from src.core.backtester import _compute_metrics_jit

# Fallback para linters estáticos que no reconocen prange
if False:  # pylint: disable=using-constant-test
    prange = range

class PermutationTest:
    """
    Implementa el test de permutación (Bar-Shuffling) para detectar sesgo de selección.
    Compara el rendimiento observado contra distribuciones aleatorias.
    """

    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations

    def run_test(self, log_returns: np.ndarray, signals: np.ndarray, observed_pf: float) -> float:
        """
        Ejecuta el test de Monte Carlo.

        Returns:
            float: Quasi P-Value. Valores bajos (< 0.05) indican robustez.
        """
        # Generar p-value usando Numba paralelo
        better_count = _run_monte_carlo_jit(log_returns, signals, observed_pf, self.n_permutations)
        return better_count / self.n_permutations

@njit(parallel=True, cache=True)
def _run_monte_carlo_jit(log_returns, signals, observed_pf, n_permutations):
    """
    Simulación de Monte Carlo con bar-shuffling.
    Utiliza prange para paralelizar las simulaciones.
    """
    better_than_observed = 0

    for _ in prange(n_permutations):
        # Bar-shuffling: permutamos los retornos manteniendo las señales estáticas
        # Esto rompe la correlación temporal si el rendimiento fue por azar
        shuffled_returns = np.copy(log_returns)
        np.random.shuffle(shuffled_returns)

        # Calculamos PF sobre retornos aleatorios
        metrics = _compute_metrics_jit(shuffled_returns, signals)
        perm_pf = metrics[0]

        if perm_pf > observed_pf:
            better_than_observed += 1

    return better_than_observed
