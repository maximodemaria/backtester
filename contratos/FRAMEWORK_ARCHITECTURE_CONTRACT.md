# Contrato Maestro de Arquitectura y Estándares

Este documento es la **Fuente de Verdad** para el desarrollo dentro del framework de backtesting. Define las reglas obligatorias de diseño, cálculo y prevención de sesgos.

---

## 1. Estándar de Construcción de Estrategias

Todas las estrategias deben heredar de `BaseStrategy` y residir en `src/strategies/`.

### Estructura Obligatoria
```python
from src.core.base_strategy import BaseStrategy
from src.indicators import sma, rsi # Importar siempre de la librería centralizada

class MiEstrategia(BaseStrategy):
    def __init__(self):
        super().__init__(name="NombreUnico")

    @property
    def param_grid(self) -> list:
        # Retorna lista de diccionarios con permutaciones de parámetros
        return [{'p1': 10}, {'p1': 20}]

    def generate_signal(self, data: np.ndarray, params: dict) -> np.ndarray:
        # Responsabilidad: Lógica de decisión, NO de cálculo técnico.
        # close = data[:, 0]
        # indicador = sma(close, params['p1'])
        # return _mi_logica_jit(indicador)
        pass
```

---

## 2. Capa de Indicadores (Cálculo Técnico)

**Regla de Oro:** Queda estrictamente prohibido implementar bucles de cálculo de indicadores (SMA, EMA, RSI, etc.) dentro de las clases de estrategia.

- **Uso Obligatorio:** Se debe utilizar exclusivamente la suite de indicadores ubicada en [src/indicators/](file:///c:/Users/max/Desktop/Drive/backtester/src/indicators/).
- **Arquitectura Atomizada:** Cada indicador reside en su propio script individual (ej. `sma.py`, `rsi.py`). Esto permite un mantenimiento granular y evita archivos monolíticos.
- **Registro de Nuevos Indicadores:** Para añadir un indicador, se debe crear un nuevo archivo `.py` en `src/indicators/` y exportarlo en el `__init__.py` de dicha carpeta.
- **Funciones Puras:** Los indicadores deben ser funciones `@njit(cache=True)` que reciben arrays y devuelven arrays.
- **Manejo de NaNs:** Las estrategias deben estar preparadas para manejar `np.nan` en los periodos iniciales de los indicadores.

---

## 3. Prevención de Lookahead Bias

El sesgo de información futura es inaceptable. Se aplican las siguientes reglas:

1.  **Regla t+1:** Una señal generada con información disponible al cierre de la barra `t` solo puede ejecutarse en la barra `t+1`.
2.  **Desplazamiento (Shift):** El motor de backtesting internamente aplica `signals[i-1]`. Por tanto, las estrategias deben entregar las señales "en tiempo real" y el motor se encarga de la ejecución diferida.
3.  **Hard Check:** El motor lanzará un `AssertionError` si detecta una correlación perfecta entre la señal y el retorno del mismo periodo, indicando una fuga de información.

---

## 4. Optimización y Performance

Para garantizar el rendimiento HFT:

- **Vectorización:** Prohibido el uso de loops de Python sobre datos de precios. Todo debe ser NumPy + Numba.
- **Numba:** Toda lógica de cálculo pesada debe estar en funciones decoradas con `@njit(cache=True)`.
- **Zero-Copy:** Mantener la contigüidad de memoria (`C_CONTIGUOUS`) y usar tipos de datos `float64` para evitar copias innecesarias.

---

## 5. Checklist de Calidad para el Desarrollador

- [ ] ¿La estrategia hereda de `BaseStrategy`?
- [ ] ¿Se utilizan indicadores de `src.indicators` en lugar de implementaciones locales?
- [ ] ¿Se ha evitado el acceso a información futura?
- [ ] ¿Las funciones críticas están decoradas con `@njit`?
- [ ] ¿El archivo sigue la nomenclatura de la arquitectura actual?

---
> [!IMPORTANT]
> El incumplimiento de este contrato resultará en fallos automáticos del pipeline de validación y rechazo del código.
