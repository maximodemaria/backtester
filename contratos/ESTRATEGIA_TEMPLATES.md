# Guía de Implementación: Plugins de Estrategia

Para que una estrategia sea compatible con el Framework, debe seguir la estructura definida en `BaseStrategy`. Todo nuevo plugin debe ser autocontenido y cumplir con los siguientes requisitos:

## 1. Estructura de Archivo

- Debe ser un archivo `.py` ubicado en la carpeta `/strategies`.
- Debe heredar de `BaseStrategy` (definida en `core.base`).

## 2. Requerimientos de Código (Contrato)

### A. Clase Base

El agente debe implementar la siguiente estructura:

```python
import numpy as np
from numba import jit
from core.base import BaseStrategy

class MiEstrategia(BaseStrategy):

    @property
    def param_grid(self):
        """
        Retorna una lista de diccionarios con las combinaciones a testear.
        Ejemplo: return [{'periodo': 10}, {'periodo': 20}]
        """
        pass

    @staticmethod
    @jit(nopython=True)
    def _compute_signal(data, param_value):
        """
        Lógica vectorizada compilada con Numba.
        Recibe el buffer de datos y el parámetro.
        Retorna el vector de posiciones (-1, 0, 1).
        """
        # LA LÓGICA DEBE SER 100% VECTORIZADA
        # Ejemplo: signal = np.where(data['close'] > data['ma'], 1, -1)
        pass

    def generate_signal(self, data, params):
        return self._compute_signal(data, params['nombre_param'])

```

1. **Vectorización:** Prohibido usar `for` o `while` para recorrer los datos. Toda la lógica de señales debe ser operada sobre arrays de numpy.
2. **Numba:** La función que calcula la señal (`_compute_signal`) **debe** estar decorada con `@jit(nopython=True)`.
3. **Inmutabilidad:** La estrategia no debe modificar el dataframe original; debe retornar un nuevo array de señales.
4. **Dimensiones:** El vector de posiciones debe tener la misma longitud que el vector de retornos (`len(data)`).

## 3. Ejemplo de Integración

Cuando crees una estrategia, el framework la instanciará automáticamente. Asegúrate de que los nombres de las llaves en el `param_grid` coincidan exactamente con los argumentos que espera `_compute_signal`.

## 4. Checklist antes de enviar

- [ ] ¿El archivo hereda de `BaseStrategy`?
- [ ] ¿He utilizado `@jit(nopython=True)` en el cálculo de señales?
- [ ] ¿La señal resultante es un array de numpy de tipo `int8` o `float64`?
- [ ] ¿He validado que el `param_grid` sea una lista de diccionarios?
