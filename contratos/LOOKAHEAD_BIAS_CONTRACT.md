# CONTRATO DE PREVENCIÓN: LOOKAHEAD BIAS

El Lookahead Bias ocurre cuando información del futuro (ej. el precio de cierre de la barra 't' o datos posteriores) se filtra en el cálculo de la señal en la barra 't-1'. Para garantizar la validez del Framework, todo código nuevo debe cumplir con las siguientes cláusulas:

## 1. Regla de "Acceso a Información" (Causa Fundamental)

- La estrategia solo puede acceder a datos en el índice `t` o anteriores para decidir la posición en `t+1`.
- Prohibido el uso de funciones de `pandas` con parámetros de "alineación" o "shift" incorrectos que expongan datos futuros.

## 2. Reglas de Implementación Vectorizada

- **Indexación:** Si la estrategia calcula una señal basada en el cierre de la vela actual, la posición resultante debe ser ejecutada obligatoriamente en la **apertura de la vela siguiente**.
- **Shift Explícito:** El vector de posiciones calculado debe ser desplazado una unidad hacia adelante (`.shift(1)`) antes de multiplicarse por los retornos logarítmicos.
  - _Correcto:_ `retorno_estrategia = posicion.shift(1) * log_returns`
  - _Incorrecto:_ `retorno_estrategia = posicion * log_returns` (Esto causa un lookahead bias inmediato).

## 3. Restricciones en Funciones de Ventana (Rolling/Expanding)

- Toda función de ventana móvil (ej. `rolling(n).mean()`) debe asegurarse de que la ventana termine en `t` y no incluya el valor de `t+1` o superior.
- Al usar `Numba`, asegúrate de que el loop o la función no tome valores fuera del rango `[0, t]`.

## 4. Validación de Integridad (El Test de Humo)

Para verificar que no hay Lookahead Bias, todo nuevo plugin debe superar este test de integridad antes de entrar al `BacktesterEngine`:

```python
def check_for_lookahead(signal_vector, data_index):
    # La señal en 't' no debe ser correlacionada
    # con el retorno de la barra 't' ni anteriores.
    # Si detecta correlación, lanza excepción.
    pass

```

## 5. Checklist de Auditoría

- [ ] ¿He aplicado `.shift(1)` a mi vector de señales antes de calcular el rendimiento?
- [ ] ¿Mi lógica de entrada usa `data['close']` en la barra actual para decidir la entrada en la apertura de la siguiente?
- [ ] ¿He verificado que ningún dato del futuro haya sido introducido durante el feature engineering?

## 6. ¿Cómo lo integras en tu framework?

1. **En tu motor:** Haz que el `BacktesterEngine` lance un `Warning` o una `Exception` si detecta que no se aplicó el `.shift(1)` en el vector de señales.
2. **Validación Automática:** Puedes incluir una función en tu `DataProcessor` que verifique la correlación entre la señal calculada y los datos futuros; si es demasiado alta, es una señal inequívoca de lookahead.
