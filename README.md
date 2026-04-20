# Backtester Framework

Framework modular para validación de estrategias de trading cuantitativo y HFT.

## Estructura del Proyecto

- `src/core`: Lógica central del backtester y simulador.
- `src/strategies`: Implementación de estrategias de trading.
- `src/utils`: Herramientas auxiliares y generadores de datos.
- `contratos`: Documentación de contratos de validación y anti-bias.

## Características

- Motor de backtesting vectorial (Numba ready).
- Validación IS/OOS (In-Sample / Out-of-Sample).
- Pruebas de permutación y supervivencia.
- Arquitectura agnóstica a la latencia.
