"""
Utilidades para la generación de datos sintéticos de prueba.
"""
import numpy as np
import pandas as pd

def generate_sample_data(path: str, n_rows: int = 5000):
    """
    Genera un archivo de ejemplo (CSV o Parquet) con precios sintéticos.
    """
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n_rows)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-01', periods=n_rows, freq='h'),
        'close': price
    })

    if path.lower().endswith('.parquet'):
        df.to_parquet(path, compression='snappy', index=False)
    else:
        df.to_csv(path, index=False)

    print(f"Datos de muestra generados en: {path}")

if __name__ == "__main__":
    generate_sample_data("sample_data.csv")
