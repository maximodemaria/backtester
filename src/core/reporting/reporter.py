import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ResultsReporter:
    """
    Gestor de generación de reportes, gráficos y exportación de datos.
    Centraliza la creación de artefactos visuales y tabulares de los supervivientes.
    """
    def __init__(self, run_id: int):
        self.run_id = run_id
        self.output_dir = f"results/reports/run_{run_id}"
        self.curves_dir = f"{self.output_dir}/equity_curves"
        self._init_dirs()

    def _init_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.curves_dir, exist_ok=True)

    def generate_equity_curve(self, config_id: int, IS_returns: np.ndarray, OOS_returns: np.ndarray, params: dict):
        """
        Genera y guarda un gráfico PNG de la curva de equity acumulada (IS + OOS).
        """
        plt.figure(figsize=(12, 6))
        
        # Combinar retornos
        full_returns = np.concatenate([IS_returns, OOS_returns])
        equity_curve = np.cumsum(full_returns) * 100 # En porcentaje
        
        # Dividir visualmente IS de OOS
        is_len = len(IS_returns)
        
        plt.plot(equity_curve, label='Cumulative Return (%)', color='#1f77b4', lw=2)
        plt.axvline(x=is_len, color='red', linestyle='--', alpha=0.6, label='OOS Start')
        
        # Estética
        plt.title(f"Equity Curve - Strategy ID: {config_id}\n{params}", fontsize=10)
        plt.xlabel("Bars")
        plt.ylabel("Cumulative Return (%)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Guardar
        filename = f"strategy_{config_id}_PF{params.get('pf', 0):.2f}.png"
        path = f"{self.curves_dir}/{filename}"
        plt.savefig(path, dpi=120)
        plt.close()
        return path

    def save_consolidated_metrics(self, all_reports: list):
        """
        Exporta una tabla CSV/Parquet con todas las métricas IS/OOS de los supervivientes.
        """
        df_data = []
        for r in all_reports:
            row = {
                "strategy_id": r['id'],
                **r['params'],
                "is_pf": r['metrics_is']['profit_factor'],
                "is_return": r['metrics_is']['total_return'],
                "oos_pf": r['metrics_oos']['profit_factor'],
                "oos_return": r['metrics_oos']['total_return'],
                "montecarlo_p": r['montecarlo_p']
            }
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        csv_path = f"{self.output_dir}/consolidated_metrics.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def format_trades_to_json(self, trades_array: np.ndarray, timestamps: np.ndarray = None):
        """
        Convierte el array de trades JIT en una lista de diccionarios JSON-friendly.
        """
        trades_list = []
        for i in range(len(trades_array)):
            trade = {
                "entry_idx": int(trades_array[i, 0]),
                "exit_idx": int(trades_array[i, 1]),
                "entry_price": float(trades_array[i, 2]),
                "exit_price": float(trades_array[i, 3]),
                "net_ret": float(trades_array[i, 4])
            }
            if timestamps is not None:
                trade["entry_time"] = str(timestamps[trade["entry_idx"]])
                trade["exit_time"] = str(timestamps[trade["exit_idx"]])
            trades_list.append(trade)
        return trades_list
