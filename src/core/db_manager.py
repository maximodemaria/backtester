import sqlite3
import json
from datetime import datetime
import os

class BacktestDB:
    """
    Gestor de persistencia SQLite para trazabilidad y auditoría de backtests.
    Centraliza el almacenamiento de metadatos, resultados IS, Montecarlo y OOS.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # Crear directorio para la base de datos si no existe
            os.makedirs("results/audit", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_path = f"results/audit/run_{timestamp}.db"
        
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # PRAGMAs de Performance para alta carga
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = -10000") # 10MB
            
            # Tabla de ejecuciones (Metadatos)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy_name TEXT,
                    config_template TEXT,
                    commission_bps REAL,
                    dataset_path TEXT
                )
            """)
            # Tabla de resultados In-Sample (Masivo)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategies_is (
                    run_id INTEGER,
                    params TEXT,
                    profit_factor REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            # Tabla de resultados Montecarlo (Robustez)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS montecarlo_results (
                    run_id INTEGER,
                    strategy_params TEXT,
                    p_value REAL,
                    is_robust BOOLEAN,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            # Tabla de resultados Out-of-Sample (Final)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS oos_results (
                    run_id INTEGER,
                    strategy_params TEXT,
                    oos_pf REAL,
                    oos_sharpe REAL,
                    oos_return REAL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            
            # NUEVA: Tabla de reportería total (supervivientes)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_report (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    params TEXT,
                    metrics_is TEXT,
                    metrics_oos TEXT,
                    montecarlo_p REAL,
                    trades_json TEXT,
                    equity_curve_path TEXT,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                )
            """)
            
            # Índices para búsquedas rápidas
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_run_id ON strategies_is(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mc_run_id ON montecarlo_results(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_oos_run_id ON oos_results(run_id)")
            
            conn.commit()

    def create_run(self, strategy_name, config_template, commission_bps, dataset_path):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (strategy_name, config_template, commission_bps, dataset_path)
                VALUES (?, ?, ?, ?)
            """, (strategy_name, config_template, commission_bps, dataset_path))
            return cursor.lastrowid

    def save_is_batch(self, run_id, results):
        """Inserción masiva de resultados IS en una sola transacción."""
        if not results:
            return
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = []
            for res in results:
                # res es (params, metrics_dict)
                params_json = json.dumps(res[0])
                pf = res[1].get('profit_factor', 0.0)
                data.append((run_id, params_json, pf))
                
            cursor.executemany("""
                INSERT INTO strategies_is (run_id, params, profit_factor)
                VALUES (?, ?, ?)
            """, data)
            conn.commit()

    def save_montecarlo(self, run_id, params, p_val, is_robust):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO montecarlo_results (run_id, strategy_params, p_value, is_robust)
                VALUES (?, ?, ?, ?)
            """, (run_id, json.dumps(params), p_val, is_robust))
            conn.commit()

    def save_strategy_report_batch(self, run_id, reports):
        """Guarda de forma masiva los reportes de los supervivientes."""
        if not reports:
            return
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = []
            for r in reports:
                data.append((
                    run_id,
                    json.dumps(r['params']),
                    json.dumps(r['metrics_is']),
                    json.dumps(r['metrics_oos']),
                    r['montecarlo_p'],
                    json.dumps(r['trades']),
                    r['equity_curve_path']
                ))
                
            cursor.executemany("""
                INSERT INTO strategy_report 
                (run_id, params, metrics_is, metrics_oos, montecarlo_p, trades_json, equity_curve_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
