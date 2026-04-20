"""
Sistema de logging asíncrono para reportar eventos sin bloquear el cálculo HFT.
"""
import multiprocessing
import time
import sys

class AsyncLogger:
    """
    Versión simplificada y sincrónica para debugging.
    """
    def __init__(self, max_buffer: int = 1000):
        pass

    def log(self, message: str):
        print(f"[HFT-FRAMEWORK] {time.strftime('%H:%M:%S')} | {message}", flush=True)

    def stop(self):
        pass
