"""
Sistema de logging asíncrono para reportar eventos sin bloquear el cálculo HFT.
"""
import multiprocessing
import queue
import time
import sys

class AsyncLogger:
    """
    Sistema de logging balanceado para no bloquear el motor de cálculo HFT.
    Utiliza un proceso separado para manejar el flujo de salida.
    """
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=self._logger_worker,
            args=(self.queue, self.stop_event)
        )
        self.process.start()

    def _logger_worker(self, log_queue, stop_event):
        """Worker que reside en un proceso independiente."""
        while not stop_event.is_set() or not log_queue.empty():
            try:
                message = log_queue.get(timeout=0.1)
                print(f"[HFT-FRAMEWORK] {time.strftime('%H:%M:%S')} | {message}")
                sys.stdout.flush()
            except queue.Empty:
                continue

    def log(self, message: str):
        """Envía un mensaje a la cola sin bloquear el hilo principal."""
        self.queue.put(message)

    def stop(self):
        """Finaliza el proceso de logging de forma limpia."""
        self.stop_event.set()
        self.process.join()
