import os
import signal
import atexit
import psutil
import sys

class ProcessGuard:
    """
    Sistema de seguridad para prevenir procesos huérfanos (Zombies) en Windows.
    Rastrea y mata recursivamente a todos los procesos hijos al salir.
    """
    _instance = None

    def __init__(self, logger=None):
        self.logger = logger
        self.main_pid = os.getpid()
        self._registered = False

    @classmethod
    def get_instance(cls, logger=None):
        if cls._instance is None:
            cls._instance = cls(logger)
        return cls._instance

    def register(self):
        """Registra los manejadores de salida global."""
        if self._registered:
            return
        
        atexit.register(self.cleanup)
        # En Windows, SIGINT y SIGTERM son los más comunes
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._registered = True
        if self.logger:
            self.logger.log("Proceso Guardián activado (Protección Anti-Zombies).")

    def _signal_handler(self, signum, frame):
        """Maneja señales de interrupción (Ctrl+C)."""
        if self.logger:
            self.logger.log(f"Señal de interrupción recibida ({signum}). Iniciando limpieza forzosa...")
        self.cleanup()
        sys.exit(signum)

    def cleanup(self):
        """Mata a todos los procesos hijos de forma recursiva."""
        try:
            parent = psutil.Process(self.main_pid)
            children = parent.children(recursive=True)
            
            if not children:
                return

            if self.logger:
                self.logger.log(f"Limpiando {len(children)} procesos hijos...")

            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if self.logger:
                self.logger.log("Limpieza completada. Sistema estable.")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error durante la limpieza de procesos: {str(e)}")

def init_global_guard(logger=None):
    guard = ProcessGuard.get_instance(logger)
    guard.register()
    return guard
