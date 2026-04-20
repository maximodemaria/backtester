import importlib
import inspect
from src.core.base_strategy import BaseStrategy

def get_strategy_instance(strategy_name: str) -> BaseStrategy:
    """
    Carga e instancia dinámicamente una estrategia por su nombre.
    Busca en el directorio src.strategies.
    """
    # Mapeo de nombres comunes a rutas de módulos (snake_case)
    # Si el nombre es "QuadMA", buscamos en "src.strategies.quad_ma"
    module_name = "".join(["_" + c.lower() if c.isupper() else c for c in strategy_name]).lstrip("_")
    
    try:
        # Intentar importación dinámica
        module = importlib.import_module(f"src.strategies.{module_name}")
        
        # Buscar la clase que herede de BaseStrategy en ese módulo
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                return obj()
                
    except ImportError:
        # Fallback: Algunos nombres pueden no seguir la convención snake_case exacta
        # Intentamos buscar en todos los módulos de la carpeta
        import pkgutil
        import src.strategies as strategies_pkg
        
        for loader, name, is_pkg in pkgutil.walk_packages(strategies_pkg.__path__, "src.strategies."):
            mod = importlib.import_module(name)
            for cls_name, obj in inspect.getmembers(mod):
                if cls_name == f"{strategy_name}Strategy" or cls_name == strategy_name:
                    if inspect.isclass(obj) and issubclass(obj, BaseStrategy):
                        return obj()
    
    raise ValueError(f"Estrategia '{strategy_name}' no encontrada en src/strategies/")
