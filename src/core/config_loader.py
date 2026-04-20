import yaml
import os

class ConfigValidationError(Exception):
    """Excepción lanzada cuando hay errores en la configuración YAML."""
    pass

class EnvironmentConfig:
    """
    Carga y valida configuraciones de entorno desde archivos YAML.
    Asegura la inmutabilidad y la integridad de los datos antes de la ejecución.
    """

    def __init__(self, template_name: str):
        self.template_path = os.path.join("templates", f"{template_name}.yaml")
        self.config = self._load_and_validate()

    def _load_and_validate(self) -> dict:
        """Carga el YAML y valida campos obligatorios."""
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Template no encontrado: {self.template_path}")

        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigValidationError(f"Error al parsear YAML: {str(e)}")

        # Validaciones de Estructura
        required_env = ['dataset_path', 'commission_bps']
        if 'environment' not in config:
            raise ConfigValidationError("Falta la sección 'environment' en el template.")
        
        for field in required_env:
            if field not in config['environment']:
                raise ConfigValidationError(f"Falta el campo obligatorio '{field}' en la sección environment.")

        return config

    @property
    def dataset_path(self) -> str:
        return self.config['environment']['dataset_path']

    @property
    def commission_bps(self) -> float:
        return float(self.config['environment']['commission_bps'])

    @property
    def strategy_params(self) -> dict:
        return self.config.get('strategy_params', {})

    def __repr__(self):
        return f"<EnvironmentConfig template={self.template_path}>"
