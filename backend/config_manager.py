import json
import os

# Ruta del archivo JSON que guarda la configuración global del sistema.
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Valores predeterminados que se usan cuando no existe un archivo de configuración o cuando se desea reiniciar la configuración.
CONFIG_DEFAULT = {
    "personalizacion": {
        "fuente": "Arial",
        "tamano_letra": 18,
        "tema": "Oscuro",
        "mostrar_hud": False,
        "mostrar_landmarks": False
    },
    "traduccion": {
        "modo_ia": "Híbrido (Auto MoE)",
        "ventana_frames": 6,
        "umbral_confianza": 0.80,
        "preset": "Equilibrado"
    },
    "voces": {
        "voz_id": "",
        "activar_voz": True,
        "velocidad": 150,
        "volumen": 1.0
    },
    "usuario": {
        "gmail_vinculado": None,
        "sincronizacion_nube": False
    }
}


class ConfigManager:
    """
    Clase responsable de manejar la configuración del sistema, incluyendo la carga y el guardado de ajustes en un archivo JSON.
    """
    @staticmethod
    def cargar_configuracion():
        """
        Carga la configuración disponible en disco o crea una base inicial si no existe.
        """
        # Si el archivo no existe, se crea con los valores base para que la aplicación pueda arrancar sin errores.
        if not os.path.exists(CONFIG_PATH):
            ConfigManager.guardar_configuracion(CONFIG_DEFAULT)
            return CONFIG_DEFAULT

        try:
            # Se lee el archivo JSON para conservar los ajustes previamente guardados por el usuario.
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Si el archivo está dañado o no se puede abrir, se regresa a la configuración por defecto para evitar fallos.
            return CONFIG_DEFAULT

    @staticmethod
    def guardar_configuracion(config_data):
        """
        Guarda la configuración actual en un archivo JSON legible y con soporte para texto Unicode.
        """
        # Se escribe la configuración con formato indentado para facilitar su lectura y edición manual si hace falta.
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)