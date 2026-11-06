#import para el funcionamiento correcto del script:
import pyttsx3
import queue
import threading
from PyQt6.QtCore import QThread


# --- Hilo de síntesis de voz para hablar las traducciones sin bloquear la interfaz ---
class HiloTTS(QThread):
    """
    Hilo que maneja la síntesis de voz para leer en voz alta las traducciones sin bloquear la interfaz principal.
    Utiliza una cola de mensajes para recibir frases a hablar y permite detenerse de manera ordenada.
    """
    def __init__(self):
        """
        Inicializa el hilo de TTS y prepara la cola de mensajes para recibir frases a hablar.
        """
        super().__init__()
        # La cola de mensajes permite que la interfaz principal envíe frases al hilo de TTS sin bloquearse, y el hilo procesa cada frase en orden.
        self.cola_mensajes = queue.Queue()
        self._stop_event = threading.Event()

    def run(self):
        """
        Procesa la cola de mensajes y sintetiza voz con pyttsx3.
        """
        pythoncom_module = None
        try:
            import pythoncom as _pythoncom
            _pythoncom.CoInitialize()
            pythoncom_module = _pythoncom
        except Exception:
            pythoncom_module = None
        
        # Mantenemos el hilo activo mientras no se haya solicitado su detención y procesamos los mensajes de la cola de manera segura.
        try:
            while not self._stop_event.is_set():
                try:
                    item = self.cola_mensajes.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                mensaje, config_voces = item
                # Cada frase se dice con una instancia nueva del motor para evitar que se mezclen mensajes viejos con los nuevos.
                try:
                    engine = pyttsx3.init()

                    # Intentamos configurar la voz, velocidad y volumen según la configuración proporcionada por el usuario.
                    try:
                        voz_id = config_voces.get("voz_id")
                        if voz_id:
                            try:
                                engine.setProperty("voice", voz_id)
                            except Exception:
                                pass
                        
                        # Intentamos establecer la velocidad y el volumen de la voz según la configuración del usuario.
                        try:
                            engine.setProperty("rate", config_voces.get("velocidad", 150))
                            engine.setProperty("volume", config_voces.get("volumen", 1.0))
                        except Exception:
                            pass

                        # Este paso es el que realmente reproduce la frase en voz; si falla, se captura el error y se continúa.
                        engine.say(mensaje)
                        engine.runAndWait()
                    finally:
                        # Aunque salga un error, se limpia el motor para no dejar recursos abiertos.
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        # Intentamos eliminar la instancia del motor para liberar recursos y evitar fugas de memoria.
                        try:
                            del engine
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Error crítico en motor de voz: {e}")

        # Si ocurre un error inesperado en el hilo de síntesis de voz, se captura y se imprime en la consola para su depuración.
        finally:
            if pythoncom_module is not None:
                try:
                    pythoncom_module.CoUninitialize()
                except Exception:
                    pass

    def hablar(self, mensaje, config_voces=None):
        """
        Agrega un mensaje a la cola de síntesis de voz. Si no se proporciona configuración de voz, se usan valores predeterminados.
        """
        if config_voces is None:
            config_voces = {}
        self.cola_mensajes.put((mensaje, config_voces))

    def detener(self):
        """
        Detiene el hilo de TTS de manera segura.
        """
        self._stop_event.set()
        self.cola_mensajes.put(None)
        self.wait()