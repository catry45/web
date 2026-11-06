import pyttsx3

from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QRadioButton, QButtonGroup,
    QPushButton, QFontComboBox, QCheckBox
)
# PyQt6.QtCore y QtGui se importan para manejar la orientación de los sliders, las señales y la apariencia de los widgets.
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# Se importa ConfigManager para manejar la carga y guardado de la configuración del usuario.
from backend.config_manager import ConfigManager

class VentanaConfiguracion(QDialog):
    """
    Ventana de ajustes del sistema para personalizar la apariencia, la IA y la voz de la aplicación.
    Permite al usuario cambiar la fuente, el tamaño de letra, el modo de operación de la IA,
    la ventana de estabilización y la voz del sintetizador TTS.
    """

    def __init__(self, parent=None):
        """
        Inicializa la ventana de configuración con pestañas para cada categoría de ajustes.
        """
        super().__init__(parent)
        self.parent_app = parent
        self.setWindowTitle("⚙️ Ajustes del Sistema - Traductor LSM")
        self.resize(650, 500)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        # Se cargan los ajustes guardados para que la interfaz muestre el estado actual del usuario al abrir la ventana.
        self.config = ConfigManager.cargar_configuracion()

        # Se crea el layout principal que contendrá las pestañas y los botones de acción.
        layout_principal = QVBoxLayout(self)

        # Se organizan las opciones en pestañas para separar claramente los ajustes visuales, de IA y de voz.
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #45475a; background: #181825; }
            QTabBar::tab { background: #313244; color: #cdd6f4; padding: 10px 15px; margin: 2px; }
            QTabBar::tab:selected { background: #89b4fa; color: #11111b; font-weight: bold; }
        """)

        # Cada pestaña agrupa un bloque de configuración específico para mantener la lógica de la ventana más clara.
        self.tab_personalizacion = QWidget()
        self.tab_traduccion = QWidget()
        self.tab_voces = QWidget()

        # Se crean las pestañas y se agregan al QTabWidget con títulos descriptivos y emojis para mejorar la experiencia del usuario.
        self.crear_tab_personalizacion()
        self.crear_tab_traduccion()
        self.crear_tab_voces()
        self.tabs.addTab(self.tab_personalizacion, "🎨 Personalización")
        self.tabs.addTab(self.tab_traduccion, "🧠 Traducción / IA")
        self.tabs.addTab(self.tab_voces, "🗣️ Voces (TTS)")

        # En la parte inferior se dejan las acciones principales: guardar cambios o cerrar sin aplicar nada.
        layout_botones = QHBoxLayout()
        btn_guardar = QPushButton("💾 Guardar Cambios")
        btn_guardar.setStyleSheet("background-color: #a6e3a1; color: #11111b; padding: 8px; font-weight: bold;")
        btn_guardar.clicked.connect(self.guardar_cambios)

        btn_cancelar = QPushButton("Cancelar")
        btn_cancelar.setStyleSheet("background-color: #f38ba8; color: #11111b; padding: 8px;")
        btn_cancelar.clicked.connect(self.reject)

        layout_botones.addStretch()
        layout_botones.addWidget(btn_cancelar)
        layout_botones.addWidget(btn_guardar)

        layout_principal.addWidget(self.tabs)
        layout_principal.addLayout(layout_botones)

    # --- PESTAÑA 1: PERSONALIZACIÓN ---
    # Aquí se ajusta la apariencia del texto principal para mejorar la legibilidad de la traducción en pantalla.
    def crear_tab_personalizacion(self):
        """
        Construye la pestaña de personalización visual de la aplicación.
        """
        layout = QVBoxLayout(self.tab_personalizacion)

        layout.addWidget(QLabel("Fuente del Texto de Traducción:"))
        self.combo_fuentes = QFontComboBox()
        self.combo_fuentes.setCurrentFont(QFont(self.config["personalizacion"]["fuente"]))
        layout.addWidget(self.combo_fuentes)

        layout.addWidget(QLabel("Tamaño de Letra:"))
        self.slider_tamano = QSlider(Qt.Orientation.Horizontal)
        self.slider_tamano.setRange(12, 36)
        self.slider_tamano.setValue(self.config["personalizacion"]["tamano_letra"])
        layout.addWidget(self.slider_tamano)

        # Controles de Interfaz (Modo Usuario / Desarrollador) ---
        layout.addWidget(QLabel("<br><b>Opciones de Visualización:</b>"))
        
        self.chk_mostrar_hud = QCheckBox("Mostrar HUD Técnico (Confianza, Motor, Arquitectura)")
        # Por defecto False si no existe
        self.chk_mostrar_hud.setChecked(self.config["personalizacion"].get("mostrar_hud", False))
        layout.addWidget(self.chk_mostrar_hud)

        self.chk_mostrar_landmarks = QCheckBox("Mostrar Puntos de Rastreo (MediaPipe Landmarks)")
        # Por defecto False si no existe
        self.chk_mostrar_landmarks.setChecked(self.config["personalizacion"].get("mostrar_landmarks", False))
        layout.addWidget(self.chk_mostrar_landmarks)

        # Se agrega un espacio flexible al final para que los elementos se mantengan arriba y no se estiren innecesariamente.
        layout.addStretch()

    # --- PESTAÑA 2: TRADUCCIÓN E IA ---
    def crear_tab_traduccion(self):
        """
        Construye la pestaña que controla la lógica de traducción y la sensibilidad del sistema.
        """
        layout = QVBoxLayout(self.tab_traduccion)

        # El usuario puede elegir si la IA trabaja en modo híbrido o si fuerza un tipo específico de análisis.
        layout.addWidget(QLabel("<b>Modo de Operación de la IA:</b>"))
        self.group_modo = QButtonGroup(self)
        
        self.rb_hibrido = QRadioButton("Híbrido Automático (Comité MoE - Estático + Dinámico)")
        self.rb_estatico = QRadioButton("Forzar Solo IA Estática (Abecedario/Números)")
        self.rb_dinamico = QRadioButton("Forzar Solo IA Dinámica (Palabras/Frases completas)")

        modo_actual = self.config["traduccion"]["modo_ia"]
        if modo_actual == "Forzar Estático": self.rb_estatico.setChecked(True)
        elif modo_actual == "Forzar Dinámico": self.rb_dinamico.setChecked(True)
        else: self.rb_hibrido.setChecked(True)

        self.group_modo.addButton(self.rb_hibrido)
        self.group_modo.addButton(self.rb_estatico)
        self.group_modo.addButton(self.rb_dinamico)

        layout.addWidget(self.rb_hibrido)
        layout.addWidget(self.rb_estatico)
        layout.addWidget(self.rb_dinamico)

        # Este valor indica cuántas decisiones se toman antes de aceptar una traducción final, lo que ayuda a reducir ruido o errores de lectura.
        layout.addWidget(QLabel("<br><b>Ventana de Estabilización (FPS / Inercia):</b>"))
        self.lbl_fps_val = QLabel(f"Ventana Actual: {self.config['traduccion']['ventana_frames']} Frames")
        self.slider_fps = QSlider(Qt.Orientation.Horizontal)
        self.slider_fps.setRange(1, 30)
        self.slider_fps.setValue(self.config["traduccion"]["ventana_frames"])
        self.slider_fps.valueChanged.connect(lambda v: self.lbl_fps_val.setText(f"Ventana Actual: {v} Frames"))

        # Se agregan controles para ajustar la sensibilidad de la IA de forma rápida mediante presets preconfigurados.
        layout.addWidget(self.lbl_fps_val)
        layout.addWidget(self.slider_fps)

        # Los presets permiten ajustar la sensibilidad del sistema de forma rápida sin cambiar cada valor de forma manual.
        layout.addWidget(QLabel("<br><b>Presets Preconfigurados:</b>"))
        btn_layout = QHBoxLayout()
        btn_rapido = QPushButton("⚡ Rápido (Sensible)")
        btn_equilibrado = QPushButton("⚖️ Equilibrado")
        btn_preciso = QPushButton("🎯 Preciso (Anti-error)")

        btn_rapido.clicked.connect(lambda: self.slider_fps.setValue(3))
        btn_equilibrado.clicked.connect(lambda: self.slider_fps.setValue(6))
        btn_preciso.clicked.connect(lambda: self.slider_fps.setValue(12))

        btn_layout.addWidget(btn_rapido)
        btn_layout.addWidget(btn_equilibrado)
        btn_layout.addWidget(btn_preciso)
        layout.addLayout(btn_layout)

        # Se agrega un espacio flexible al final para que los elementos se mantengan arriba y no se estiren innecesariamente.
        layout.addStretch()


    # --- PESTAÑA 3: VOCES TTS ---
    def _obtener_voz_espaniola(self):
        """
        Busca y devuelve la primera voz disponible en español en el sistema. 
        """
        engine = None
        # Se inicializa el motor TTS para obtener la lista de voces disponibles y filtrar por español.
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices") or []
            for voice in voices:
                nombre_voz = (getattr(voice, "name", "") or "").lower()
                id_voz = (getattr(voice, "id", "") or "").lower()

                if "spanish" in nombre_voz or "español" in nombre_voz or "es-" in id_voz or "es_" in id_voz:
                    return getattr(voice, "name", None), getattr(voice, "id", None)
            
            # Si no se encuentra ninguna voz en español, se devuelve None para ambos valores.
            return None, None
        except Exception:
            return None, None
        # Se asegura de liberar los recursos del motor TTS para evitar bloqueos o fugas de memoria.
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
                try:
                    del engine
                except Exception:
                    pass

    def crear_tab_voces(self):
        """
        Construye la pestaña para seleccionar la voz TTS disponible,
        activar la narración y ajustar velocidad y volumen de forma dinámica.
        """
        layout = QVBoxLayout(self.tab_voces)

        # Habilitador de voz
        self.chk_activar_voz = QCheckBox("🔊 Habilitar Narrador Automático (TTS)")
        self.chk_activar_voz.setChecked(self.config["voces"].get("activar_voz", True))
        self.chk_activar_voz.setStyleSheet("font-weight: bold; font-size: 14px; color: #a6e3a1;")
        layout.addWidget(self.chk_activar_voz)
        # Tarjeta informativa de la voz
        layout.addWidget(QLabel("<br><b>Voz del Sintetizador (Offline) Bloqueada a Español:</b>"))

        # Se obtiene la primera voz en español disponible en el sistema para mostrarla al usuario y permitirle probarla.
        voz_espanol_name, voz_espanol_id = self._obtener_voz_espaniola()
        self.lbl_voz_actual = QLabel()
        self.lbl_voz_actual.setStyleSheet("""
            QLabel {
                background-color: #313244; color: #a6adc8; border: 1px solid #45475a;
                border-radius: 6px; padding: 10px; font-size: 13px;
            }
        """)

        # Se muestra un mensaje de advertencia si no se encuentra ninguna voz en español,
        # para que el usuario sepa que no podrá usar TTS correctamente.
        if voz_espanol_name:
            self.lbl_voz_actual.setText(f"✅ {voz_espanol_name}")
            self.voz_seleccionada_id = voz_espanol_id
        else:
            self.lbl_voz_actual.setText("⚠️ No se detectó ninguna voz en español en Windows")
            self.voz_seleccionada_id = None
            self.lbl_voz_actual.setStyleSheet(
                "QLabel { background-color: #313244; color: #f38ba8; border: 1px solid #f38ba8; padding: 10px; border-radius: 6px; }"
            )

        # Se agrega la etiqueta de voz al layout para que el usuario pueda verla claramente.
        layout.addWidget(self.lbl_voz_actual)

        # Controles de velocidad y volumen
        velocidad_actual = max(50, min(300, int(self.config["voces"].get("velocidad", 150))))
        self.lbl_velocidad = QLabel(f"<br><b>Velocidad de Lectura:</b> {velocidad_actual} PPM")
        layout.addWidget(self.lbl_velocidad)

        # Se crea un slider para ajustar la velocidad de lectura del sintetizador TTS, con un rango de 50 a 300 palabras por minuto.
        self.slider_velocidad = QSlider(Qt.Orientation.Horizontal)
        self.slider_velocidad.setRange(50, 300)
        self.slider_velocidad.setValue(velocidad_actual)
        self.slider_velocidad.valueChanged.connect(
            lambda v: self.lbl_velocidad.setText(f"<br><b>Velocidad de Lectura:</b> {v} PPM")
        )
        # Se agrega el slider al layout para que el usuario pueda interactuar con él y ajustar la velocidad de lectura.
        layout.addWidget(self.slider_velocidad)


        # Control de volumen con un rango del 1% al 100%
        volumen_raw = self.config["voces"].get("volumen", 1.0)
        volumen_actual = max(0, min(100, int(float(volumen_raw) * 100)))
        self.lbl_volumen = QLabel(f"<br><b>Volumen del Sintetizador:</b> {volumen_actual}%")
        layout.addWidget(self.lbl_volumen)

        self.slider_volumen = QSlider(Qt.Orientation.Horizontal)
        self.slider_volumen.setRange(0, 100)
        self.slider_volumen.setValue(volumen_actual)
        self.slider_volumen.valueChanged.connect(
            lambda v: self.lbl_volumen.setText(f"<br><b>Volumen del Sintetizador:</b> {v}%")
        )
        # Se agrega el slider de volumen al layout para que el usuario pueda ajustar el nivel de salida del sintetizador TTS en tiempo real.
        layout.addWidget(self.slider_volumen)

        # Botón de prueba
        self.btn_probar = QPushButton("🔊 Escuchar Muestra de Voz")
        self.btn_probar.setStyleSheet("margin-top: 15px; padding: 8px; font-weight: bold;")
        self.btn_probar.clicked.connect(self.probar_voz)
        if not self.voz_seleccionada_id:
            self.btn_probar.setEnabled(False)
        layout.addWidget(self.btn_probar)
        
        # Se agrega un espacio flexible al final para que los elementos se mantengan arriba y no se estiren innecesariamente.
        layout.addStretch()

    def probar_voz(self):
        """
        Envía una muestra al hilo TTS leyendo los valores en vivo de los sliders.
        """
        # Se obtiene la voz seleccionada y se verifica que exista antes de intentar reproducir la muestra.
        voz_id = getattr(self, "voz_seleccionada_id", None)
        if not voz_id:
            return

        # Se prepara un diccionario con la configuración de voz actual para pasarla al hilo TTS y reproducir la muestra.
        if self.parent_app and hasattr(self.parent_app, "hilo_tts"):
            config_prueba = {
                "voz_id": voz_id,
                "velocidad": self.slider_velocidad.value(),
                "volumen": self.slider_volumen.value() / 100.0
            }
            mensaje = "Prueba de voz, ajustando los parámetros del sistema LSM."
            self.parent_app.hilo_tts.hablar(mensaje, config_prueba)

    def guardar_cambios(self):
        """
        Guarda en memoria y en disco todos los ajustes elegidos por el usuario.
        """
        # Pestaña 1
        self.config["personalizacion"]["fuente"] = self.combo_fuentes.currentFont().family()
        self.config["personalizacion"]["tamano_letra"] = self.slider_tamano.value()
        # Guardar estados visuales
        self.config["personalizacion"]["mostrar_hud"] = self.chk_mostrar_hud.isChecked()
        self.config["personalizacion"]["mostrar_landmarks"] = self.chk_mostrar_landmarks.isChecked()

        # Pestaña 2
        if self.rb_estatico.isChecked():
            modo = "Forzar Estático"
        elif self.rb_dinamico.isChecked():
            modo = "Forzar Dinámico"
        else:
            modo = "Híbrido (Auto MoE)"

        self.config["traduccion"]["modo_ia"] = modo
        self.config["traduccion"]["ventana_frames"] = self.slider_fps.value()

        # Pestaña 3
        self.config["voces"]["activar_voz"] = self.chk_activar_voz.isChecked()
        self.config["voces"]["voz_id"] = getattr(self, "voz_seleccionada_id", None)
        self.config["voces"]["velocidad"] = self.slider_velocidad.value()
        self.config["voces"]["volumen"] = self.slider_volumen.value() / 100.0

        # Guardamos la configuración en disco para que los cambios persistan entre sesiones.
        ConfigManager.guardar_configuracion(self.config)
        self.accept()