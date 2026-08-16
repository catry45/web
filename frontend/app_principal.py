import cv2
import datetime
import logging

# PyQt6 para la interfaz gráfica y la interacción con el usuario.
from PyQt6.QtWidgets import (
    QLabel, QMainWindow, QVBoxLayout,
    QHBoxLayout, QWidget, QFrame, QInputDialog, QMessageBox,
    QPushButton
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

# Importamos el administrador de configuración, ventanas y motores (hillos)
from backend.config_manager import ConfigManager
from frontend.dialogs.configuracion import VentanaConfiguracion
from frontend.dialogs.historial_export import VentanaHistorialExport
from backend.hilos.hilo_video import HiloVideo
from backend.hilos.hilo_tts import HiloTTS

# --- CONFIGURACIÓN DEL LOGGING ---
# Define cómo se verán los mensajes en la terminal y el nivel mínimo a mostrar.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- VENTANA PRINCIPAL DE LA APLICACIÓN Y CONEXIÓN CON LOS HILOS ---
class TraductorLSM(QMainWindow):
    """
    Ventana principal de la aplicación que muestra la cámara, la traducción en tiempo real
    y permite acceder a la configuración y al historial de traducciones.
    """
    def __init__(self):
        """
        Inicializa la ventana principal, configura la interfaz de usuario y arranca los hilos de procesamiento.
        """
        super().__init__() 
        self.setWindowTitle("Traductor LSM - Prototipo") 
        self.resize(1000, 750) 
        self.setStyleSheet("QMainWindow { background-color: #1e1e2e; }") 
        
        logging.info("Iniciando la interfaz gráfica (Frontend)...")
        
        # Carga la configuración inicial
        self.configuracion = ConfigManager.cargar_configuracion()

        # --- CONSTRUCCIÓN DE LA UI ---
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget_central.setLayout(layout)

        # Encabezado
        header_layout = QHBoxLayout()
        self.lbl_titulo = QLabel("Traductor de Lengua de Señas")
        self.lbl_titulo.setStyleSheet("color: #cdd6f4; font-size: 28px; font-weight: bold; font-family: 'Segoe UI';")
        header_layout.addWidget(self.lbl_titulo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Botón Ajustes
        self.btn_config = QPushButton("⚙️ Ajustes")
        self.btn_config.setStyleSheet("""
            QPushButton {
                background-color: #313244; color: #cdd6f4; font-size: 14px; 
                border-radius: 8px; padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #89b4fa; color: #11111b; }
        """)
        self.btn_config.clicked.connect(self.abrir_configuraciones)
        layout.addLayout(header_layout)

        # HUD
        hud_layout = QHBoxLayout()
        self.lbl_arquitectura = QLabel("Arquitectura: Híbrida (MoE)")
        self.lbl_arquitectura.setStyleSheet("color: #a6adc8; font-size: 14px;")
        self.lbl_motor = QLabel("Motor: Iniciando...")
        self.lbl_motor.setStyleSheet("color: #a6e3a1; font-size: 14px; font-weight: bold;")
        self.lbl_confianza = QLabel("Confianza IA: 0.0%")
        self.lbl_confianza.setStyleSheet("color: #f9e2af; font-size: 14px; font-weight: bold;")
        
        hud_layout.addWidget(self.lbl_arquitectura)
        hud_layout.addWidget(self.lbl_motor)
        hud_layout.addWidget(self.lbl_confianza)
        layout.addLayout(hud_layout)

        # Marco de Video
        self.marco_camara = QFrame() 
        self.marco_camara.setFixedSize(660, 500) 
        self.marco_camara.setStyleSheet("background-color: #11111b; border: 2px solid #45475a; border-radius: 15px;") 
        marco_layout = QVBoxLayout() 
        self.marco_camara.setLayout(marco_layout) 
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        marco_layout.addWidget(self.lbl_video)
        layout.addWidget(self.marco_camara, alignment=Qt.AlignmentFlag.AlignCenter)

        # Traducción actual
        self.lbl_traduccion = QLabel("Esperando...")
        self.aplicar_fuente_traduccion()
        layout.addWidget(self.lbl_traduccion, alignment=Qt.AlignmentFlag.AlignCenter)

        # Instrucciones
        self.lbl_instrucciones = QLabel('Presiona "C" en el teclado para el Modo Corrección (Active Learning)')
        self.lbl_instrucciones.setStyleSheet("color: #7f849c; font-size: 12px; margin-top: 5px;")
        layout.addWidget(self.lbl_instrucciones, alignment=Qt.AlignmentFlag.AlignCenter)

        # Footer (Historial y Ajustes)
        footer_layout = QHBoxLayout()
        self.btn_historial = QPushButton("📜 Historial")
        self.btn_historial.setStyleSheet("""
            QPushButton {
                background-color: #313244; color: #cdd6f4; font-size: 14px; 
                border-radius: 8px; padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #a6e3a1; color: #11111b; }
        """)
        self.btn_historial.clicked.connect(self.abrir_historial)
        footer_layout.addWidget(self.btn_config, alignment=Qt.AlignmentFlag.AlignLeft)
        footer_layout.addWidget(self.btn_historial, alignment=Qt.AlignmentFlag.AlignLeft)
        footer_layout.addStretch() 
        layout.addLayout(footer_layout)

        # Propiedades de estado
        self.ultima_oracion_hablada = "Esperando..."
        self.historial_traducciones = []

        # --- INICIO DE MOTORES (HILOS) ---
        logging.info("Arrancando hilo de síntesis de voz (TTS)...")
        self.hilo_tts = HiloTTS()
        self.hilo_tts.start()

        logging.info("Arrancando hilo de Visión Computacional (IA)...")
        self.hilo = HiloVideo()
        self.hilo.cambio_pixmap_signal.connect(self.actualizar_imagen)
        self.hilo.datos_ia_signal.connect(self.actualizar_hud)
        self.hilo.start()

    def aplicar_fuente_traduccion(self):
        """
        Aplica la fuente y tamaño de letra configurados a la etiqueta de traducción.
        """
        fuente_nombre = self.configuracion["personalizacion"]["fuente"]
        tamano = self.configuracion["personalizacion"]["tamano_letra"]
        self.lbl_traduccion.setFont(QFont(fuente_nombre, tamano, QFont.Weight.Bold))
        self.lbl_traduccion.setStyleSheet("color: #89b4fa; padding-top: 10px;")

    def abrir_historial(self):
        """
        Abre la ventana de historial de traducciones, permitiendo al usuario revisar y exportar las traducciones previas.
        """
        logging.info("Abriendo ventana de Historial de Sesión.")
        dialogo = VentanaHistorialExport(self.historial_traducciones, self)
        dialogo.exec()

    def abrir_configuraciones(self):
        """
        Abre la ventana de configuración, permitiendo al usuario ajustar parámetros de la aplicación.
        """
        logging.info("Abriendo panel de Ajustes.")
        dialogo = VentanaConfiguracion(self)
        if dialogo.exec():
            logging.info("Nuevos ajustes guardados. Aplicando cambios a los hilos en vivo...")
            self.configuracion = ConfigManager.cargar_configuracion()
            self.aplicar_fuente_traduccion()
            self.hilo.actualizar_configuracion(self.configuracion)

    def actualizar_imagen(self, frame_cv):
        """
        Actualiza la imagen de la cámara en la interfaz principal.
        """
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        imagen_qt = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(imagen_qt))

    def actualizar_hud(self, oracion, confianza, motor):
        """
        Actualiza la interfaz de usuario con la traducción actual, el nivel de confianza y el motor activo.
        También maneja la lógica de historial y síntesis de voz (TTS).
        """
        self.lbl_traduccion.setText(oracion)
        self.lbl_confianza.setText(f"Confianza IA: {confianza*100:.1f}%")
        color_motor = "#f9e2af" if motor == "Estático" else "#a6e3a1"
        self.lbl_motor.setStyleSheet(f"color: {color_motor}; font-size: 14px; font-weight: bold;")
        self.lbl_motor.setText(f"Motor Activo: IA {motor.upper()}")

        modo_actual = self.configuracion["traduccion"]["modo_ia"]
        self.lbl_arquitectura.setText(f"Arquitectura: {modo_actual}")

        # Historial y lógica de TTS
        if oracion != "Esperando..." and oracion != self.ultima_oracion_hablada:
            self.ultima_oracion_hablada = oracion
            hora_actual = datetime.datetime.now().strftime("%H:%M:%S")
            log_texto = f"[{hora_actual}] IA: {oracion}"
            
            logging.info(f"Nueva traducción detectada: {oracion} (Confianza: {confianza*100:.1f}%)")
            self.historial_traducciones.append(log_texto)

            voces_cfg = self.configuracion.get("voces", {})
            if voces_cfg.get("activar_voz", True) and hasattr(self, "hilo_tts"):
                self.hilo_tts.hablar(oracion, voces_cfg)

    def keyPressEvent(self, event):
        """
        Detecta la pulsación de teclas y activa el Modo Corrección (Active Learning) si se presiona 'C'.
        """
        if event.key() == Qt.Key.Key_C:
            logging.info("Usuario solicitó Modo Corrección (Active Learning).")
            todas_las_palabras = self.hilo.palabras_lstm + self.hilo.letras_estaticas
            palabra, ok = QInputDialog.getItem(
                self, "Modo Corrección", 
                "Selecciona la seña CORRECTA que estabas haciendo:", 
                todas_las_palabras, 0, False
            )
            # Si el usuario confirma la selección y proporciona una palabra válida, se intenta guardar la muestra de corrección.
            if ok and palabra:
                exito = self.hilo.guardar_muestra_correccion(palabra)
                if exito:
                    logging.info(f"Muestra de corrección guardada exitosamente para la clase: {palabra}")
                    QMessageBox.information(self, "Aprendizaje Activo", f"✅ Dato guardado para '{palabra}'.")
                else:
                    logging.warning(f"Fallo al guardar muestra de corrección para: {palabra} (frames insuficientes).")
                    QMessageBox.warning(self, "Error", "⚠️ Faltan frames para guardar una seña dinámica.")

    def closeEvent(self, event):
        """
        Maneja el evento de cierre de la ventana principal.
        """
        logging.info("Secuencia de apagado iniciada. Deteniendo motores...")
        self.hilo.detener()
        if hasattr(self, "hilo_tts"):
            self.hilo_tts.detener()
        logging.info("Apagado completado de forma segura.")
        event.accept()