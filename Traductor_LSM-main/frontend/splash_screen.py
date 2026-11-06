from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class SplashScreen(QWidget):
    """
    Pantalla de carga inicial que se muestra mientras se preparan los motores de Inteligencia Artificial.
    """
    # Esta señal indica que la inicialización del sistema ya terminó y la siguiente ventana puede abrirse.
    terminado_signal = pyqtSignal()

    def __init__(self):
        """
        Inicializa la pantalla de carga con un diseño centrado y moderno.
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Configura la apariencia y el contenido de la pantalla de carga inicial.
        """
        # Se fija el tamaño y se elimina el marco para que la pantalla se vea como una intro limpia y moderna.
        self.setFixedSize(450, 300)
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # El contenido se centra vertical y horizontalmente para mantener una composición visual equilibrada.
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Se definen los textos principales que muestran el nombre del proyecto y el estado de carga.
        self.label_titulo = QLabel("MANOS QUE HABLAN (LSM)")
        self.label_titulo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.label_titulo.setStyleSheet("color: #FFFFFF;")
        self.label_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_estado = QLabel("Inicializando motores de Inteligencia Artificial...")
        self.label_estado.setFont(QFont("Segoe UI", 10))
        self.label_estado.setStyleSheet("color: #A0A0A0;")
        self.label_estado.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Este bloque crea el panel central con fondo oscuro y bordes redondeados para dar una apariencia más pulida.
        self.contenedor = QWidget()
        self.contenedor.setStyleSheet("""
            background-color: #1E1E2E;
            border-radius: 15px;
            border: 1px solid #45475A;
        """)
        container_layout = QVBoxLayout(self.contenedor)
        container_layout.addWidget(self.label_titulo)
        container_layout.addWidget(self.label_estado)

        # Se incorpora el panel principal dentro de la pantalla de carga.
        layout.addWidget(self.contenedor)
        self.setLayout(layout)

        # Se simula la preparación del sistema para dar tiempo a que los modelos y recursos se carguen correctamente.
        QTimer.singleShot(1500, self.finalizar_carga)

    def finalizar_carga(self):
        """
        Actualiza el texto de estado para indicar que la inicialización ya está casi completa.
        """
        self.label_estado.setText("¡Cerebros cargados con éxito! Abriendo sistema...")
        QTimer.singleShot(800, self.cerrar_splash)

    def cerrar_splash(self):
        """
        Cierra la pantalla inicial y avisa a la aplicación principal que ya puede continuar.
        """
        self.close()
        self.terminado_signal.emit()