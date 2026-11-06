import sys
import os

# Se agrega la ruta del proyecto al entorno para asegurar que las importaciones internas funcionen
# sin depender de la ubicación exacta desde la que se ejecute la app.
directorio_actual = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directorio_actual)

from PyQt6.QtWidgets import QApplication
from frontend.splash_screen import SplashScreen
from frontend.app_principal import TraductorLSM


def main():
    """
    Inicia la aplicación y coordina la carga inicial con la ventana principal.
    """
    app = QApplication(sys.argv)

    # Primero se muestra la pantalla de bienvenida para dar tiempo a que los recursos y modelos se preparen antes de abrir la interfaz principal.
    splash = SplashScreen()
    splash.show()

    ventana_principal = None

    def abrir_app_principal():
        """
        Crea y abre la ventana principal cuando la carga inicial ha terminado.
        """
        nonlocal ventana_principal
        # Cuando la pantalla de inicio termina, se crea la interfaz principal y se presenta al usuario.
        ventana_principal = TraductorLSM()
        ventana_principal.show()

    # La señal del splash marca el momento exacto en el que es seguro abrir la aplicación completa.
    splash.terminado_signal.connect(abrir_app_principal)

    sys.exit(app.exec())

# Se asegura de que la función main solo se ejecute si este archivo es el punto de entrada principal del programa.
if __name__ == "__main__":
    main()