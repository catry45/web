import os
import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QFileDialog, QMessageBox, QLabel
)
# Importamos herramientas para generar PDFs con formato A4 estándar y estilo de texto.
from PyQt6.QtGui import QFont, QPdfWriter, QTextDocument, QPageSize
from PyQt6.QtCore import pyqtSignal

class VentanaHistorialExport(QDialog):
    """
    Ventana para revisar y guardar el historial de traducciones de la sesión.
    Permite exportar a TXT y PDF usando funcionalidades nativas de PyQt6.
    """

    # Se declara la señal como un atributo de clase, es para enviar una señal
    # de los cambios a "app_principal" y se muestre en la barra de estado
    mensaje_estado = pyqtSignal(str)

    def __init__(self, historial_frases, parent=None):
        super().__init__(parent)
        # Guardamos la lista de traducciones que se han realizado en esta sesión.
        self.historial = historial_frases
        self.setWindowTitle("📜 Historial y Exportación de Traducción LSM")
        self.resize(550, 450)
        self.setup_ui()

    def setup_ui(self):
        """
        Arma la interfaz visual: título, área de texto con el historial y botones de acción.
        """
        layout = QVBoxLayout(self)

        # Título principal de la ventana.
        lbl_titulo = QLabel("Registro de Traducciones de la Sesión")
        lbl_titulo.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(lbl_titulo)

        # Área de texto donde se muestra el historial (no se puede editar en GUI para proteger los datos).
        self.txt_historial = QTextEdit()
        self.txt_historial.setReadOnly(True)
        self.refrescar_vista_texto()
        layout.addWidget(self.txt_historial)

        # Organizamos los botones en una fila horizontal.
        btn_layout = QHBoxLayout()

        # Botón para descargar el historial como archivo de texto plano.
        btn_exportar_txt = QPushButton("📄 Exportar TXT")
        btn_exportar_txt.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")
        btn_exportar_txt.clicked.connect(self.exportar_txt)
        btn_layout.addWidget(btn_exportar_txt)

        # Botón para exportar una versión más formal en formato PDF.
        btn_exportar_pdf = QPushButton("📕 Exportar PDF")
        btn_exportar_pdf.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 6px;")
        btn_exportar_pdf.clicked.connect(self.exportar_pdf)
        btn_layout.addWidget(btn_exportar_pdf)

        # Botón para limpiar el historial (con confirmación para evitar errores).
        btn_limpiar = QPushButton("🗑️ Limpiar")
        btn_limpiar.setStyleSheet("background-color: #f57c00; color: white; font-weight: bold; padding: 6px;")
        btn_limpiar.clicked.connect(self.limpiar_historial)
        btn_layout.addWidget(btn_limpiar)

        # Cierra la ventana sin afectar el historial en memoria.
        btn_cerrar = QPushButton("Cerrar")
        btn_cerrar.clicked.connect(self.accept)
        btn_layout.addWidget(btn_cerrar)

        layout.addLayout(btn_layout)

    def refrescar_vista_texto(self):
        """
        Actualiza lo que se ve en pantalla con el contenido del historial.
        """
        if self.historial:
            # Numeramos cada frase para que el usuario vea el orden de las traducciones.
            lineas = [f"{idx + 1}. {frase}" for idx, frase in enumerate(self.historial)]
            self.txt_historial.setText("\n".join(lineas))
        else:
            # Si no hay traducciones, mostramos un mensaje indicándolo.
            self.txt_historial.setText("No hay traducciones registradas en esta sesión.")

    def exportar_txt(self):
        """
        Descarga el historial en formato texto (.txt) con una marca de tiempo.
        """
        if not self.historial:
            QMessageBox.information(self, "Aviso", "El historial está vacío. No hay datos para exportar.")
            return

        # Generamos un nombre único con la fecha y hora para no sobrescribir archivos anteriores.
        nombre_defecto = f"traduccion_lsm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        ruta_archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar Historial (TXT)", nombre_defecto, "Archivos de Texto (*.txt);;Todos los Archivos (*)"
        )

        if ruta_archivo:
            try:
                # Escribimos el archivo con un encabezado profesional que incluye proyecto y fecha.
                with open(ruta_archivo, "w", encoding="utf-8") as f:
                    f.write("====================================================\n")
                    f.write("               PROTOTIPO TRADUCTOR LSM              \n")
                    f.write(f"      Fecha: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
                    f.write("====================================================\n\n")
                    for idx, frase in enumerate(self.historial, 1):
                        f.write(f"[{idx}] {frase}\n")

                QMessageBox.information(self, "Éxito", f"TXT exportado correctamente:\n{os.path.basename(ruta_archivo)}")
                # Enviamos el nombre del archivo a la barra de estado ---
                nombre_archivo = os.path.basename(ruta_archivo)
                self.mensaje_estado.emit(f"📄 Historial exportado a TXT: {nombre_archivo}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo TXT:\n{str(e)}")

    def exportar_pdf(self):
        """
        Crea un PDF profesional con el historial en formato A4 estándar.
        """
        if not self.historial:
            QMessageBox.information(self, "Aviso", "El historial está vacío. No hay datos para exportar.")
            return

        # El nombre se crea automáticamente para diferenciar cada exportación.
        nombre_defecto = f"traduccion_lsm_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        ruta_archivo, _ = QFileDialog.getSaveFileName(
            self, "Guardar Documento (PDF)", nombre_defecto, "Archivos PDF (*.pdf);;Todos los Archivos (*)"
        )

        if ruta_archivo:
            try:
                # Configuramos el generador de PDF con tamaño A4 y resolución estándar.
                writer = QPdfWriter(ruta_archivo)
                writer.setCreator("Prototipo Traductor LSM - CBTIS 212")
                writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
                writer.setResolution(96)  # 96 DPI es la resolución estándar para pantalla e impresión.

                # Creamos un documento que se formatea con HTML para tener un PDF bien diseñado.
                documento = QTextDocument()
                html_content = f"""
                <html>
                <head>
                    <style>  /* Estilo general del documento para mantener consistencia visual. */
                        body {{ 
                            font-family: Arial, sans-serif;
                            font-size: 14pt;
                            color: #212121;
                            margin: 20px;
                        }}
                        h1 {{ /* Título principal con color distintivo y centrado para resaltar la identidad del proyecto. */
                            color: #1565C0;
                            text-align: center;
                            font-size: 22pt;
                            margin-bottom: 5px;
                        }}
                        h3 {{ /* Subtítulo con color neutro y centrado para proporcionar contexto. */
                            color: #616161;
                            text-align: center;
                            font-size: 14pt;
                            margin-top: 0px;
                            font-weight: normal;
                        }}
                        .fecha {{ /* Información de fecha con estilo discreto para no distraer del contenido principal. */
                            font-size: 11pt;
                            color: #424242;
                            margin-bottom: 15px;
                        }}
                        hr {{ /* Línea horizontal para separar secciones del documento, con un color gris claro. */
                            border: none;
                            border-top: 1px solid #BDBDBD;
                            margin: 15px 0;
                        }}
                        ul {{ /* Lista de frases con espaciado adecuado para mejorar la legibilidad. */
                            font-size: 13pt;
                            line-height: 1.8;
                            padding-left: 20px;
                        }}
                        li {{ /* Cada elemento de la lista tiene un margen inferior para evitar que se vean amontonados. */
                            margin-bottom: 8px;
                        }}
                        .footer {{ /* Pie de página con información de generación automática, centrado y con un color gris tenue. */
                            text-align: center;
                            font-size: 10pt;
                            color: #9E9E9E;
                            margin-top: 30px;
                        }}
                    </style>
                </head>
                <body>
                    <h1>Historial de Traducción LSM</h1>
                    <h3>Prototipo de Inclusión Social</h3>
                    <div class="fecha"><b>Fecha de Exportación:</b> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div>
                    <hr>
                    <ul>
                """

                # Cada frase del historial se coloca como una entrada de la lista del documento.
                for idx, frase in enumerate(self.historial, 1):
                    html_content += f"<li><b>[{idx}]</b> {frase}</li>"

                # Se cierra la lista y se agrega un pie de página indicando que el documento fue generado automáticamente.
                html_content += """
                    </ul>
                    <hr>
                    <div class="footer">
                        Documento generado automáticamente por el Motor de Inteligencia Artificial LSM.
                    </div>
                </body>
                </html>
                """

                # El contenido HTML final se renderiza directamente sobre el escritor PDF.
                documento.setHtml(html_content)
                documento.print(writer)

                # Se muestra un mensaje de éxito al usuario.
                QMessageBox.information(self, "Éxito", f"PDF exportado correctamente en formato A4:\n{os.path.basename(ruta_archivo)}")
                # Enviamos el aviso a la barra de estado ---
                nombre_archivo = os.path.basename(ruta_archivo)
                self.mensaje_estado.emit(f"📕 Historial exportado a PDF: {nombre_archivo}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo PDF:\n{str(e)}")

    def limpiar_historial(self):
        """
        Elimina el contenido actual tras pedir confirmación al usuario.
        """
        if not self.historial:
            return

        # Se confirma la acción para prevenir eliminación accidental del historial de la sesión.
        confirmacion = QMessageBox.question(
            self, "Confirmar Limpieza", "¿Estás seguro de que deseas borrar el historial de la sesión actual?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        # Si el usuario confirma, se limpia la lista y se actualiza la vista de texto para reflejar el cambio.
        if confirmacion == QMessageBox.StandardButton.Yes:
            self.historial.clear()
            self.refrescar_vista_texto()
            # Avisamos a la app principal que se limpió ---
            self.mensaje_estado.emit("🗑️ Historial de la sesión borrado correctamente.")