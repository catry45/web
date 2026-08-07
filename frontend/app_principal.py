import sys
import cv2
import numpy as np
import mediapipe as mp
import os
import datetime
from tensorflow.keras.models import load_model

from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QFrame, QInputDialog, QMessageBox)
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from PyQt6.QtGui import QImage, QPixmap

# --- 1. Procesa la cámara y las predicciones fuera del hilo de la interfaz. ---
class HiloVideo(QThread):
    # Estas señales llevan cada frame y el estado de la predicción a la UI.
    cambio_pixmap_signal = pyqtSignal(np.ndarray)
    datos_ia_signal = pyqtSignal(str, float, str) # Envía: (Oración, Confianza, Motor activo)
    # El constructor inicializa variables y diccionarios de palabras.
    def __init__(self):
        super().__init__()
        self.corriendo = True
        
        # Diccionarios de palabras
        # Las clases deben coincidir con el orden usado al entrenar cada modelo.
        self.palabras_lstm = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
        self.letras_estaticas = ['A', 'B', 'C', 'L']
        
        # Estado de la secuencia actual y datos que se muestran en la interfaz.
        self.secuencia = []
        self.historial_predicciones = []
        self.oracion = "Esperando..."
        self.historial_muneca_izq = []
        self.historial_muneca_der = []
        self.historial_estado_motor = []
        self.motor_activo_actual = "Estático"
        self.ultimo_keypoints = np.zeros(1662) # Almacena el último frame procesado para correcciones
    # Extrae y normaliza los keypoints de pose, rostro y manos.
    def extraer_y_normalizar(self, resultados):
        # Tomamos el centro de los hombros como origen para reducir el efecto de la distancia a la cámara.
        ref_x, ref_y, ref_z = 0.5, 0.5, 0.0
        escala = 1.0
        
        if resultados.pose_landmarks:
            hombro_izq = resultados.pose_landmarks.landmark[11]
            hombro_der = resultados.pose_landmarks.landmark[12]
            ref_x = (hombro_izq.x + hombro_der.x) / 2
            ref_y = (hombro_izq.y + hombro_der.y) / 2
            ref_z = (hombro_izq.z + hombro_der.z) / 2
            escala = np.sqrt((hombro_izq.x - hombro_der.x)**2 + (hombro_izq.y - hombro_der.y)**2)
            if escala == 0: escala = 1.0

        pose = []
        if resultados.pose_landmarks:
            for idx in [11, 12, 13, 14, 15, 16]:
                lm = resultados.pose_landmarks.landmark[idx]
                pose.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            pose = [0.0] * (6 * 3)

        # Solo usamos puntos relevantes del rostro para mantener el vector manejable.
        puntos_rostro_clave = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        face = []
        if resultados.face_landmarks:
            for idx in puntos_rostro_clave:
                lm = resultados.face_landmarks.landmark[idx]
                face.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            face = [0.0] * (len(puntos_rostro_clave) * 3)

        lh = []
        if resultados.left_hand_landmarks:
            for lm in resultados.left_hand_landmarks.landmark:
                lh.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            lh = [0.0] * (21 * 3)

        rh = []
        if resultados.right_hand_landmarks:
            for lm in resultados.right_hand_landmarks.landmark:
                rh.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            rh = [0.0] * (21 * 3)

        return np.concatenate([pose, face, lh, rh])
    # Guarda la muestra actual para incorporarla en un próximo reentrenamiento.
    def guardar_muestra_correccion(self, palabra_correcta):
        directorio_frontend = os.path.dirname(os.path.abspath(__file__))
        directorio_raiz = os.path.abspath(os.path.join(directorio_frontend, ".."))

        # Determinamos si la seña es dinámica o estática según el diccionario.
        tipo_senia = "dinamica" if palabra_correcta in self.palabras_lstm else "estatica"
        ruta_carpeta = os.path.join(directorio_raiz, "historial_errores", tipo_senia, palabra_correcta)
        os.makedirs(ruta_carpeta, exist_ok=True)

        # Guardamos la muestra con un nombre único basado en la fecha y hora.
        fecha_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{tipo_senia}_{fecha_hora}.npy"
        ruta_guardado = os.path.join(ruta_carpeta, nombre_archivo)

        # Si es dinámica guardamos los 30 frames, si es estática solo el ultimo frame
        muestra = self.secuencia.copy() if tipo_senia == "dinamica" else self.ultimo_keypoints.copy()
        
        # Evitamos guardar secuencias incompletas que no servirían para el modelo.
        if tipo_senia == "dinamica" and len(muestra) < 30:
            return False # si es que faltan frames para guardar una muestra dinámica válida
        # Guardamos la muestra en formato numpy.
        np.save(ruta_guardado, np.array(muestra, dtype=np.float32))
        return True
    # El hilo principal de video se ejecuta aquí, cargando los modelos y procesando cada frame.
    def run(self):
        # Los modelos se cargan aquí para que TensorFlow viva en el mismo hilo que los procesa.
        directorio_frontend = os.path.dirname(os.path.abspath(__file__))
        directorio_raiz = os.path.abspath(os.path.join(directorio_frontend, ".."))
        ruta_lstm = os.path.join(directorio_raiz, "modelos", "modelo_lsm.keras")
        ruta_estatico = os.path.join(directorio_raiz, "modelos", "modelo_lsm_estatico.keras")
        
        try:
            modelo_lstm = load_model(ruta_lstm)
            modelo_estatico = load_model(ruta_estatico)
        except Exception as e:
            print(f"Error cargando modelos: {e}. Revisa las rutas.")
            return
        
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_contador = 0
        # Inicializamos el motor en "Estático" para que la IA comience con el modelo de letras.
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.corriendo:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                frame_contador += 1 

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                mp_drawing.draw_landmarks(image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                keypoints = self.extraer_y_normalizar(resultados)
                self.ultimo_keypoints = keypoints  # Se usa si el usuario corrige la última predicción con "c".
                self.secuencia.append(keypoints)
                self.secuencia = self.secuencia[-30:]

                # --- DIRECTOR DE ORQUESTA ( Medimos el movimiento reciente de las muñecas para elegir el modelo adecuado.) ---
                velocidad_maxima = 0.0
                mano_detectada = False
                # --- MEDICIÓN DE VELOCIDAD DE LA MANO IZQUIERDA ---
                if resultados.left_hand_landmarks:
                    mano_detectada = True
                    muneca_izq = np.array([resultados.left_hand_landmarks.landmark[0].x, resultados.left_hand_landmarks.landmark[0].y])
                    self.historial_muneca_izq.append(muneca_izq)
                    self.historial_muneca_izq = self.historial_muneca_izq[-5:]
                    if len(self.historial_muneca_izq) == 5:
                        vel = np.linalg.norm(self.historial_muneca_izq[-1] - self.historial_muneca_izq[0])
                        velocidad_maxima = max(velocidad_maxima, vel)
                else:
                    self.historial_muneca_izq = []
                # --- MEDICIÓN DE VELOCIDAD DE LA MANO DERECHA ---
                if resultados.right_hand_landmarks:
                    mano_detectada = True
                    muneca_der = np.array([resultados.right_hand_landmarks.landmark[0].x, resultados.right_hand_landmarks.landmark[0].y])
                    self.historial_muneca_der.append(muneca_der)
                    self.historial_muneca_der = self.historial_muneca_der[-5:]
                    if len(self.historial_muneca_der) == 5:
                        vel = np.linalg.norm(self.historial_muneca_der[-1] - self.historial_muneca_der[0])
                        velocidad_maxima = max(velocidad_maxima, vel)
                else:
                    self.historial_muneca_der = []

                # --- ESTABILIZACIÓN DEL MOTOR (Suavizamos la decisión del motor para que no cambie por un solo frame extraño.) ---
                if mano_detectada:
                    estado_instantaneo = 0 if velocidad_maxima < 0.018 else 1
                    self.historial_estado_motor.append(estado_instantaneo)
                    self.historial_estado_motor = self.historial_estado_motor[-10:]

                    if len(self.historial_estado_motor) == 10:
                        conteo_dinamico = sum(self.historial_estado_motor)
                        if conteo_dinamico >= 4:
                            self.motor_activo_actual = "Dinámico"
                        elif conteo_dinamico <= 1:
                            self.motor_activo_actual = "Estático"

                # --- EJECUCIÓN DEL MODELO (Ejecutamos el modelo estático o el dinámico según el movimiento detectado.) ---     
                confianza_actual = 0.0
                if frame_contador % 2 == 0 and mano_detectada:
                    if self.motor_activo_actual == "Estático":
                        pred_estatica = modelo_estatico.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                        idx = np.argmax(pred_estatica)
                        confianza_actual = pred_estatica[idx]
                        
                        if confianza_actual > 0.85:
                            self.historial_predicciones.append(self.letras_estaticas[idx])
                            
                    elif self.motor_activo_actual == "Dinámico" and len(self.secuencia) == 30:
                        pred_lstm = modelo_lstm.predict(np.expand_dims(self.secuencia, axis=0), verbose=0)[0]
                        idx = np.argmax(pred_lstm)
                        confianza_actual = pred_lstm[idx]
                        
                        if confianza_actual > 0.65:
                            self.historial_predicciones.append(self.palabras_lstm[idx])

                    # Exigimos varias predicciones iguales antes de actualizar la palabra mostrada.
                    self.historial_predicciones = self.historial_predicciones[-6:] 
                    if len(self.historial_predicciones) == 6:
                        palabra_mas_comun = max(set(self.historial_predicciones), key=self.historial_predicciones.count)
                        conteo = self.historial_predicciones.count(palabra_mas_comun)
                        if conteo >= 4 and self.oracion != palabra_mas_comun:
                            self.oracion = palabra_mas_comun 
                    
                    if not mano_detectada or confianza_actual < 0.50:
                        self.historial_predicciones = []

                # La UI solo recibe el frame anotado y el estado ya resumido.
                self.cambio_pixmap_signal.emit(image)
                self.datos_ia_signal.emit(self.oracion, confianza_actual, self.motor_activo_actual)
        # Cerramos la cámara al salir del hilo.        
        cap.release()
    # Detenemos el hilo de video de manera segura.
    def detener(self):
        self.corriendo = False
        self.wait()

# --- 2. VENTANA PRINCIPAL DE LA APLICACIÓN ---
class TraductorLSM(QMainWindow):
    # Constructor de la ventana principal.
    def __init__(self):
        super().__init__() # Inicializamos la ventana principal.
        self.setWindowTitle("Traductor LSM - Prototipo") # Título de la ventana.
        self.resize(1000, 750) # Tamaño inicial de la ventana.
        self.setStyleSheet("QMainWindow { background-color: #121212; }") # Fondo oscuro para resaltar el video y los textos.
        # Creamos un widget central y un layout vertical para organizar los elementos.
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget_central.setLayout(layout)

        # Encabezado de la ventana.
        self.lbl_titulo = QLabel("Traductor de Lengua de Señas")
        self.lbl_titulo.setStyleSheet("color: #FFFFFF; font-size: 28px; font-weight: bold; font-family: 'Segoe UI';")
        layout.addWidget(self.lbl_titulo, alignment=Qt.AlignmentFlag.AlignCenter)

        # Indicadores compactos de confianza, motor y arquitectura.
        hud_layout = QHBoxLayout()
        
        self.lbl_arquitectura = QLabel("Arquitectura: Híbrida (MoE)")
        self.lbl_arquitectura.setStyleSheet("color: #AAAAAA; font-size: 14px;")
        
        self.lbl_motor = QLabel("Motor: Iniciando...")
        self.lbl_motor.setStyleSheet("color: #00E676; font-size: 14px; font-weight: bold;")
        
        self.lbl_confianza = QLabel("Confianza IA: 0.0%")
        self.lbl_confianza.setStyleSheet("color: #FFD600; font-size: 14px; font-weight: bold;")
        # Agregamos los indicadores al layout horizontal y luego al layout principal.
        hud_layout.addWidget(self.lbl_arquitectura)
        hud_layout.addWidget(self.lbl_motor)
        hud_layout.addWidget(self.lbl_confianza)
        layout.addLayout(hud_layout)

        # Área donde se muestra el video con los puntos detectados.
        self.marco_camara = QFrame() # Creamos un marco para contener el video de la cámara.
        self.marco_camara.setFixedSize(660, 500) # Tamaño fijo del marco de la cámara.
        self.marco_camara.setStyleSheet("background-color: #1E1E1E; border: 2px solid #333333; border-radius: 20px;") # Estilo del marco de la cámara.
        marco_layout = QVBoxLayout() # Creamos un layout vertical para el marco de la cámara.
        self.marco_camara.setLayout(marco_layout) # Agregamos un layout vertical al marco de la cámara.

        # Creamos un QLabel para mostrar el video de la cámara y lo centramos.
        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        marco_layout.addWidget(self.lbl_video)
        layout.addWidget(self.marco_camara, alignment=Qt.AlignmentFlag.AlignCenter)

        # Resultado que se mantiene visible mientras llega la siguiente predicción.
        self.lbl_traduccion = QLabel("Esperando...")
        self.lbl_traduccion.setStyleSheet("color: #FFFFFF; font-size: 48px; font-weight: bold; padding-top: 10px;")
        layout.addWidget(self.lbl_traduccion, alignment=Qt.AlignmentFlag.AlignCenter)

        # Acceso rápido para guardar una corrección del usuario.
        self.lbl_instrucciones = QLabel('Presiona "c" en tu teclado para corregir una seña (Active Learning)')
        self.lbl_instrucciones.setStyleSheet("color: #777777; font-size: 12px; margin-top: 10px;")
        layout.addWidget(self.lbl_instrucciones, alignment=Qt.AlignmentFlag.AlignCenter)

        # Conectar Hilo de Video con la interfaz.
        self.hilo = HiloVideo()
        self.hilo.cambio_pixmap_signal.connect(self.actualizar_imagen)
        self.hilo.datos_ia_signal.connect(self.actualizar_hud)
        self.hilo.start()
    # Actualiza la imagen del video en la interfaz.
    def actualizar_imagen(self, frame_cv):
        # QImage necesita los datos en RGB, mientras OpenCV trabaja en BGR.
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        imagen_qt = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.lbl_video.setPixmap(QPixmap.fromImage(imagen_qt))

    # Actualiza los indicadores de confianza, motor y traducción en la interfaz.
    def actualizar_hud(self, oracion, confianza, motor):
        # Actualizamos todos los indicadores juntos para que reflejen el mismo frame.
        self.lbl_traduccion.setText(oracion)
        self.lbl_confianza.setText(f"Confianza IA: {confianza*100:.1f}%")
        color_motor = "#FFD600" if motor == "Estático" else "#00E676"
        self.lbl_motor.setStyleSheet(f"color: {color_motor}; font-size: 14px; font-weight: bold;")
        self.lbl_motor.setText(f"Motor Activo: IA {motor.upper()}")

    # Permite al usuario corregir la predicción de la IA presionando "c".
    def keyPressEvent(self, event):
        """Abre el formulario para corregir la última seña detectada."""
        if event.key() == Qt.Key.Key_C:
            todas_las_palabras = self.hilo.palabras_lstm + self.hilo.letras_estaticas
            
            # El diálogo evita salir de la aplicación para registrar una corrección.
            palabra, ok = QInputDialog.getItem(
                self, 
                "Modo Corrección", 
                "La IA falló. Selecciona la seña CORRECTA que estabas haciendo:", 
                todas_las_palabras, 
                0, False
            )
            # Guardamos la corrección solo si el usuario selecciona una palabra y confirma.
            if ok and palabra:
                exito = self.hilo.guardar_muestra_correccion(palabra)
                if exito:
                    QMessageBox.information(self, "Aprendizaje Activo", f"✅ Dato guardado para '{palabra}'.\nLa IA lo estudiará en el próximo reentrenamiento.")
                else:
                    QMessageBox.warning(self, "Error", "⚠️ No hubo suficientes frames capturados para guardar esta seña dinámica. Intenta de nuevo.")
    
    # Cerramos el hilo de video al cerrar la ventana principal.
    def closeEvent(self, event):
        self.hilo.detener()
        event.accept()
# --- 3. EJECUCIÓN DE LA APLICACIÓN ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = TraductorLSM()
    ventana.show()
    sys.exit(app.exec())