#import para el funcionamiento correcto del script:
import os
import cv2
import numpy as np
import mediapipe as mp
import datetime
import threading
from PyQt6.QtCore import QThread, pyqtSignal
from tensorflow.keras.models import load_model # Se importa para cargar los modelos de IA entrenados para la traducción de señas.
from frontend.dialogs.configuracion import ConfigManager
from collections import deque

# --- Hilo principal para capturar el video y ejecutar la detección en segundo plano ---
class HiloVideo(QThread):
    """
    Hilo que captura video de la cámara, procesa los frames con MediaPipe y ejecuta los modelos de IA para traducir señas.
    Emite señales para actualizar la interfaz con la imagen procesada y los resultados de la predicción.
    """
    cambio_pixmap_signal = pyqtSignal(np.ndarray)
    datos_ia_signal = pyqtSignal(str, float, str) 

    def __init__(self):
        """
        Inicializa el hilo de video y establece los parámetros iniciales para la detección de señas.
        """
        super().__init__()
        self.corriendo = True
        # Aquí se definen las clases que el sistema puede reconocer, tanto palabras dinámicas como letras estáticas.
        # Esto ayuda a mantener la lógica de decisión más clara cuando se compara la salida del modelo con la intención del user.
        self.palabras_lstm = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
        self.letras_estaticas = ['A', 'B', 'C', 'L']

        # Guardamos una pequeña ventana temporal de datos para que la IA pueda decidir con más estabilidad y evitar cambios bruscos.
        self.secuencia = []
        self.historial_predicciones = []
        self.oracion = "Esperando..."
        self.historial_muneca_izq = []
        self.historial_muneca_der = []
        self.historial_estado_motor = []
        self.motor_activo_actual = "Estático"
        self.ultimo_keypoints = np.zeros(1662)
        self.config_lock = threading.Lock()
        config_inicial = ConfigManager.cargar_configuracion()
        self.modo_ia = config_inicial["traduccion"]["modo_ia"]
        self.ventana_frames = config_inicial["traduccion"]["ventana_frames"]
        self.mostrar_landmarks = config_inicial.get("personalizacion", {}).get("mostrar_landmarks", False)


    def actualizar_configuracion(self, configuracion):
        """
        Actualiza los parámetros del hilo sin romper la sincronización entre hilos.
        """
        with self.config_lock:
            self.modo_ia = configuracion["traduccion"]["modo_ia"]
            self.ventana_frames = configuracion["traduccion"]["ventana_frames"]
            self.mostrar_landmarks = configuracion.get("personalizacion", {}).get("mostrar_landmarks", False)


    def extraer_y_normalizar(self, resultados):
        """
        Extrae y normaliza los puntos clave de la imagen para su uso en el modelo de IA.
        Normaliza respecto al centro de los hombros para que sea invariante a la distancia de la cámara.
        """
        # Se definen valores de referencia para normalizar los puntos clave y asegurar que la posición de la persona en el frame no afecte la predicción.
        ref_x, ref_y, ref_z = 0.5, 0.5, 0.0
        escala = 1.0
        # Se calcula un punto de referencia basado en los hombros para normalizar la posición de la persona en el frame.
        if resultados.pose_landmarks:
            hombro_izq = resultados.pose_landmarks.landmark[11]
            hombro_der = resultados.pose_landmarks.landmark[12]
            ref_x = (hombro_izq.x + hombro_der.x) / 2
            ref_y = (hombro_izq.y + hombro_der.y) / 2
            ref_z = (hombro_izq.z + hombro_der.z) / 2
            escala = np.sqrt((hombro_izq.x - hombro_der.x)**2 + (hombro_izq.y - hombro_der.y)**2)
            if escala == 0: escala = 1.0

        pose = []  # Puntos clave del cuerpo, normalizados para que la postura no dependa tanto de la posición en la cámara.
        if resultados.pose_landmarks:
            for idx in [11, 12, 13, 14, 15, 16]:
                lm = resultados.pose_landmarks.landmark[idx]
                pose.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            pose = [0.0] * (6 * 3)

        puntos_rostro_clave = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        face = []  # Puntos del rostro para que la expresión y la orientación también influyan en la clasificación.
        if resultados.face_landmarks:
            for idx in puntos_rostro_clave:
                lm = resultados.face_landmarks.landmark[idx]
                face.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            face = [0.0] * (len(puntos_rostro_clave) * 3)

        lh = [] # Lista para almacenar las coordenadas normalizadas de la mano izquierda.
        if resultados.left_hand_landmarks:
            for lm in resultados.left_hand_landmarks.landmark:
                lh.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            lh = [0.0] * (21 * 3)

        rh = []  # Lista para almacenar las coordenadas normalizadas de la mano derecha.
        if resultados.right_hand_landmarks:
            for lm in resultados.right_hand_landmarks.landmark:
                rh.extend([(lm.x - ref_x) / escala, (lm.y - ref_y) / escala, (lm.z - ref_z) / escala])
        else:
            rh = [0.0] * (21 * 3)
        # Se coloca todo en un único vector para que cada frame tenga una entrada uniforme para el modelo.
        return np.concatenate([pose, face, lh, rh])

    def guardar_muestra_correccion(self, palabra_correcta):
        """
        Guarda la muestra de la seña actual para reentrenamiento del modelo (Active Learning).
        """
        # 1. Obtenemos la ruta actual (backend/hilos)
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        # 2. Subimos DOS niveles (hilos -> backend -> traductor_lsm)
        directorio_raiz = os.path.abspath(os.path.join(directorio_actual, "..", ".."))

        # Separamos palabras dinámicas de letras estáticas para guardar cada ejemplo en la carpeta correcta.
        tipo_senia = "dinamica" if palabra_correcta in self.palabras_lstm else "estatica"
        ruta_carpeta = os.path.join(directorio_raiz, "historial_errores", tipo_senia, palabra_correcta)
        os.makedirs(ruta_carpeta, exist_ok=True)

        # El nombre se basa en la hora para evitar sobrescribir archivos antiguos.
        fecha_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"{tipo_senia}_{fecha_hora}.npy"
        ruta_guardado = os.path.join(ruta_carpeta, nombre_archivo)

        # Las señas dinámicas guardan una secuencia completa; las estáticas guardan solo el último frame detectado.
        muestra = self.secuencia.copy() if tipo_senia == "dinamica" else self.ultimo_keypoints.copy()

        # Si la seña es dinámica, verificamos que tengamos suficientes frames para considerarla válida antes de guardarla.
        if tipo_senia == "dinamica" and len(muestra) < 30:
            return False 
        # Guardamos la muestra en formato NumPy para su posterior análisis o reentrenamiento del modelo.
        np.save(ruta_guardado, np.array(muestra, dtype=np.float32))
        return True


    def run(self):
        """
        Captura video, procesa frames con MediaPipe y ejecuta los modelos de IA en tiempo real.
        """
        # 1. Obtenemos la ruta actual (backend/hilos)
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Subimos DOS niveles para llegar a la carpeta raíz 
        directorio_raiz = os.path.abspath(os.path.join(directorio_actual, "..", ".."))
        
        # 3. Construimos las rutas absolutas seguras para encontrar los modelos
        ruta_lstm = os.path.join(directorio_raiz, "modelos", "modelo_lsm.keras")
        ruta_estatico = os.path.join(directorio_raiz, "modelos", "modelo_lsm_estatico.keras")

        # Si el modelo no carga correctamente, se corta la ejecución con un mensaje claro.
        try:
            modelo_lstm = load_model(ruta_lstm)
            modelo_estatico = load_model(ruta_estatico)
        except Exception as e:
            print(f"Error cargando modelos: {e}. Revisa las rutas.")
            return
        
        mp_holistic = mp.solutions.holistic  # MediaPipe para detectar pose, manos y rostro en una sola pasada.
        mp_drawing = mp.solutions.drawing_utils  # Permite dibujar los landmarks en el frame para visualizar el análisis en tiempo real.

        # Se abre la cámara con el backend de Windows para que la captura funcione de forma más estable en esta máquina.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_contador = 0

        # Configuramos el hilo para que procese frames continuamente mientras la ventana principal esté abierta y el hilo no haya sido detenido.
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.corriendo:
                ret, frame = cap.read()
                if not ret: break
                
                # Ajustamos el tamaño y la orientación para que la detección sea más uniforme y consistente.
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                frame_contador += 1 

                # MediaPipe trabaja mejor con imágenes RGB, así que convertimos el frame antes del análisis.
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultados = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extraemos la configuración del usuario
                with self.config_lock:
                    dibujar_puntos = self.mostrar_landmarks
                # Solo dibuja si el usuario lo solicita
                if dibujar_puntos:
                    mp_drawing.draw_landmarks(image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Guardamos el frame actual como entrada del modelo y también como historial temporal para detectar movimiento.
                keypoints = self.extraer_y_normalizar(resultados)
                self.ultimo_keypoints = keypoints 
                self.secuencia.append(keypoints)
                self.secuencia = self.secuencia[-30:]

                # --- FEEDBACK VISUAL SUTIL (EFECTO NEÓN DIFUMINADO) ---
                # 1. Detectar manos considerando el "Efecto Espejo" de cv2.flip
                mano_izq_pantalla = bool(resultados.right_hand_landmarks)
                mano_der_pantalla = bool(resultados.left_hand_landmarks)

                if mano_izq_pantalla or mano_der_pantalla:
                    # A. Lienzo Negro Puro (Optimización de memoria en lugar de copiar la imagen completa)
                    overlay = np.zeros_like(image)
                    alto, ancho, _ = image.shape
                    
                    # B. Grosor base ancho (ej. 45 px) para dar espacio al difuminado
                    grosor_base = 45 
                    
                    # C. Colores BGR más saturados para que resalten al difuminarse
                    color_izq = (255, 255, 0)   # Cyan intenso
                    color_der = (255, 0, 255)   # Magenta intenso
                    color_mixto = (255, 128, 128) # Púrpura luminoso
                    
                    # D. Dibujamos franjas sólidas gruesas en los bordes del lienzo negro
                    if mano_izq_pantalla and mano_der_pantalla:
                        # Marco completo
                        cv2.rectangle(overlay, (0, 0), (ancho, alto), color_mixto, grosor_base)
                    else:
                        if mano_izq_pantalla:
                            cv2.rectangle(overlay, (0, 0), (grosor_base, alto), color_izq, -1)
                        if mano_der_pantalla:
                            cv2.rectangle(overlay, (ancho - grosor_base, 0), (ancho, alto), color_der, -1)
                    
                    # E. Sub-sampling (Bloom Effect)
                    # 1. Achicamos el lienzo a 1/4 de su tamaño original.
                    ancho_peq, alto_peq = ancho // 4, alto // 4
                    overlay_peq = cv2.resize(overlay, (ancho_peq, alto_peq), interpolation=cv2.INTER_LINEAR)
                    
                    # 2. Desenfoque Gaussiano Severo (Blur)
                    # El kernel (17, 17) define qué tan lejos viaja y se difumina la luz hacia el centro. 
                    # Debe ser siempre un número impar. OpenCV lo procesa rapidísimo en C++ (es como funciona por debajo OpenCV).
                    overlay_blur_peq = cv2.GaussianBlur(overlay_peq, (17, 17), 0)
                    
                    # 3. Estiramos la imagen difuminada de regreso a su tamaño original (640x480).
                    # La interpolación lineal (INTER_LINEAR) crea un degradado ultra suave de forma muy rápida.
                    overlay_blur = cv2.resize(overlay_blur_peq, (ancho, alto), interpolation=cv2.INTER_LINEAR)
                    
                    # F. Fusión Aditiva (Pantalla)
                    # En lugar de hacer la imagen semitransparente, SUMAMOS la luz del desenfoque 
                    # a la imagen original. El centro (negro = 0) no altera el video, pero los bordes brillan.
                    # El 0.8 es la intensidad del brillo (80%).
                    image = cv2.addWeighted(image, 1.0, overlay_blur, 0.8, 0)

                # --- DIRECTOR DE ORQUESTA (se decide si la señal actual se interpreta como movimiento estático o dinámico)---
                velocidad_maxima = 0.0
                mano_detectada = False
                # Se calcula la velocidad de la muñeca izquierda y derecha para determinar si el usuario está haciendo un movimiento dinámico o una seña estática.
                if resultados.left_hand_landmarks:
                    mano_detectada = True
                    muneca_izq = np.array([resultados.left_hand_landmarks.landmark[0].x, resultados.left_hand_landmarks.landmark[0].y])
                    self.historial_muneca_izq.append(muneca_izq)
                    self.historial_muneca_izq = self.historial_muneca_izq[-5:]
                    if len(self.historial_muneca_izq) == 5:
                        vel = np.linalg.norm(self.historial_muneca_izq[-1] - self.historial_muneca_izq[0])
                        velocidad_maxima = max(velocidad_maxima, vel)
                else:
                    self.historial_muneca_izq = [] # Si no se detecta la mano izquierda, se reinicia el historial para evitar falsos positivos.

                # Se calcula la velocidad de la muñeca derecha y se compara con la izquierda para determinar el tipo de seña.
                if resultados.right_hand_landmarks:
                    mano_detectada = True
                    muneca_der = np.array([resultados.right_hand_landmarks.landmark[0].x, resultados.right_hand_landmarks.landmark[0].y])
                    self.historial_muneca_der.append(muneca_der)
                    self.historial_muneca_der = self.historial_muneca_der[-5:]
                    if len(self.historial_muneca_der) == 5:
                        vel = np.linalg.norm(self.historial_muneca_der[-1] - self.historial_muneca_der[0])
                        velocidad_maxima = max(velocidad_maxima, vel)
                else:
                    self.historial_muneca_der = [] # Si no se detecta la mano derecha, se reinicia el historial para evitar falsos positivos.

                # --- HISTÉRESIS ASIMÉTRICA ADAPTATIVA (Vinculada al Slider) ---
                # Si el usuario forzó un modo específico desde la configuración, ese valor tiene prioridad.
                with self.config_lock:
                    modo_ia = self.modo_ia
                    umbral_filtro = self.ventana_frames

                if mano_detectada:
                    # 1. Detectamos movimiento (0.02 es la sensibilidad a micro-movimientos)
                    estado_instantaneo = 1 if velocidad_maxima > 0.02 else 0
                    self.historial_estado_motor.append(estado_instantaneo)
                    
                    # 2. Recortamos la lista normal, sin borrarla.
                    # Hacemos que la ventana de tiempo dependa del slider del usuario (umbral_filtro)
                    # Limitamos a un máximo de 15 frames para evitar que el cambio de motor se sienta "laggeado" si el usuario pone 30.
                    ventana_histeresis = min(15, umbral_filtro) 
                    self.historial_estado_motor = self.historial_estado_motor[-ventana_histeresis:]

                    # 3. Lógica Asimétrica basada en porcentajes
                    if len(self.historial_estado_motor) == ventana_histeresis:
                        conteo_dinamico = sum(self.historial_estado_motor)
                        
                        # Matemáticas para arrancar (Rápido): Si el ~30% de los frames tienen movimiento, salta a Dinámico.
                        limite_arranque = max(2, int(ventana_histeresis * 0.30))
                        
                        # Matemáticas para frenar (Lento): Exigimos que haya 0 o máximo 1 frame de ruido para volver a Estático.
                        limite_freno = 1 
                        
                        if conteo_dinamico >= limite_arranque: 
                            self.motor_activo_actual = "Dinámico"
                        elif conteo_dinamico <= limite_freno: 
                            self.motor_activo_actual = "Estático"

                # Override Manual de Motor desde Configuraciones
                motor_a_usar = self.motor_activo_actual
                if modo_ia == "Forzar Estático":
                    motor_a_usar = "Estático"
                elif modo_ia == "Forzar Dinámico":
                    motor_a_usar = "Dinámico"

                # --- EJECUCIÓN DEL MODELO SEGÚN EL TIPO DE SEÑA DETECTADA ---
                confianza_actual = 0.0
                if frame_contador % 2 == 0 and mano_detectada:  # Procesar cada dos frames, reduce carga sin perder mucha fluidez.

                    if motor_a_usar == "Estático":  # Para letras o poses fijas se usa el modelo estático.
                        pred_estatica = modelo_estatico.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                        idx = np.argmax(pred_estatica)
                        confianza_actual = pred_estatica[idx]
                        if confianza_actual > 0.85: # Si la confianza es suficientemente alta, se agrega la predicción al historial para su posterior análisis y filtrado (Estático).
                            self.historial_predicciones.append(self.letras_estaticas[idx])

                    elif motor_a_usar == "Dinámico" and len(self.secuencia) == 30:  # Para señas de movimiento necesita una secuencia completa de 30 fps.
                        pred_lstm = modelo_lstm.predict(np.expand_dims(self.secuencia, axis=0), verbose=0)[0]
                        idx = np.argmax(pred_lstm)
                        confianza_actual = pred_lstm[idx]
                        # Si la confianza es suficientemente alta, se agrega la predicción al historial para su posterior análisis y filtrado (Dinámico).
                        if confianza_actual > 0.65:
                            self.historial_predicciones.append(self.palabras_lstm[idx])

                    # Se revisa la ventana de predicciones para reducir ruido y evitar decisiones súbitas.
                    self.historial_predicciones = self.historial_predicciones[-umbral_filtro:] 
                    if len(self.historial_predicciones) == umbral_filtro:
                        palabra_mas_comun = max(set(self.historial_predicciones), key=self.historial_predicciones.count)
                        conteo = self.historial_predicciones.count(palabra_mas_comun)
                        
                        # Se exige una mayoría suficiente para aceptar una traducción y evitar falsos positivos.
                        mayoria_requerida = max(1, int(umbral_filtro * 0.65))
                        if conteo >= mayoria_requerida and self.oracion != palabra_mas_comun:
                            self.oracion = palabra_mas_comun 

                    # Si no se detecta la mano o la confianza es muy baja, se limpia el historial de predicciones.
                    if not mano_detectada or confianza_actual < 0.50:
                        self.historial_predicciones = []

                # Enviamos el frame y la predicción al hilo principal de la interfaz.
                self.cambio_pixmap_signal.emit(image)
                self.datos_ia_signal.emit(self.oracion, confianza_actual, motor_a_usar)
                
        # Cuando el hilo termina, se libera la cámara para que otros programas puedan usarla y se evita que quede bloqueada.
        cap.release()

    def detener(self):
        """
        Detiene el hilo de video y libera recursos.
        """
        self.corriendo = False
        self.wait()