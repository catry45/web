import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- 1. CARGAR EL CEREBRO ---
# Cargamos los archivos binarios que fueron entrenados
modelo = load_model('modelos/modelo_lsm.keras')

# 12 palabras en ESTRICTO ORDEN ALFABÉTICO (así las aprendió la IA)
palabras = ['A', 'Ayudar', 'B', 'Bien', 'C', 'Comer', 'Dolor', 'Gracias', 'Hola', 'L', 'Mal', 'Por_favor']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 2. EXTRACCIÓN ---
def extraer_y_normalizar(resultados):
    # (El código matemático se mantiene exactamente igual para proteger la precisión)
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

# --- 3. MEMORIA A CORTO PLAZO ---
secuencia = [] # Aquí se guardan los últimos 30 fotogramas que la IA usará para entender qué palabra estás haciendo 
historial_predicciones = [] # Guarda las últimas 15 predicciones para evitar que la IA se vuelva loca con un movimiento raro
oracion = "" # Aquí se guardará la palabra final que se muestra en pantalla (ej. "Hola", "Gracias", etc.)

# --- 4. ENCENDER LA CÁMARA ---
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Leer la imagen con MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Dibujar los puntos en pantalla para que te veas
        mp_drawing.draw_landmarks(image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extraer las coordenadas matemáticas
        keypoints = extraer_y_normalizar(resultados)
        secuencia.append(keypoints)

        # Efecto "Ventana Deslizante": Solo conservamos los últimos 30 fotogramas
        secuencia = secuencia[-30:]

        # --- LA MAGIA DE LA PREDICCIÓN ---
        # Si ya tenemos 30 fotogramas acumulados, le preguntamos al modelo
        # --- LA MAGIA DE LA PREDICCIÓN CON FILTRO ---
        if len(secuencia) == 30:
            prediccion = modelo.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
            indice_ganador = np.argmax(prediccion)
            
            # Guardamos esta predicción en el historial
            historial_predicciones.append(indice_ganador)
            
            # Mantenemos el historial siempre en 15 predicciones máximo
            historial_predicciones = historial_predicciones[-15:]
            
            # FILTRO: ¿La predicción actual es la más común en el historial?
            # Contamos cuántas veces ha salido esta palabra en los últimos 15 intentos
            conteo_ganador = historial_predicciones.count(indice_ganador)
            
            # Si la IA está más del 70% segura Y esta palabra se ha repetido al menos 10 veces en el historial
            if prediccion[indice_ganador] > 0.70 and conteo_ganador > 10:
                nueva_palabra = palabras[indice_ganador]
                
                # Solo actualizamos si es diferente para evitar parpadeos
                if oracion != nueva_palabra:
                    oracion = nueva_palabra
        # Mostrar el resultado final en la pantalla
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Traduccion: {oracion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Traductor LSM en Vivo', image)
        
        # Presiona 'q' para salir
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()