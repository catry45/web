import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- 1. CARGAR LOS CEREBROS ---
print("Cargando IAs en la memoria...")
modelo_lstm = load_model('modelos/modelo_lsm.keras')
modelo_estatico = load_model('modelos/modelo_lsm_estatico.keras')

# EL LSTM SOLO CONOCE 8 PALABRAS (Dinámicas)
palabras_lstm = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
# El Estático conoce estas 4 (Estáticas)
letras_estaticas = ['A', 'B', 'C', 'L']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- 2. EXTRACCIÓN ---
def extraer_y_normalizar(resultados):
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

# --- 3. MEMORIA Y OPTIMIZACIÓN ---
secuencia = [] 
historial_predicciones = [] 
oracion = "" 
frame_contador = 0 

# --- 4. ENCENDER LA CÁMARA ---
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame_contador += 1 

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extraer_y_normalizar(resultados)
        secuencia.append(keypoints)
        secuencia = secuencia[-30:]

        # --- FUSIÓN DE IAs + OPTIMIZACIÓN DE RENDIMIENTO ---
        # Evaluamos cada 2 fotogramas (súper rápido y sin colapsar el procesador)
        if frame_contador % 2 == 0:
            
            # 1. EVALUACIÓN ESTÁTICA
            pred_estatica = modelo_estatico.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            idx_estatico = np.argmax(pred_estatica)
            
            # Exigimos 90% de seguridad para evitar falsos positivos
            if pred_estatica[idx_estatico] > 0.90:
                nueva_palabra = letras_estaticas[idx_estatico]
                historial_predicciones.append(nueva_palabra)
                
            # 2. EVALUACIÓN DINÁMICA
            elif len(secuencia) == 30:
                pred_lstm = modelo_lstm.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
                idx_lstm = np.argmax(pred_lstm)
                
                # Exigimos 65% para movimientos
                if pred_lstm[idx_lstm] > 0.65:
                    nueva_palabra = palabras_lstm[idx_lstm]
                    historial_predicciones.append(nueva_palabra)

            # --- FILTRO DE ESTABILIZACIÓN RÁPIDA ---
            historial_predicciones = historial_predicciones[-8:] # Solo los últimos 8 intentos
            if len(historial_predicciones) > 0:
                palabra_mas_comun = max(set(historial_predicciones), key=historial_predicciones.count)
                conteo = historial_predicciones.count(palabra_mas_comun)
                
                # Solo pedimos 4 coincidencias rápidas para mostrar la palabra en pantalla
                if conteo >= 4 and oracion != palabra_mas_comun:
                    oracion = palabra_mas_comun

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Traduccion: {oracion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Traductor LSM en Vivo', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()