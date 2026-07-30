import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model

# --- 1. CARGAR LOS CEREBROS (COMITÉ DE EXPERTOS) ---
print("Cargando IAs en la memoria...")
modelo_lstm = load_model('modelos/modelo_lsm.keras')
modelo_estatico = load_model('modelos/modelo_lsm_estatico.keras')

palabras_lstm = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
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
historial_muneca_izq = [] 
historial_muneca_der = [] 

historial_estado_motor = []  
motor_activo_actual = "Estático" 

# --- 4. ENCENDER LA CÁMARA ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR CRÍTICO: No se detectó ninguna cámara web conectada.")
    exit()
else:
    print("✅ Cámara conectada y lista.")
    print("✅ Sistema de traducción operando. Presiona 'c' para corregir un error o 'q' para salir.")

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

        # --- EL DIRECTOR DE ORQUESTA ---
        velocidad_maxima = 0.0
        mano_detectada = False
        vel_izq = 0.0
        vel_der = 0.0
        
        if resultados.left_hand_landmarks:
            mano_detectada = True
            muneca_izq = np.array([resultados.left_hand_landmarks.landmark[0].x, resultados.left_hand_landmarks.landmark[0].y])
            historial_muneca_izq.append(muneca_izq)
            historial_muneca_izq = historial_muneca_izq[-5:]
            if len(historial_muneca_izq) == 5:
                vel_izq = np.linalg.norm(historial_muneca_izq[-1] - historial_muneca_izq[0])
                if vel_izq > velocidad_maxima: velocidad_maxima = vel_izq
        else:
            historial_muneca_izq = []

        if resultados.right_hand_landmarks:
            mano_detectada = True
            muneca_der = np.array([resultados.right_hand_landmarks.landmark[0].x, resultados.right_hand_landmarks.landmark[0].y])
            historial_muneca_der.append(muneca_der)
            historial_muneca_der = historial_muneca_der[-5:]
            if len(historial_muneca_der) == 5:
                vel_der = np.linalg.norm(historial_muneca_der[-1] - historial_muneca_der[0])
                if vel_der > velocidad_maxima: velocidad_maxima = vel_der
        else:
            historial_muneca_der = []

        # --- ESTABILIZACIÓN DEL MOTOR ---
        if mano_detectada:
            estado_instantaneo = 0 if velocidad_maxima < 0.018 else 1
            historial_estado_motor.append(estado_instantaneo)
            historial_estado_motor = historial_estado_motor[-10:]

            if len(historial_estado_motor) == 10:
                conteo_dinamico = sum(historial_estado_motor)
                if conteo_dinamico >= 4:
                    motor_activo_actual = "Dinámico"
                elif conteo_dinamico <= 1:
                    motor_activo_actual = "Estático"

        # --- FUSIÓN DE IAs + OPTIMIZACIÓN ---
        confianza_actual = 0.0
        
        if frame_contador % 2 == 0 and mano_detectada:
            if motor_activo_actual == "Estático":
                pred_estatica = modelo_estatico.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                idx_estatico = np.argmax(pred_estatica)
                confianza_actual = pred_estatica[idx_estatico]
                
                if confianza_actual > 0.85:
                    nueva_palabra = letras_estaticas[idx_estatico]
                    historial_predicciones.append(nueva_palabra)
                    
            elif motor_activo_actual == "Dinámico" and len(secuencia) == 30:
                pred_lstm = modelo_lstm.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
                idx_lstm = np.argmax(pred_lstm)
                confianza_actual = pred_lstm[idx_lstm]
                
                if confianza_actual > 0.65:
                    nueva_palabra = palabras_lstm[idx_lstm]
                    historial_predicciones.append(nueva_palabra)

            # --- FILTRO DE ESTABILIZACIÓN DE TEXTO ---
            historial_predicciones = historial_predicciones[-6:] 
            
            # Solo evaluamos si la memoria está llena
            if len(historial_predicciones) == 6:
                palabra_mas_comun = max(set(historial_predicciones), key=historial_predicciones.count)
                conteo = historial_predicciones.count(palabra_mas_comun)
                
                # La IA debe predecir la misma palabra en 4 de los últimos 6 frames.
                # Esto asegura que el movimiento maduró y no es una transición a medias.
                if conteo >= 4 and oracion != palabra_mas_comun:
                    oracion = palabra_mas_comun # Actualizamos la oración traducida
            
            # Limpiador de basura: Si no hay mano o la IA duda mucho, borramos el historial
            if not mano_detectada or confianza_actual < 0.50:
                historial_predicciones = []

        # --- HUD ---
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, f"Traduccion: {oracion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "Arquitectura: Hibrida (MoE)", (410, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, f"Confianza IA: {confianza_actual*100:.1f}%", (410, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        if mano_detectada:
            color_motor = (0, 255, 255) if motor_activo_actual == "Estático" else (0, 255, 0)
            cv2.putText(image, f"Motor Activo: IA {motor_activo_actual.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_motor, 2)

        cv2.imshow('Traductor LSM en Vivo', image)
        
        tecla = cv2.waitKey(10) & 0xFF
        
        # --- MODO DE CORRECCIÓN ---
        if tecla == ord('c'):
            print("\n" + "="*40)
            print("🛑 MODO CORRECCIÓN ACTIVADO")
            print("="*40)
            print(f"La IA tradujo erróneamente: '{oracion}'")
            palabra_correcta = input("Escribe la palabra CORRECTA (respetando mayúsculas/guiones): ")
            
            if palabra_correcta in palabras_lstm or palabra_correcta in letras_estaticas:
                # --- LÓGICA DE CONFIRMACIÓN ---
                confirmacion = input(f"¿Estás seguro de guardar esta muestra como '{palabra_correcta}'? (s/n): ").strip().lower()
                # Si el usuario confirma, guardamos la secuencia en la carpeta correspondiente
                if confirmacion == 's':
                    ruta_carpeta = f"dataset_lsm_holistic/{palabra_correcta}"
                    os.makedirs(ruta_carpeta, exist_ok=True)
                    # Contar cuántos archivos de corrección ya existen para esa clase
                    num_archivos = len([name for name in os.listdir(ruta_carpeta) if name.startswith("correccion_")])
                    ruta_guardado = f"{ruta_carpeta}/correccion_{num_archivos}.npy"
                    # Guardar la secuencia en un archivo .npy
                    np.save(ruta_guardado, np.array(secuencia)) 
                    print(f"✅ ¡Dato guardado en {ruta_guardado}! La IA lo estudiará en el próximo reentrenamiento.")
                else:
                    print("🗑️ Corrección descartada por el usuario.")
            else:
                print("❌ Palabra no reconocida en el diccionario. Corrección cancelada.")
            print("Reanudando cámara...\n")
        
        elif tecla == ord('q'):
            break
# --- 5. CERRAR CÁMARA ---
cap.release()
cv2.destroyAllWindows()