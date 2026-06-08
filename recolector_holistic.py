import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# --- CONFIGURACIÓN ---
seña_a_grabar = "Dolor"
repeticiones_objetivo = 30    
frames_por_seña = 30 

ruta_base = "dataset_lsm_holistic"
os.makedirs(os.path.join(ruta_base, seña_a_grabar), exist_ok=True)

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extraer_y_normalizar(resultados):
    # (El código matemático se mantiene exactamente igual para proteger tu precisión)
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

print(f"--- MODO DE GRABACIÓN SEGURO ---")
print(f"Seña objetivo: '{seña_a_grabar}'")

repeticion_actual = 0

# Cambiamos for por while para controlar cuándo avanzamos
while repeticion_actual < repeticiones_objetivo:
    
    # 1. PANTALLA DE ESPERA
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Listo: {seña_a_grabar} ({repeticion_actual+1}/{repeticiones_objetivo})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "[r] Grabar  |  [q] Salir", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Recolector LSM', frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('r'):
            break # Rompe este bucle y pasa a grabar
        elif tecla == ord('q'):
            print("Cerrando el programa...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit() # Cierra todo el script

    secuencia_marcadore = []
    
    # 2. PROCESO DE GRABACIÓN (30 frames)
    for frame_num in range(frames_por_seña):
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = holistic.process(rgb_frame)

        if resultados.face_landmarks:
            mp_drawing.draw_landmarks(frame, resultados.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1))
        if resultados.pose_landmarks:
            mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if resultados.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if resultados.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        datos_frame = extraer_y_normalizar(resultados)
        secuencia_marcadore.append(datos_frame)
        
        cv2.putText(frame, f"GRABANDO... {frame_num+1}/30", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Recolector LSM', frame)
        cv2.waitKey(1)

    # 3. PANTALLA DE DECISIÓN (Guardar o Descartar)
    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, "REVISION:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "[g] Guardar  |  [d] Descartar error  |  [q] Salir", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Recolector LSM', frame)
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord('g'):
            # Guardamos y avanzamos el contador
            np.save(os.path.join(ruta_base, seña_a_grabar, f"{repeticion_actual}.npy"), secuencia_marcadore)
            print(f"Muestra {repeticion_actual+1} guardada.")
            repeticion_actual += 1 
            break 
            
        elif tecla == ord('d'):
            # No guardamos nada, el contador no avanza
            print(f"Muestra {repeticion_actual+1} descartada. Repitamos el movimiento.")
            break 
            
        elif tecla == ord('q'):
            print("Cerrando el programa...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

cap.release()
cv2.destroyAllWindows()
print(f"¡Recolección de la seña '{seña_a_grabar}' terminada exitosamente!")