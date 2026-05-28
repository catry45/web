import cv2
import mediapipe as mp

# 1. Inicializar los módulos de MediaPipe HOLISTIC
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# (Opcional) Cargar estilos de dibujo para que el prototipo se vea más profesional
mp_drawing_styles = mp.solutions.drawing_styles

# Configurar el reconocedor integral (Cuerpo, Rostro y Manos)
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
print("Iniciando cámara Holistic... Presiona 'q' para cerrar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: No se puede acceder a la cámara web.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el fotograma con el modelo Holistic
    resultados = holistic.process(rgb_frame)

    # --- 3. DIBUJAR LOS COMPONENTES DETECTADOS ---

    # A) Dibujar la Malla Facial (Gestos)
    if resultados.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            resultados.face_landmarks, 
            mp_holistic.FACEMESH_TESSELATION, # Conecta los puntos como una red
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )

    # B) Dibujar la Postura del Cuerpo (Hombros, codos, posición)
    if resultados.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            resultados.pose_landmarks, 
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # C) Dibujar la Mano Izquierda
    if resultados.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            resultados.left_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

    # D) Dibujar la Mano Derecha
    if resultados.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            resultados.right_hand_landmarks, 
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    # 4. Mostrar la ventana
    cv2.imshow("Prototipo LSM - Holistic", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()