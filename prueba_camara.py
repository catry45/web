import cv2
import mediapipe as mp

# 1. Inicializar los módulos de MediaPipe para manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar el reconocedor de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. Inicializar la captura de la cámara web (0 suele ser la cámara integrada)
cap = cv2.VideoCapture(0)

print("Iniciando cámara... Presiona la tecla 'q' para cerrar la aplicación.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: No se puede acceder a la cámara web.")
        break

    # Efecto espejo para que sea más natural interactuar con la pantalla
    frame = cv2.flip(frame, 1)
    
    # MediaPipe requiere imágenes en formato RGB, pero OpenCV captura en BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el fotograma para buscar los puntos de las manos
    resultados = hands.process(rgb_frame)

    # 3. Si se detecta alguna mano, dibujar sus articulaciones
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    # 4. Mostrar el resultado en una ventana interactiva
    cv2.imshow("Prueba de Concepto - Detección LSM", frame)

    # Romper el ciclo si el usuario presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Limpieza de recursos al cerrar
cap.release()
cv2.destroyAllWindows()