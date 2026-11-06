"""
Motor de predicción LSM – lógica extraída del prototipo original.
Comité de Expertos (MoE): modelo estático (letras) + LSTM (palabras dinámicas).
"""

import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque


class PredictorLSM:
    """
    Encapsula MediaPipe Holistic + los dos modelos Keras
    y la lógica de histéresis / estabilización del prototipo.
    """

    # Clases del proyecto
    PALABRAS_LSTM = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
    LETRAS_ESTATICAS = ['A', 'B', 'C', 'L']

    # Índices de landmarks de rostro usados en el entrenamiento original
    PUNTOS_ROSTRO = [
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17,
        314, 405, 321, 375, 291, 409, 270, 269, 267,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324
    ]

    def __init__(self, ruta_modelos: str):
        self.ruta_modelos = ruta_modelos

        print("⏳ Cargando modelos Keras...")
        ruta_lstm = os.path.join(ruta_modelos, "modelo_lsm.keras")
        ruta_estatico = os.path.join(ruta_modelos, "modelo_lsm_estatico.keras")

        # Fallback a versiones con mayor precisión si existen
        if not os.path.exists(ruta_lstm):
            alt = os.path.join(ruta_modelos, "modelo_lsm_v2_98pct_12clases.keras")
            if os.path.exists(alt):
                ruta_lstm = alt
        if not os.path.exists(ruta_estatico):
            alt = os.path.join(ruta_modelos, "modelo_lsm_estatico_v1_97pct_4clases.keras")
            if os.path.exists(alt):
                ruta_estatico = alt

        self.modelo_lstm = load_model(ruta_lstm)
        self.modelo_estatico = load_model(ruta_estatico)
        print("✅ Modelos cargados correctamente.")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Estado interno (por sesión / conexión)
        self.reset()

    def reset(self):
        """Reinicia la memoria temporal de una sesión."""
        self.secuencia = []
        self.historial_predicciones = []
        self.oracion = ""
        self.historial_muneca_izq = []
        self.historial_muneca_der = []
        self.historial_estado_motor = []
        self.motor_activo = "Estático"
        self.frame_contador = 0
        self.ultima_confianza = 0.0
        self.ultima_prediccion_cruda = None

    def extraer_y_normalizar(self, resultados) -> np.ndarray:
        """
        Extrae y normaliza landmarks respecto al centro de los hombros.
        Debe ser idéntico al usado en el entrenamiento.
        """
        ref_x, ref_y, ref_z = 0.5, 0.5, 0.0
        escala = 1.0

        if resultados.pose_landmarks:
            hombro_izq = resultados.pose_landmarks.landmark[11]
            hombro_der = resultados.pose_landmarks.landmark[12]
            ref_x = (hombro_izq.x + hombro_der.x) / 2
            ref_y = (hombro_izq.y + hombro_der.y) / 2
            ref_z = (hombro_izq.z + hombro_der.z) / 2
            escala = np.sqrt(
                (hombro_izq.x - hombro_der.x) ** 2 +
                (hombro_izq.y - hombro_der.y) ** 2
            )
            if escala == 0:
                escala = 1.0

        # Pose (6 puntos)
        pose = []
        if resultados.pose_landmarks:
            for idx in [11, 12, 13, 14, 15, 16]:
                lm = resultados.pose_landmarks.landmark[idx]
                pose.extend([
                    (lm.x - ref_x) / escala,
                    (lm.y - ref_y) / escala,
                    (lm.z - ref_z) / escala
                ])
        else:
            pose = [0.0] * 18

        # Rostro
        face = []
        if resultados.face_landmarks:
            for idx in self.PUNTOS_ROSTRO:
                lm = resultados.face_landmarks.landmark[idx]
                face.extend([
                    (lm.x - ref_x) / escala,
                    (lm.y - ref_y) / escala,
                    (lm.z - ref_z) / escala
                ])
        else:
            face = [0.0] * (len(self.PUNTOS_ROSTRO) * 3)

        # Mano izquierda
        lh = []
        if resultados.left_hand_landmarks:
            for lm in resultados.left_hand_landmarks.landmark:
                lh.extend([
                    (lm.x - ref_x) / escala,
                    (lm.y - ref_y) / escala,
                    (lm.z - ref_z) / escala
                ])
        else:
            lh = [0.0] * 63

        # Mano derecha
        rh = []
        if resultados.right_hand_landmarks:
            for lm in resultados.right_hand_landmarks.landmark:
                rh.extend([
                    (lm.x - ref_x) / escala,
                    (lm.y - ref_y) / escala,
                    (lm.z - ref_z) / escala
                ])
        else:
            rh = [0.0] * 63

        return np.concatenate([pose, face, lh, rh])

    def _actualizar_motor(self, resultados) -> bool:
        """Decide si usar el modelo Estático o Dinámico según velocidad de la muñeca."""
        velocidad_maxima = 0.0
        mano_detectada = False

        if resultados.left_hand_landmarks:
            mano_detectada = True
            muneca = np.array([
                resultados.left_hand_landmarks.landmark[0].x,
                resultados.left_hand_landmarks.landmark[0].y
            ])
            self.historial_muneca_izq.append(muneca)
            self.historial_muneca_izq = self.historial_muneca_izq[-5:]
            if len(self.historial_muneca_izq) == 5:
                vel = np.linalg.norm(
                    self.historial_muneca_izq[-1] - self.historial_muneca_izq[0]
                )
                velocidad_maxima = max(velocidad_maxima, vel)
        else:
            self.historial_muneca_izq = []

        if resultados.right_hand_landmarks:
            mano_detectada = True
            muneca = np.array([
                resultados.right_hand_landmarks.landmark[0].x,
                resultados.right_hand_landmarks.landmark[0].y
            ])
            self.historial_muneca_der.append(muneca)
            self.historial_muneca_der = self.historial_muneca_der[-5:]
            if len(self.historial_muneca_der) == 5:
                vel = np.linalg.norm(
                    self.historial_muneca_der[-1] - self.historial_muneca_der[0]
                )
                velocidad_maxima = max(velocidad_maxima, vel)
        else:
            self.historial_muneca_der = []

        if mano_detectada:
            estado = 0 if velocidad_maxima < 0.018 else 1
            self.historial_estado_motor.append(estado)
            self.historial_estado_motor = self.historial_estado_motor[-10:]

            if len(self.historial_estado_motor) == 10:
                conteo = sum(self.historial_estado_motor)
                if conteo >= 4:
                    self.motor_activo = "Dinámico"
                elif conteo <= 1:
                    self.motor_activo = "Estático"

        return mano_detectada

    def procesar_frame(self, frame_bgr: np.ndarray, dibujar_landmarks: bool = True) -> dict:
        """
        Procesa un frame BGR y devuelve el resultado de la predicción.

        Returns
        -------
        dict con:
            prediccion, confianza, motor, oracion, mano_detectada, frame_anotado (opcional)
        """
        import cv2

        self.frame_contador += 1
        frame = cv2.flip(frame_bgr, 1)  # espejo natural

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = self.holistic.process(image_rgb)

        keypoints = self.extraer_y_normalizar(resultados)
        self.secuencia.append(keypoints)
        self.secuencia = self.secuencia[-30:]

        mano_detectada = self._actualizar_motor(resultados)
        confianza = 0.0
        prediccion_cruda = None

        # Inferencia cada 2 frames para aliviar CPU
        if self.frame_contador % 2 == 0 and mano_detectada:
            if self.motor_activo == "Estático":
                pred = self.modelo_estatico.predict(
                    np.expand_dims(keypoints, axis=0), verbose=0
                )[0]
                idx = int(np.argmax(pred))
                confianza = float(pred[idx])
                if confianza > 0.85:
                    prediccion_cruda = self.LETRAS_ESTATICAS[idx]
                    self.historial_predicciones.append(prediccion_cruda)

            elif self.motor_activo == "Dinámico" and len(self.secuencia) == 30:
                pred = self.modelo_lstm.predict(
                    np.expand_dims(np.array(self.secuencia), axis=0), verbose=0
                )[0]
                idx = int(np.argmax(pred))
                confianza = float(pred[idx])
                if confianza > 0.65:
                    prediccion_cruda = self.PALABRAS_LSTM[idx]
                    self.historial_predicciones.append(prediccion_cruda)

            # Estabilización: mayoría en los últimos 6 frames
            self.historial_predicciones = self.historial_predicciones[-6:]
            if len(self.historial_predicciones) == 6:
                mas_comun = max(set(self.historial_predicciones),
                                key=self.historial_predicciones.count)
                if (self.historial_predicciones.count(mas_comun) >= 4
                        and self.oracion != mas_comun):
                    self.oracion = mas_comun

            if not mano_detectada or confianza < 0.50:
                self.historial_predicciones = []

        self.ultima_confianza = confianza
        self.ultima_prediccion_cruda = prediccion_cruda

        # Dibujar landmarks (opcional)
        frame_out = frame.copy()
        if dibujar_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_out, resultados.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )
            self.mp_drawing.draw_landmarks(
                frame_out, resultados.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )
            self.mp_drawing.draw_landmarks(
                frame_out, resultados.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        return {
            "prediccion": prediccion_cruda or self.oracion or None,
            "confianza": round(confianza * 100, 1),
            "motor": self.motor_activo,
            "oracion": self.oracion,
            "mano_detectada": mano_detectada,
            "frame_anotado": frame_out,
        }
