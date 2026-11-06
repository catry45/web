# 🤟 Traductor de Lenguaje de Señas en Tiempo Real

Prototipo de software diseñado para la traducción e inclusión social mediante Inteligencia Artificial y Visión Computacional. Desarrollado para el Concurso de Prototipos del CBTIS 212.

## 🚀 Características del Proyecto
* **Procesamiento Eficiente:** Extracción y normalización matemática de articulaciones mediante *landmarks* de MediaPipe Holistic.
* **Arquitectura Robusta:** Implementación de redes neuronales recurrentes (LSTM) con `EarlyStopping` para prevenir el sobreajuste.
* **Bajo Costo y Escalabilidad:** Funciona en hardware convencional utilizando cámaras web estándar, procesando inferencias a 30 FPS.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.11+
* **Visión Artificial:** OpenCV & Google MediaPipe Holistic
* **Inteligencia Artificial:** TensorFlow & Keras (Modelos LSTM y Dense)
* **Control de Versiones:** Git & GitHub

## 📁 Estado del Prototipo
Actualmente en **Fase 2 (Entrenamiento e Inferencia Dinámica)**.
- [x] Configuración del entorno y control de versiones.
- [x] Desarrollo del motor de captura holística y normalización de datos.
- [x] Recolección de dataset colaborativo (120 muestras minimas por clase).
- [x] Entrenamiento de modelos de alta precisión (LSTM 98.15% y Estático).
- [x] Traducción en tiempo real mediante ventana deslizante.
- [x] Implementación de "Comité de Expertos" (IA Estática + IA Dinámica con histéresis).
- [x] Desarrollo de la Interfaz Gráfica de Usuario (GUI con PyQt6).