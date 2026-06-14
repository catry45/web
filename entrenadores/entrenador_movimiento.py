import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. CONFIGURACIÓN ---
# En tu script de la LSTM (ej. entrenador.py o entrenador_lstm.py):
ruta_base = "../dataset_lsm_holistic"
modelo.save('../modelos/modelo_lsm.keras')
# Leemos automáticamente el nombre de tus carpetas (A, Ayudar, B, Bien, etc.)
palabras = np.array(sorted(os.listdir(ruta_base)))
print(f"Palabras detectadas ({len(palabras)}): {palabras}")

# --- 2. CARGA DE DATOS ---
secuencias, etiquetas = [], []

# Mapeamos cada palabra a un número de ID (ej. A=0, Ayudar=1, etc.)
mapa_etiquetas = {etiqueta: num for num, etiqueta in enumerate(palabras)}

print("Cargando matrices en la memoria...")
for palabra in palabras:
    ruta_palabra = os.path.join(ruta_base, palabra)
    # Leemos todos los archivos dentro de la carpeta de la palabra
    for nombre_archivo in os.listdir(ruta_palabra):
        if nombre_archivo.endswith(".npy"): # Nos aseguramos de que sea un numpy array
            ruta_archivo = os.path.join(ruta_palabra, nombre_archivo)
            res = np.load(ruta_archivo)
            secuencias.append(res)
            etiquetas.append(mapa_etiquetas[palabra])

# Convertimos a arreglos matemáticos puros
X = np.array(secuencias)
y = to_categorical(etiquetas).astype(int)

# Dividimos los datos: 90% para que estudie, 10% para examen sorpresa (validación)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.1)

# --- 3. CONSTRUCCIÓN DEL CEREBRO (RED LSTM) ---
# X.shape[1] = 30 fotogramas | X.shape[2] = 264 coordenadas normalizadas
modelo = Sequential()
modelo.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
modelo.add(LSTM(128, return_sequences=True, activation='relu'))
modelo.add(LSTM(64, return_sequences=False, activation='relu'))
modelo.add(Dense(64, activation='relu'))
modelo.add(Dropout(0.2)) # Apagamos algunas neuronas al azar para que no memorice, sino que entienda
modelo.add(Dense(len(palabras), activation='softmax')) # Capa final de decisión

modelo.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. ENTRENAMIENTO ---
print("\n--- INICIANDO ENTRENAMIENTO DE LA IA ---")
# Epochs = las veces que repasará todos tus videos completos
modelo.fit(X_entrenamiento, y_entrenamiento, epochs=150, validation_data=(X_prueba, y_prueba))

# --- 5. EXPORTAR EL MODELO ---
modelo.summary()
modelo.save('modelo_lsm.keras')
print("\n¡Entrenamiento terminado! Cerebro matemático guardado como 'modelo_lsm.keras'")