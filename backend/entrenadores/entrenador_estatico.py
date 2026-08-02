import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# --- 1. CONFIGURACIÓN DINÁMICA ---
directorio_script = os.path.dirname(os.path.abspath(__file__))
ruta_base = os.path.abspath(os.path.join(directorio_script, '..', '..', 'dataset_lsm_holistic'))
ruta_modelo = os.path.abspath(os.path.join(directorio_script, '..', '..', 'modelos', 'modelo_lsm_estatico.keras')) # 

palabras_estaticas = ['A', 'B', 'C', 'L'] # SOLO entrenaremos las letras estáticas
print(f"Entrenando Especialista de Forma para: {palabras_estaticas}")

# --- 2. CARGA DINÁMICA Y "AUMENTO DE DATOS" ---
secuencias_estaticas, etiquetas = [], []
mapa_etiquetas = {etiqueta: num for num, etiqueta in enumerate(palabras_estaticas)}

print("Extrayendo fotogramas estáticos...")
for palabra in palabras_estaticas:
    ruta_palabra = os.path.join(ruta_base, palabra)
    
    if not os.path.exists(ruta_palabra):
        continue

    for nombre_archivo in os.listdir(ruta_palabra):
        if not nombre_archivo.endswith(".npy"):
            continue
            
        ruta_archivo = os.path.join(ruta_palabra, nombre_archivo)
        try:
            res = np.load(ruta_archivo)
            # Tomamos los fotogramas del 12 al 17 (6 fotogramas) para simular un "aumento de datos" y evitar sobreajuste
            for fotograma in res[12:18]:  # Ajusta el rango según tus necesidades
                secuencias_estaticas.append(fotograma)
                etiquetas.append(mapa_etiquetas[palabra])
        except Exception as e:
            print(f"❌ Error leyendo {ruta_archivo}: {e}")

X = np.array(secuencias_estaticas)
y = to_categorical(etiquetas).astype(int)

# Dividimos 90% estudio, 10% validación
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.1)

# --- 3. CONSTRUCCIÓN DEL CEREBRO (RED DENSA SIMPLE) ---
# Aquí SÍ usamos 'relu' porque no hay memoria temporal (LSTM), son simples "fotos" matemáticas.
modelo_estatico = Sequential()
modelo_estatico.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
modelo_estatico.add(BatchNormalization()) # Estabiliza los datos
modelo_estatico.add(Dropout(0.3)) # Obliga a no memorizar (sobreajuste)
modelo_estatico.add(Dense(64, activation='relu'))
modelo_estatico.add(BatchNormalization())
modelo_estatico.add(Dense(len(palabras_estaticas), activation='softmax'))

modelo_estatico.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. ENTRENAMIENTO CON FRENO ---
print("\n--- INICIANDO ENTRENAMIENTO ESTÁTICO ---")
parada_temprana = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
guardado_seguro = ModelCheckpoint(ruta_modelo, monitor='val_categorical_accuracy', save_best_only=True)

modelo_estatico.fit(
    X_entrenamiento, 
    y_entrenamiento, 
    epochs=150, 
    validation_data=(X_prueba, y_prueba),
    callbacks=[parada_temprana, guardado_seguro]
)

# --- 5. FIN ---
print(f"\n¡Especialista de Forma guardado exitosamente en '{ruta_modelo}'!")