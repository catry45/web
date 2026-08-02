import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- 1. CONFIGURACIÓN ---
# Esto obtiene la ruta de 'backend/entrenadores/'
directorio_script = os.path.dirname(os.path.abspath(__file__))

# Sube DOS niveles: sale de 'entrenadores' -> sale de 'backend' -> llega a la raíz
directorio_raiz = os.path.abspath(os.path.join(directorio_script, '..', '..'))

ruta_base = os.path.join(directorio_raiz, 'dataset_lsm_holistic')
ruta_modelo = os.path.join(directorio_raiz, 'modelos', 'modelo_lsm.keras')

print(f"Buscando el dataset en: {ruta_base}")

# Verificamos si existe antes de seguir para que el error sea claro
if not os.path.exists(ruta_base):
    print(f"¡ERROR! No se encontró la carpeta en: {ruta_base}")
    print("Asegúrate de que la carpeta se llame exactamente así y esté en la ruta correcta.")
    exit()

# --- 2. CARGA DE DATOS ---
secuencias, etiquetas = [], []

# Definir explícitamente las carpetas que corresponden a palabras de movimiento.
palabras = ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']

# Filtrar solo las carpetas que realmente existen en el dataset.
palabras = [p for p in palabras if os.path.isdir(os.path.join(ruta_base, p))]

if not palabras: # Si no hay carpetas válidas, lanzamos un error claro
    raise FileNotFoundError("No se encontraron carpetas de movimiento válidas en el dataset.")

print(f"Entrenando solo con clases de movimiento: {palabras}")

# Mapeamos cada palabra a un número de ID (ej. Ayudar=0, Bien=1, etc.)
mapa_etiquetas = {etiqueta: num for num, etiqueta in enumerate(palabras)}

print("Cargando matrices en la memoria...")
for palabra in palabras:
    ruta_palabra = os.path.join(ruta_base, palabra)
    for nombre_archivo in os.listdir(ruta_palabra):
        # 1. Filtro estricto: Solo archivos .npy
        if not nombre_archivo.endswith(".npy"):
            continue 
            
        ruta_archivo = os.path.join(ruta_palabra, nombre_archivo)
        
        try:
            window = np.load(ruta_archivo)
            secuencias.append(window)
            etiquetas.append(mapa_etiquetas[palabra])
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en el archivo: {ruta_archivo}")
            print(f"Detalle del error: {e}")

# Convertimos a arreglos matemáticos puros
X = np.array(secuencias)
y = to_categorical(etiquetas).astype(int)

# Dividimos los datos: 90% para que estudie, 10% para examen (validación)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.1)

# --- 3. CONSTRUCCIÓN DEL CEREBRO (RED LSTM ROBUSTA) ---
modelo = Sequential()
modelo.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2])))
modelo.add(BatchNormalization()) # ESTABILIZADOR 1

modelo.add(LSTM(128, return_sequences=True, activation='tanh'))
modelo.add(BatchNormalization()) # ESTABILIZADOR 2

modelo.add(LSTM(64, return_sequences=False, activation='tanh'))
modelo.add(Dense(64, activation='relu')) # Capa oculta
modelo.add(Dropout(0.2)) # Apagamos neuronas para evitar memorización (sobreajuste)
modelo.add(Dense(len(palabras), activation='softmax')) # Capa de decisión final

modelo.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. ENTRENAMIENTO Y CALLBACKS ---
print("\n--- INICIANDO ENTRENAMIENTO DE LA IA ---")

# Freno de seguridad
parada_temprana = EarlyStopping(
    monitor='val_categorical_accuracy', # Monitorizamos precisión en lugar de pérdida
    patience=15,
    restore_best_weights=True
)

# Seguro de guardado automático de la mejor versión
checkpoint = ModelCheckpoint(
    ruta_modelo, 
    monitor='val_categorical_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max' # Buscamos la precisión máxima
)

# Entrenamos con los callbacks activos
modelo.fit(
    X_entrenamiento, # Arreglos de entrenamiento
    y_entrenamiento, # Etiquetas de entrenamiento
    epochs=150, # Número de vueltas de entrenamiento
    validation_data=(X_prueba, y_prueba), # Validación para medir desempeño
    verbose=1, # Mostrar progreso en consola
    callbacks=[parada_temprana, checkpoint] # Freno de seguridad y guardado automático
)

# --- 5. RESUMEN FINAL ---
modelo.summary()
print(f"\n¡Entrenamiento terminado! El mejor Cerebro matemático se ha guardado de forma segura.")