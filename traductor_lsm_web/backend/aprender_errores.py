import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# --- CONFIGURACIÓN ---
directorio_backend = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.abspath(os.path.join(directorio_backend, ".."))

ruta_base = os.path.join(directorio_raiz, "dataset_lsm_holistic")
historial_dir = os.path.join(directorio_raiz, "historial_errores")

modelos = {
    "dinamico": {
        "path": os.path.join(directorio_raiz, "modelos", "modelo_lsm.keras"), 
        "clases": ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']
    },
    "estatico": {
        "path": os.path.join(directorio_raiz, "modelos", "modelo_lsm_estatico.keras"), 
        "clases": ['A', 'B', 'C', 'L']
    }
}

# --- FUNCIONES ---
def reentrenar_modelo(tipo):
    ruta_modelo = modelos[tipo]["path"]
    lista_clases = modelos[tipo]["clases"]
    
    print("\n" + "="*50)
    print(f"🧠 INICIANDO ACTIVE LEARNING: CEREBRO {tipo.upper()} 🧠")
    print("="*50)

    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encontró el modelo en '{ruta_modelo}'.")
        return
    # Cargar y compilar el modelo
    try:
        modelo = load_model(ruta_modelo)
        modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"✅ Modelo {tipo} cargado y compilado correctamente.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el modelo: {e}")
        return
    # Crear backup de seguridad del modelo antes de reentrenar
    if not os.path.exists(f"{ruta_modelo}.bak"):
        try:
            shutil.copy(ruta_modelo, f"{ruta_modelo}.bak")
            print(f"🛡️ Backup de seguridad creado exitosamente (.bak).")
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo crear el backup: {e}")
    # Crear un mapa de clases a índices
    mapa = {c: i for i, c in enumerate(lista_clases)} # Diccionario de mapeo clase -> índice
    X, y = [], [] # Datos y etiquetas para reentrenamiento
    archivos_a_fusionar = [] # Lista de tuplas (origen, destino) para mover archivos exitosos al dataset principal
    hay_correcciones_nuevas = False # Bandera para indicar si hay correcciones nuevas que procesar

    for clase in lista_clases:
        # 1. Cargar datos originales del dataset principal
        ruta_clase_base = os.path.join(ruta_base, clase)
        if os.path.exists(ruta_clase_base): # Verificar que la carpeta de la clase exista
            for arch in os.listdir(ruta_clase_base):  # Iterar sobre los archivos en la carpeta de la clase
                if not arch.endswith(".npy"): continue # Filtrar solo archivos .npy
                try: # Cargar el archivo .npy y verificar su forma según el tipo de modelo
                    dato = np.load(os.path.join(ruta_clase_base, arch))
                    if tipo == "dinamico" and dato.shape == (30, 264): # Verificar que la forma sea la esperada para el modelo dinámico
                        X.append(dato)
                        y.append(mapa[clase])
                    elif tipo == "estatico": # Verificar que la forma sea la esperada para el modelo estático
                        if dato.shape == (30, 264):
                            X.append(dato[-1])
                            y.append(mapa[clase])
                        elif len(dato.shape) == 1 and dato.shape[0] == 264:
                            X.append(dato)
                            y.append(mapa[clase])
                except: pass

        # 2. Cargar NUEVAS correcciones estructuradas desde historial_errores
        ruta_clase_historial = os.path.join(historial_dir, clase)
        if os.path.exists(ruta_clase_historial):
            for arch in os.listdir(ruta_clase_historial):
                if not arch.endswith(".npy"): continue
                
                # Filtrar para no cruzar datos estáticos en el modelo dinámico y viceversa
                if tipo == "dinamico" and "estatica" in arch: continue
                if tipo == "estatico" and "dinamica" in arch: continue
                # Cargar el archivo de corrección y verificar su forma
                ruta_completa_error = os.path.join(ruta_clase_historial, arch)
                try:
                    dato = np.load(ruta_completa_error)
                    if tipo == "dinamico" and dato.shape == (30, 264): # Verificar que la forma sea la esperada para el modelo dinámico
                        X.append(dato)
                        y.append(mapa[clase])
                        archivos_a_fusionar.append((ruta_completa_error, os.path.join(ruta_clase_base, arch)))
                        hay_correcciones_nuevas = True
                    elif tipo == "estatico" and len(dato.shape) == 1 and dato.shape[0] == 264: # Verificar que la forma sea la esperada para el modelo estático
                        X.append(dato)
                        y.append(mapa[clase])
                        archivos_a_fusionar.append((ruta_completa_error, os.path.join(ruta_clase_base, arch)))
                        hay_correcciones_nuevas = True
                except: pass # Ignorar errores de carga de archivos .npy

    # --- REENTRENAMIENTO Y FUSIÓN ---
    if hay_correcciones_nuevas and len(X) > 0:
        print(f"\n📊 Métrica: Entrenando con {len(X)} muestras totales ({len(archivos_a_fusionar)} Correcciones Nuevas).")
        X = np.array(X)
        y_cat = to_categorical(y, num_classes=len(lista_clases))
        # Evaluar el modelo antes de reentrenar
        try:
            print("⚙️ Inyectando nuevo conocimiento (Ajuste Fino)...")
            eval_pre = modelo.evaluate(X, y_cat, verbose=0)
            precision_pre = eval_pre[1] 
            # Reentrenar el modelo con las nuevas correcciones
            historial = modelo.fit(X, y_cat, epochs=5, verbose=0) 
            precision_post = historial.history['accuracy'][-1]
            
            print(f"📈 Análisis Post-Entrenamiento:")
            print(f"   - Precisión Base: {precision_pre*100:.2f}%")
            print(f"   - Precisión Nueva: {precision_post*100:.2f}%")
            # Comparar la precisión antes y después del reentrenamiento
            if precision_post >= (precision_pre - 0.02):
                modelo.save(ruta_modelo)
                print(f"✅ ¡Cerebro actualizado con éxito!")
                
                # FUSIÓN DE DATOS: Mover archivos exitosos al dataset principal
                for origen, destino in archivos_a_fusionar:
                    os.makedirs(os.path.dirname(destino), exist_ok=True)
                    shutil.move(origen, destino)
                print(f"🧹 {len(archivos_a_fusionar)} correcciones integradas al dataset permanente.")
            else:
                print(f"⚠️ ¡ALERTA! El reentrenamiento degradó el modelo. Se ha cancelado el guardado.")
        except Exception as e:
            print(f"❌ ERROR al reentrenar o evaluar: {e}")
    else:
        print(f"ℹ️ No hay correcciones compatibles nuevas para el cerebro {tipo}.")

# --- EJECUCIÓN ---
reentrenar_modelo("dinamico")
reentrenar_modelo("estatico")