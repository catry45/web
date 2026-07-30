import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# --- CONFIGURACIÓN ---
ruta_base = "dataset_lsm_holistic"
historial_dir = "historial_errores"
modelos = {
    "dinamico": {"path": "modelos/modelo_lsm.keras", "clases": ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']},
    "estatico": {"path": "modelos/modelo_lsm_estatico.keras", "clases": ['A', 'B', 'C', 'L']}
}
# Crear la carpeta de historial si no existe
os.makedirs(historial_dir, exist_ok=True)
# --- FUNCIONES ---
def reentrenar_modelo(tipo):
    ruta_modelo = modelos[tipo]["path"]
    lista_clases = modelos[tipo]["clases"]
    
    print("\n" + "="*50)
    print(f"🧠 INICIANDO ACTIVE LEARNING: CEREBRO {tipo.upper()} 🧠")
    print("="*50)
    #--- Verificación de existencia del modelo ---
    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encontró el modelo en '{ruta_modelo}'.")
        return
    #-- CARGA DEL MODELO ---
    try:
        modelo = load_model(ruta_modelo)
        
        # --- Re-compilar el modelo para asegurar métricas estandarizadas ---
        modelo.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        print(f"✅ Modelo {tipo} cargado y compilado correctamente.")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el modelo: {e}")
        return
    #--- CREACIÓN DE BACKUP ---
    if not os.path.exists(f"{ruta_modelo}.bak"):
        try:
            shutil.copy(ruta_modelo, f"{ruta_modelo}.bak")
            print(f"🛡️ Backup de seguridad creado exitosamente (.bak).")
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo crear el backup: {e}")
    #-- Preparación de datos ---
    mapa = {c: i for i, c in enumerate(lista_clases)}# Mapeo de clases a índices
    X, y = [], [] # Listas para almacenar datos y etiquetas
    archivos_correccion_procesados = [] # Lista para almacenar archivos de corrección procesados
    hay_correcciones_nuevas = False # Bandera para indicar si hay correcciones nuevas
    # --- CARGA DE DATOS ---
    for clase in lista_clases: # Recorrer cada clase
        ruta_carpeta = os.path.join(ruta_base, clase) # Ruta de la carpeta de la clase
        if not os.path.exists(ruta_carpeta): continue # Si la carpeta no existe, saltar a la siguiente clase
        for arch in os.listdir(ruta_carpeta): # Recorrer cada archivo en la carpeta
            if not arch.endswith(".npy"): continue # Si el archivo no es .npy, saltar al siguiente
            # Construir la ruta completa del archivo    
            ruta_completa = os.path.join(ruta_carpeta, arch)
            es_correccion = arch.startswith("correccion_")
            if es_correccion: hay_correcciones_nuevas = True
                # Si es una corrección, se procesará y luego se moverá a la carpeta de historial
            try:
                dato = np.load(ruta_completa)
                if tipo == "dinamico":# Para el modelo dinámico, se espera un array de 30 frames con 264 características cada uno
                    if dato.shape == (30, 264):
                        X.append(dato)
                        y.append(mapa[clase])
                elif tipo == "estatico": # Para el modelo estático, se espera un array de 264 características (un solo frame)
                    if dato.shape == (30, 264):
                        X.append(dato[-1]) 
                        y.append(mapa[clase])
                    elif len(dato.shape) == 1 and dato.shape[0] == 264:
                        X.append(dato)
                        y.append(mapa[clase])
                # Si el archivo es una corrección, agregarlo a la lista de procesados para moverlo después
                if es_correccion:
                    archivos_correccion_procesados.append((ruta_completa, arch)) # Guardar la ruta y el nombre del archivo para moverlo después 
            except Exception as e: # Si hay un error al cargar el archivo, mostrar un mensaje y continuar con el siguiente
                print(f"❌ ERROR: Archivo corrupto ignorado ({arch}): {e}")
    # --- REENTRENAMIENTO ---
    if hay_correcciones_nuevas and len(X) > 0:
        print(f"\n📊 Métrica: Entrenando con {len(X)} muestras totales (Originales + {len(archivos_correccion_procesados)} Correcciones Nuevas).")
        X = np.array(X)
        y_cat = to_categorical(y, num_classes=len(lista_clases))
        # Reentrenamiento del modelo con las nuevas correcciones
        try:
            print("⚙️ Inyectando nuevo conocimiento (Ajuste Fino)...")
            
            # Evaluación previa segura
            eval_pre = modelo.evaluate(X, y_cat, verbose=0)
            precision_pre = eval_pre[1] # El índice 1 corresponde a la 'accuracy' gracias a la compilación
            
            # Ajuste Fino (Fine-Tuning)
            historial = modelo.fit(X, y_cat, epochs=5, verbose=0) 
            precision_post = historial.history['accuracy'][-1]
            
            print(f"📈 Análisis Post-Entrenamiento:")
            print(f"   - Precisión Base: {precision_pre*100:.2f}%")
            print(f"   - Precisión Nueva: {precision_post*100:.2f}%")
            # Comparación de precisión para decidir si guardar el modelo
            if precision_post >= (precision_pre - 0.02): # Permitir una degradación máxima del 2%
                modelo.save(ruta_modelo) # Guardar el modelo actualizado
                print(f"✅ ¡Cerebro actualizado con éxito! Cambios guardados.")
                # Mover las correcciones procesadas a la carpeta de historial
                for ruta_origen, nombre_arch in archivos_correccion_procesados: # Mover cada archivo de corrección procesado a la carpeta de historial 
                    ruta_destino = os.path.join(historial_dir, f"{tipo}_{nombre_arch}") # Construir la ruta de destino con el prefijo del tipo de cerebro
                    shutil.move(ruta_origen, ruta_destino) # Mover el archivo a la carpeta de historial
                print(f"🧹 {len(archivos_correccion_procesados)} correcciones archivadas de forma segura.")
            else:
                print(f"⚠️ ¡ALERTA! El reentrenamiento degradó el modelo. Se ha cancelado el guardado.")
        # --- Manejo de errores durante el reentrenamiento y evaluación ---
        except Exception as e:
            print(f"❌ ERROR al reentrenar o evaluar: {e}")
    else:
        print(f"ℹ️ No hay correcciones nuevas para el cerebro {tipo}. Todo en orden.")
# --- EJECUCIÓN ---
reentrenar_modelo("dinamico")
reentrenar_modelo("estatico")