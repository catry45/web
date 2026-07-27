import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# --- CONFIGURACIÓN BÁSICA ---
ruta_base = "dataset_lsm_holistic"
historial_dir = "historial_errores"
modelos = {
    "dinamico": {"path": "modelos/modelo_lsm.keras", "clases": ['Ayudar', 'Bien', 'Comer', 'Dolor', 'Gracias', 'Hola', 'Mal', 'Por_favor']},
    "estatico": {"path": "modelos/modelo_lsm_estatico.keras", "clases": ['A', 'B', 'C', 'L']}
}

os.makedirs(historial_dir, exist_ok=True)

def reentrenar_modelo(tipo):
    ruta_modelo = modelos[tipo]["path"]
    lista_clases = modelos[tipo]["clases"]
    
    print("\n" + "="*50)
    print(f"🧠 INICIANDO ACTIVE LEARNING: CEREBRO {tipo.upper()} 🧠")
    print("="*50)
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encontró el modelo en '{ruta_modelo}'.")
        return

    try:
        modelo = load_model(ruta_modelo)
        # --- EVALUACIÓN INICIAL DEL MODELO ---
        print(f"✅ Modelo {tipo} cargado. Obteniendo métricas base...")
    except Exception as e:
        print(f"❌ ERROR CRÍTICO al cargar el modelo: {e}")
        return

    if not os.path.exists(f"{ruta_modelo}.bak"):
        try:
            shutil.copy(ruta_modelo, f"{ruta_modelo}.bak")
            print(f"🛡️ Backup de seguridad creado exitosamente (.bak).")
        except Exception as e:
            print(f"⚠️ Advertencia: No se pudo crear el backup: {e}")

    mapa = {c: i for i, c in enumerate(lista_clases)}
    X, y = [], []
    archivos_correccion_procesados = []
    hay_correcciones_nuevas = False
    
    # --- RECOLECCIÓN DE DATOS DE ENTRENAMIENTO ---
    for clase in lista_clases:
        ruta_carpeta = os.path.join(ruta_base, clase)
        if not os.path.exists(ruta_carpeta): continue
        for arch in os.listdir(ruta_carpeta):
            if not arch.endswith(".npy"): continue
                
            ruta_completa = os.path.join(ruta_carpeta, arch)
            es_correccion = arch.startswith("correccion_")
            if es_correccion: hay_correcciones_nuevas = True
                
            try:
                dato = np.load(ruta_completa)
                if tipo == "dinamico":
                    if dato.shape == (30, 264):
                        X.append(dato)
                        y.append(mapa[clase])
                elif tipo == "estatico":
                    if dato.shape == (30, 264):
                        X.append(dato[-1]) 
                        y.append(mapa[clase])
                    elif len(dato.shape) == 1 and dato.shape[0] == 264:
                        X.append(dato)
                        y.append(mapa[clase])
                
                if es_correccion:
                    archivos_correccion_procesados.append((ruta_completa, arch))
            except Exception as e:
                print(f"❌ ERROR: Archivo corrupto ignorado ({arch}): {e}")
    
    if hay_correcciones_nuevas and len(X) > 0:
        print(f"\n📊 Métrica: Entrenando con {len(X)} muestras totales (Originales + {len(archivos_correccion_procesados)} Correcciones Nuevas).")
        X = np.array(X)
        y_cat = to_categorical(y, num_classes=len(lista_clases))
        
        # --- REVICION PREVIA Y GUARDADO SEGURO ---
        try:
            print("⚙️ Inyectando nuevo conocimiento (Ajuste Fino)...")
            
            # Evaluamos la precisión antes del ajuste
            _, precision_pre = modelo.evaluate(X, y_cat, verbose=0)
            
            # Ajuste fino del modelo
            historial = modelo.fit(X, y_cat, epochs=5, verbose=0) 
            precision_post = historial.history['accuracy'][-1]
            
            print(f"📈 Análisis Post-Entrenamiento:")
            print(f"   - Precisión Base: {precision_pre*100:.2f}%")
            print(f"   - Precisión Nueva: {precision_post*100:.2f}%")

            # Solo guardamos si la precisión no baja demasiado
            if precision_post >= (precision_pre - 0.02):
                modelo.save(ruta_modelo)
                print(f"✅ ¡Cerebro actualizado con éxito! Cambios guardados.")
                
                for ruta_origen, nombre_arch in archivos_correccion_procesados:
                    ruta_destino = os.path.join(historial_dir, f"{tipo}_{nombre_arch}")
                    shutil.move(ruta_origen, ruta_destino)
                print(f"🧹 {len(archivos_correccion_procesados)} correcciones archivadas de forma segura.")
            else:
                print(f"⚠️ ¡ALERTA! El reentrenamiento degradó el modelo. Se ha cancelado el guardado para proteger la IA.")
                print(f"   Sugerencia: Revisa la calidad de las correcciones guardadas.")

        except Exception as e:
            print(f"❌ ERROR al reentrenar o evaluar: {e}")
    else:
        print(f"ℹ️ No hay correcciones nuevas para el cerebro {tipo}. Todo en orden.")

reentrenar_modelo("dinamico")
reentrenar_modelo("estatico")