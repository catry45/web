import numpy as np
import os

# --- 1. CONFIGURACIÓN INICIAL ---
ruta_script = os.path.dirname(os.path.abspath(__file__))
directorio_raiz = os.path.abspath(os.path.join(ruta_script, '..'))
ruta_base = os.path.join(directorio_raiz, 'dataset_lsm_holistic')

print("\n" + "="*50)
print("🔍 INICIANDO AUDITORÍA DE DATOS LSM 🔍")
print("="*50)
print(f"📁 Analizando directorio: {ruta_base}\n")

if not os.path.exists(ruta_base):
    print(f"❌ ERROR CRÍTICO: No se encontró la carpeta '{ruta_base}'.")
    print("Verifica que el nombre sea correcto y que existan datos.")
    exit()

# --- 2. MÉTRICAS DE SALUD ---
total_archivos = 0
archivos_validos = 0
archivos_corruptos = 0
detalles_errores = []

# --- 3. PROCESO DE AUDITORÍA ---
# Recorremos solo las carpetas de señas
carpetas_palabras = [d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))]

for palabra in carpetas_palabras:
    ruta_palabra = os.path.join(ruta_base, palabra)
    
    # Solo revisamos archivos .npy
    archivos_npy = [f for f in os.listdir(ruta_palabra) if f.endswith('.npy')]
    
    if len(archivos_npy) == 0:
        print(f"⚠️ ADVERTENCIA: La carpeta de la seña '{palabra}' está vacía.")
        continue
        
    print(f"📂 Revisando seña: '{palabra}' ({len(archivos_npy)} muestras)...")
    
    for archivo in archivos_npy:
        total_archivos += 1
        ruta_archivo = os.path.join(ruta_palabra, archivo)
        
        try:
            data = np.load(ruta_archivo)
            # Validamos la forma de cada muestra
            if data.shape != (30, 264):
                archivos_corruptos += 1
                detalles_errores.append(f"❌ [{palabra}] {archivo} | Forma incorrecta: {data.shape} (Se esperaba (30, 264))")
            else:
                archivos_validos += 1
        except Exception as e:
            archivos_corruptos += 1
            detalles_errores.append(f"❌ [{palabra}] {archivo} | Archivo dañado o ilegible: {e}")

# --- 4. REPORTE FINAL ---
print("\n" + "="*50)
print("📊 REPORTE FINAL DE SALUD DEL DATASET 📊")
print("="*50)
print(f"Total de muestras analizadas : {total_archivos}")
print(f"✅ Muestras íntegras          : {archivos_validos}")
print(f"⚠️ Muestras corruptas         : {archivos_corruptos}")
print("="*50)

if archivos_corruptos > 0:
    print("\n🚨 ACCIÓN REQUERIDA (DETALLE DE ERRORES):")
    print("Se recomienda eliminar los siguientes archivos y regrabarlos:")
    for error in detalles_errores:
        print(error)
else:
    print("\n✨ ¡CERTIFICADO DE CALIDAD! ✨")
    print("Todos los datos tienen la dimensión matemática perfecta.")
    print("El dataset está listo y seguro para entrenar a la IA.\n")