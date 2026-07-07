import numpy as np
import os

# Obtiene la ruta de donde está guardado este script (verificar_datos.py)
ruta_script = os.path.dirname(os.path.abspath(__file__))

# Une esa ruta con el nombre de tu carpeta
ruta_base = os.path.join(ruta_script, "dataset_lsm_holistic")

print(f"Buscando en: {ruta_base}") # Esto te ayudará a debuggear si falla

for palabra in os.listdir(ruta_base):
    ruta_palabra = os.path.join(ruta_base, palabra)
    for archivo in os.listdir(ruta_palabra):
        data = np.load(os.path.join(ruta_palabra, archivo))
        if data.shape != (30, 264):
            print(f"❌ Error en: {archivo} | Forma actual: {data.shape}")
        # Si todo está bien, no imprime nada