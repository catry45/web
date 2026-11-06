"""
Servidor web del Traductor LSM.
- FastAPI + WebSocket para streaming de frames en tiempo real
- Usa los modelos Keras y MediaPipe del prototipo original
"""

import os
import sys
import base64
import asyncio
import cv2
import numpy as np
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent          # .../backend
WEB_DIR = BASE_DIR.parent                           # .../traductor_lsm_web
# Modelos: buscamos en varias ubicaciones posibles
CANDIDATOS_MODELOS = [
    WEB_DIR.parent / "Traductor_LSM_src" / "Traductor_LSM-main" / "modelos",
    WEB_DIR.parent / "Traductor_LSM" / "modelos",
    Path.cwd() / "modelos",
    Path.cwd().parent / "modelos",
]

def encontrar_modelos() -> Path:
    for p in CANDIDATOS_MODELOS:
        if (p / "modelo_lsm.keras").exists() or (p / "modelo_lsm_v2_98pct_12clases.keras").exists():
            return p
        if (p / "modelo_lsm_estatico.keras").exists():
            return p
    raise FileNotFoundError(
        "No se encontraron los modelos .keras. "
        "Coloca la carpeta 'modelos' junto al proyecto o ajusta CANDIDATOS_MODELOS."
    )

# Importar el motor de predicción
sys.path.insert(0, str(BASE_DIR))
from predictor import PredictorLSM

app = FastAPI(
    title="Traductor LSM API",
    description="Backend del prototipo de Lengua de Señas Mexicana",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el predictor una sola vez al arrancar
print("=" * 50)
print("  Traductor LSM – Servidor Web")
print("=" * 50)
RUTA_MODELOS = encontrar_modelos()
print(f"📁 Modelos en: {RUTA_MODELOS}")
predictor = PredictorLSM(str(RUTA_MODELOS))

# Servir la interfaz estática
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "modelos": {
            "lstm": PredictorLSM.PALABRAS_LSTM,
            "estatico": PredictorLSM.LETRAS_ESTATICAS,
        }
    }


@app.get("/api/clases")
async def clases():
    return {
        "letras": PredictorLSM.LETRAS_ESTATICAS,
        "palabras": PredictorLSM.PALABRAS_LSTM,
    }


@app.websocket("/ws/traducir")
async def websocket_traducir(ws: WebSocket):
    """
    WebSocket principal.
    El cliente envía frames JPEG en base64.
    El servidor responde con JSON de predicción (+ frame anotado opcional).
    """
    await ws.accept()
    # Cada conexión tiene su propio estado de secuencia
    # (el predictor es compartido, pero reseteamos memoria por sesión)
    predictor.reset()
    print("🔌 Cliente WebSocket conectado")

    try:
        while True:
            data = await ws.receive_json()

            # Comandos de control
            if data.get("cmd") == "reset":
                predictor.reset()
                await ws.send_json({"tipo": "info", "mensaje": "Sesión reiniciada"})
                continue

            if data.get("cmd") == "ping":
                await ws.send_json({"tipo": "pong"})
                continue

            # Frame en base64
            b64 = data.get("frame")
            if not b64:
                continue

            # Decodificar
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                await ws.send_json({"tipo": "error", "mensaje": f"Frame inválido: {e}"})
                continue

            if frame is None:
                continue

            # Reducir resolución si viene muy grande (ahorro de CPU)
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            dibujar = data.get("landmarks", True)
            resultado = predictor.procesar_frame(frame, dibujar_landmarks=dibujar)

            # Preparar respuesta
            respuesta = {
                "tipo": "prediccion",
                "prediccion": resultado["prediccion"],
                "confianza": resultado["confianza"],
                "motor": resultado["motor"],
                "oracion": resultado["oracion"],
                "mano_detectada": resultado["mano_detectada"],
            }

            # Enviar frame anotado (opcional, más pesado)
            if data.get("devolver_frame", False) and resultado.get("frame_anotado") is not None:
                _, buf = cv2.imencode(".jpg", resultado["frame_anotado"],
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                respuesta["frame"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

            await ws.send_json(respuesta)

    except WebSocketDisconnect:
        print("🔌 Cliente WebSocket desconectado")
        predictor.reset()
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
        try:
            await ws.send_json({"tipo": "error", "mensaje": str(e)})
        except Exception:
            pass
        predictor.reset()


if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Abre en el navegador: http://localhost:8000")
    print("   WebSocket:           ws://localhost:8000/ws/traducir\n")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
