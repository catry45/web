# Traductor LSM – Versión Web

Interfaz de navegador + backend FastAPI que usa **los mismos modelos Keras** del prototipo de escritorio (PyQt6).

## Arquitectura

```
Navegador (HTML/JS)
    │  WebSocket (frames JPEG)
    ▼
FastAPI  ──►  MediaPipe Holistic  ──►  Comité de Expertos
                                         ├─ modelo_lsm_estatico.keras  (A, B, C, L)
                                         └─ modelo_lsm.keras           (palabras dinámicas)
```

## Requisitos

- Python **3.11.x** (el mismo del proyecto original)
- Dependencias del proyecto principal (`requirements.txt`)
- Adicionales para el servidor web:

```bash
pip install -r backend/requirements_web.txt
```

## Cómo arrancar

1. Asegúrate de que la carpeta `modelos/` con los `.keras` esté accesible  
   (el servidor la busca automáticamente en varias rutas).

2. Desde la carpeta `traductor_lsm_web`:

```bash
cd traductor_lsm_web
python backend/server.py
```

3. Abre el navegador en:

```
http://localhost:8000
```

4. Haz clic en **Iniciar cámara** y permite el acceso.

## Clases reconocidas

| Tipo | Clases |
|------|--------|
| Letras (modelo estático) | A, B, C, L |
| Palabras (modelo LSTM) | Ayudar, Bien, Comer, Dolor, Gracias, Hola, Mal, Por_favor |

## API

| Endpoint | Descripción |
|----------|-------------|
| `GET /` | Interfaz web |
| `GET /api/health` | Estado del servidor y clases |
| `GET /api/clases` | Lista de letras y palabras |
| `WS /ws/traducir` | Streaming de frames + predicciones |

## Estructura

```
traductor_lsm_web/
├── index.html
├── css/styles.css
├── js/app.js
├── backend/
│   ├── server.py          # FastAPI + WebSocket
│   ├── predictor.py       # Lógica MoE (MediaPipe + Keras)
│   └── requirements_web.txt
└── README.md
```

## Notas

- Los frames se envían a ~10–12 FPS para no saturar la CPU.
- La estabilización de texto es la misma del prototipo original.
- El motor (Estático / Dinámico) se decide por la velocidad de la muñeca.
