# app.py
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import SkyLensModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model_b0.pth"  # tu MIX+SD renombrado

app = FastAPI(title="SkyLens API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SkyLensModel(MODEL_PATH)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": True, "model_path": str(MODEL_PATH)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"No pude leer la imagen: {e}"}, status_code=400)

    img_rgb = np.array(img)

    try:
        out = model.run(img_rgb)
    except Exception as e:
        return JSONResponse({"error": f"Fallo inferencia: {e}"}, status_code=500)

    return {
        "pct_by_class": out.pct_by_class,
        "coverage_ui": out.coverage_ui,
        "recommendations": out.recommendations,
        "mask_png": out.mask_png_b64,
        "overlay_png": out.overlay_png_b64,
        "confidence_mean": float(out.conf.mean()),
        "shape": {"h": int(img_rgb.shape[0]), "w": int(img_rgb.shape[1])},
    }