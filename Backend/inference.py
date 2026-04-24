# inference.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import base64
import io

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# Necesario para pad reflect (si no lo tienes: pip install opencv-python)
import cv2


# =========================
# CONFIG (TU MODELO)
# =========================
ENCODER_NAME = "efficientnet-b0"   # IMPORTANTE: tu .pth tiene keys tipo encoder._conv_stem.weight
NUM_CLASSES = 4                    # 0=fondo, 1=veg, 2=pav, 3=estruct

TILE = 512
OVERLAP = 96  # si ves “costuras” sube a 128 (más lento)

# Colores (RGB) para UI
C_BG   = (0, 0, 0)
C_VEG  = (34, 197, 94)     # verde
C_PAV  = (245, 158, 11)    # amarillo/naranja
C_EST  = (239, 68, 68)     # rojo


@dataclass
class InferenceOutput:
    mask: np.ndarray                 # (H,W) uint8 0..3
    conf: np.ndarray                 # (H,W) float32 0..1
    pct_by_class: Dict[str, float]   # incluye fondo (0..3) suma 100
    coverage_ui: Dict[str, float]    # sin fondo (veg/pav/est) suma 100
    recommendations: List[str]
    mask_png_b64: str                # data:image/png;base64,...
    overlay_png_b64: str             # data:image/png;base64,...


# =========================
# HELPERS
# =========================
def _to_png_b64(arr: np.ndarray) -> str:
    """
    arr: RGB uint8 (H,W,3) o RGBA uint8 (H,W,4)
    """
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


def _stats_pct(mask: np.ndarray, n: int) -> Dict[str, float]:
    total = float(mask.size)
    return {str(c): float((mask == c).sum() * 100.0 / total) for c in range(n)}


def _coverage_ui_from_mask(mask: np.ndarray) -> Dict[str, float]:
    """
    CÁLCULO DIRECTO de la máscara final (evita el bug de pavimento=0 por mapeos viejos).
    """
    veg = int((mask == 1).sum())
    pav = int((mask == 2).sum())
    est = int((mask == 3).sum())
    s = veg + pav + est
    if s <= 0:
        return {"vegetacion": 0.0, "pavimento": 0.0, "estructuras": 0.0}
    return {
        "vegetacion": veg * 100.0 / s,
        "pavimento": pav * 100.0 / s,
        "estructuras": est * 100.0 / s,
    }


def _recommendations(coverage_ui: Dict[str, float]) -> List[str]:
    veg = float(coverage_ui.get("vegetacion", 0.0))
    pav = float(coverage_ui.get("pavimento", 0.0))
    est = float(coverage_ui.get("estructuras", 0.0))

    impermeable = pav + est
    recs: List[str] = []

    if veg < 18:
        recs.append("Aumentar cobertura vegetal: arborización, corredores verdes y recuperación de suelos permeables.")
    if impermeable > 60:
        recs.append("Reducir impermeabilización: pavimento permeable, jardines de lluvia y superficies drenantes.")
    if est > 35 and veg < 25:
        recs.append("Mejorar confort térmico: techos fríos/techos verdes y más sombra en vías peatonales.")
    if pav > 25 and veg < 25:
        recs.append("Mejorar movilidad y seguridad: más sombra en vías, cruces peatonales y materiales fríos.")

    if not recs:
        recs.append("Zona equilibrada: conservar cobertura actual y mejorar conectividad ecológica.")
    return recs[:3]


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask == 1] = C_VEG
    out[mask == 2] = C_PAV
    out[mask == 3] = C_EST
    return out


def _mask_rgba(mask_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fondo transparente para mostrar “Máscara” limpia.
    """
    h, w, _ = mask_rgb.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = mask_rgb
    rgba[..., 3] = (mask != 0).astype(np.uint8) * 255
    return rgba


def _overlay(img_rgb: np.ndarray, mask_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Overlay con alpha por clase (estructuras más visible).
    """
    img = img_rgb.astype(np.float32)

    alpha_map = np.zeros(mask.shape, dtype=np.float32)
    alpha_map[mask == 1] = 0.45
    alpha_map[mask == 2] = 0.55
    alpha_map[mask == 3] = 0.65

    m = mask_rgb.astype(np.float32)
    a = alpha_map[..., None]  # (H,W,1)

    out = img * (1.0 - a) + m * a
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Contorno blanco para que se vea mejor la segmentación (sobre todo techos)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    for cls in (1, 2, 3):
        binm = (mask == cls).astype(np.uint8) * 255
        if binm.sum() == 0:
            continue
        e = cv2.Canny(binm, 50, 150)
        edges = cv2.bitwise_or(edges, e)

    out[edges > 0] = (255, 255, 255)
    return out


def _postprocess(mask: np.ndarray) -> np.ndarray:
    """
    Postproceso SUAVE:
    - Cierra huecos de estructuras/pavimento
    - SOLO rellena sobre fondo (no pisa vegetación/pavimento/estructuras ya asignadas)
    """
    out = mask.copy()

    def close_fill(cls_id: int, k: int = 7, it: int = 1):
        nonlocal out
        binm = (out == cls_id).astype(np.uint8) * 255
        if binm.sum() == 0:
            return
        kernel = np.ones((k, k), np.uint8)
        closed = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=it)
        # Rellena SOLO donde era fondo
        out[(closed == 255) & (out == 0)] = cls_id

    # estructuras un poco más fuerte
    close_fill(3, k=9, it=1)
    # pavimento medio
    close_fill(2, k=7, it=1)

    return out


# =========================
# MODEL WRAPPER
# =========================
class SkyLensModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        ).to(self.device)

        state = torch.load(self.model_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def predict_big_image(self, img_rgb: np.ndarray, tile: int = TILE, overlap: int = OVERLAP) -> Tuple[np.ndarray, np.ndarray]:
        H, W, _ = img_rgb.shape
        step = tile - overlap
        if step <= 0:
            raise ValueError("overlap debe ser menor que tile")

        probs_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        w_sum = np.zeros((H, W), dtype=np.float32)

        # pesos sin ceros (evita “matar” bordes)
        w = np.ones((tile, tile), dtype=np.float32)

        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1 = min(y0 + tile, H)
                x1 = min(x0 + tile, W)

                patch = img_rgb[y0:y1, x0:x1]
                ph, pw = patch.shape[:2]

                # pad reflect a tile x tile
                if ph != tile or pw != tile:
                    pad_y = tile - ph
                    pad_x = tile - pw
                    patch = cv2.copyMakeBorder(
                        patch, 0, pad_y, 0, pad_x,
                        borderType=cv2.BORDER_REFLECT_101
                    )

                x = torch.from_numpy(patch).permute(2, 0, 1).contiguous().float().unsqueeze(0) / 255.0
                x = x.to(self.device)

                logits = self.model(x)            # (1,C,tile,tile)
                prob = F.softmax(logits, dim=1)[0]  # (C,tile,tile)
                prob_np = prob.detach().float().cpu().numpy()

                hh = (y1 - y0)
                ww_ = (x1 - x0)
                prob_np = prob_np[:, :hh, :ww_]
                wwgt = w[:hh, :ww_]

                probs_sum[:, y0:y1, x0:x1] += prob_np * wwgt[None, :, :]
                w_sum[y0:y1, x0:x1] += wwgt

        probs_avg = probs_sum / (w_sum[None, :, :] + 1e-8)
        mask = np.argmax(probs_avg, axis=0).astype(np.uint8)
        conf = np.max(probs_avg, axis=0).astype(np.float32)
        return mask, conf

    def run(self, img_rgb: np.ndarray) -> InferenceOutput:
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError("img_rgb debe ser (H,W,3)")

        mask, conf = self.predict_big_image(img_rgb, tile=TILE, overlap=OVERLAP)

        mask = _postprocess(mask)

        pct_by_class = _stats_pct(mask, NUM_CLASSES)
        coverage_ui = _coverage_ui_from_mask(mask)
        recs = _recommendations(coverage_ui)

        mask_rgb = _colorize_mask(mask)
        overlay = _overlay(img_rgb, mask_rgb, mask)
        mask_rgba = _mask_rgba(mask_rgb, mask)

        return InferenceOutput(
            mask=mask,
            conf=conf,
            pct_by_class=pct_by_class,
            coverage_ui=coverage_ui,
            recommendations=recs,
            mask_png_b64=_to_png_b64(mask_rgba),
            overlay_png_b64=_to_png_b64(overlay),
        )