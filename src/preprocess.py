"""
Preprocesamiento de imágenes para mejorar el OCR de boletas y facturas peruanas.

Pasos: redimensionamiento -> escala de grises -> eliminación de ruido
-> mejora de contraste (CLAHE) -> binarización -> corrección de inclinación.
Cada paso se guarda para poder mostrarlo en la app.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Carga de archivos (JPG, PNG, PDF)
# ---------------------------------------------------------------------------
def load_file(data: bytes, filename: str, dpi: int = 300) -> list[np.ndarray]:
    """Devuelve una lista de imágenes BGR (una por página si es PDF)."""
    if filename.lower().endswith(".pdf"):
        import pymupdf as fitz

        pages = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pages.append(img)
        return pages

    pil = Image.open(io.BytesIO(data))
    # Respeta la orientación EXIF de fotos tomadas con celular
    try:
        from PIL import ImageOps

        pil = ImageOps.exif_transpose(pil)
    except Exception:
        pass
    pil = pil.convert("RGB")
    return [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)]


# ---------------------------------------------------------------------------
# Pasos individuales
# ---------------------------------------------------------------------------
def resize(img: np.ndarray, target_width: int = 1600) -> np.ndarray:
    """Escala la imagen a un ancho fijo. Las boletas térmicas suelen tener
    letra muy pequeña, así que ampliar ayuda a Tesseract."""
    h, w = img.shape[:2]
    if w == target_width:
        return img
    scale = target_width / w
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str = "nlmeans", strength: int = 10) -> np.ndarray:
    if method == "mediana":
        k = max(3, strength // 3 * 2 + 1)  # tamaño de kernel impar
        return cv2.medianBlur(gray, k)
    if method == "gaussiano":
        return cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.fastNlMeansDenoising(gray, None, h=strength, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE: ecualización de histograma adaptativa por bloques."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize(gray: np.ndarray, method: str = "adaptativa", block: int = 41, c: int = 10) -> np.ndarray:
    if method == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    block = block if block % 2 == 1 else block + 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


def deskew(img: np.ndarray, max_angle: float = 10.0, step: float = 0.25) -> tuple[np.ndarray, float]:
    """Corrige la inclinación por perfil de proyección: prueba ángulos y se
    queda con el que hace las líneas de texto más 'nítidas' (mayor varianza
    de la suma por filas)."""
    gray = to_gray(img)
    h, w = gray.shape
    # Trabaja en una versión pequeña y sin bordes (los bordes oscuros de la
    # foto confunden la estimación)
    scale = 800 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    my, mx = int(sh * 0.06), int(sw * 0.06)
    small = small[my:sh - my, mx:sw - mx]
    th = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 15)
    ch, cw = th.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D((cw / 2, ch / 2), a, 1.0)
        rot = cv2.warpAffine(th, m, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(a), score
    if abs(best_angle) < 0.3:
        return img, 0.0
    m = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, best_angle


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    do_resize: bool = True
    target_width: int = 1600
    do_denoise: bool = True
    denoise_method: str = "nlmeans"      # nlmeans | mediana | gaussiano
    denoise_strength: int = 10
    do_contrast: bool = True
    clahe_clip: float = 2.0
    do_binarize: bool = True
    binarize_method: str = "adaptativa"  # adaptativa | otsu
    block_size: int = 41
    c_value: int = 10
    do_deskew: bool = True


@dataclass
class PreprocessResult:
    final: np.ndarray
    steps: dict[str, np.ndarray] = field(default_factory=dict)
    angle: float = 0.0


def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    cfg = cfg or PreprocessConfig()
    steps: dict[str, np.ndarray] = {"1. Original": img}
    out = img
    if cfg.do_resize:
        out = resize(out, cfg.target_width)
        steps["2. Redimensionada"] = out
    out = to_gray(out)
    steps["3. Escala de grises"] = out
    if cfg.do_denoise:
        out = denoise(out, cfg.denoise_method, cfg.denoise_strength)
        steps["4. Sin ruido"] = out
    if cfg.do_contrast:
        out = enhance_contrast(out, cfg.clahe_clip)
        steps["5. Contraste (CLAHE)"] = out
    angle = 0.0
    if cfg.do_deskew:
        out, angle = deskew(out)
        if angle:
            steps[f"6. Enderezada ({angle:+.1f}°)"] = out
    if cfg.do_binarize:
        out = binarize(out, cfg.binarize_method, cfg.block_size, cfg.c_value)
        steps["7. Binarizada"] = out
    return PreprocessResult(final=out, steps=steps, angle=angle)


def to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""
