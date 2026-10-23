"""
OCR con Tesseract (pytesseract). Devuelve el texto y la confianza promedio
por palabra, que se usa para comparar la imagen original vs. preprocesada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytesseract
from pytesseract import Output

# PSM (page segmentation mode) más útiles para boletas:
#  4 = una columna de texto de tamaño variable (bueno para tickets)
#  6 = un bloque uniforme de texto
#  3 = segmentación automática
PSM_OPTIONS = {
    4: "4 - Una columna (tickets)",
    6: "6 - Bloque uniforme",
    3: "3 - Automático",
    11: "11 - Texto disperso",
}


@dataclass
class OCRResult:
    text: str
    confidence: float  # promedio 0-100 de las palabras reconocidas
    n_words: int


def run_ocr(img: np.ndarray, lang: str = "spa", psm: int = 4) -> OCRResult:
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(img, lang=lang, config=config, output_type=Output.DICT)

    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]
    conf = float(np.mean(confs)) if confs else 0.0

    # Reconstruye el texto respetando líneas
    lines: dict[tuple, list[str]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
    return OCRResult(text=text, confidence=round(conf, 1), n_words=len(confs))


def available_langs() -> list[str]:
    try:
        return pytesseract.get_languages(config="")
    except Exception:
        return ["eng"]
