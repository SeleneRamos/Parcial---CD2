"""
Evalúa el pipeline sobre una carpeta de imágenes y guarda una tabla con la
confianza del OCR (sin y con preprocesamiento), los campos extraídos y el tiempo.

Uso:
    python tools/evaluar.py data/raw            # tus fotos reales
    python tools/evaluar.py samples             # boletas sintéticas
Salida: resultados/evaluacion.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp import analyze  # noqa: E402
from src.ocr import run_ocr  # noqa: E402
from src.preprocess import load_file, preprocess  # noqa: E402

carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"})
if not archivos:
    sys.exit(f"No hay imágenes en {carpeta}")

filas = []
for f in archivos:
    for pag, img in enumerate(load_file(f.read_bytes(), f.name), start=1):
        t0 = time.time()
        raw = run_ocr(img)
        a_raw = analyze(raw.text)
        pp = preprocess(img)
        ocr = run_ocr(pp.final)
        a = analyze(ocr.text)
        c = a["comprobante"]
        filas.append({
            "archivo": f.name, "pagina": pag,
            "conf_sin_prep": raw.confidence, "conf_con_prep": ocr.confidence,
            "campos_sin_prep": a_raw["comprobante"].campos_encontrados,
            "campos_con_prep": c.campos_encontrados,
            "tipo": c.tipo, "ruc_emisor": c.ruc_emisor, "ruc_valido": c.ruc_emisor_valido,
            "fecha": c.fecha_emision, "total": c.total, "igv_cuadra": c.igv_cuadra,
            "categoria": a["clasificacion"]["categoria"],
            "angulo_corregido": round(pp.angle, 2), "segundos": round(time.time() - t0, 1),
        })
        print(f"{f.name}: conf {raw.confidence} -> {ocr.confidence} | "
              f"campos {filas[-1]['campos_sin_prep']} -> {c.campos_encontrados} | {filas[-1]['categoria']}")

df = pd.DataFrame(filas)
Path("resultados").mkdir(exist_ok=True)
df.to_csv("resultados/evaluacion.csv", index=False, encoding="utf-8-sig")
print("\nPromedios")
print(df[["conf_sin_prep", "conf_con_prep", "campos_sin_prep", "campos_con_prep"]].mean().round(2).to_string())
print(f"RUC válido: {df['ruc_valido'].mean():.0%} | IGV cuadra: {df['igv_cuadra'].mean():.0%}")
print("Guardado en resultados/evaluacion.csv")
