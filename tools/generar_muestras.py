"""
Genera boletas/facturas SINTÉTICAS (negocios ficticios) con ruido, sombra e
inclinación, solo para probar el pipeline antes de tener el dataset real.
Uso: python tools/generar_muestras.py
"""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def ruc_con_dv(base10: str) -> str:
    pesos = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    dv = 11 - sum(int(d) * p for d, p in zip(base10, pesos)) % 11
    return base10 + str({10: 0, 11: 1}.get(dv, dv))


CASOS = [
    dict(archivo="01_polleria.png", empresa="POLLERIA EL BUEN SABOR S.A.C.", ruc="2060123456",
         tipo="BOLETA DE VENTA ELECTRONICA", serie="B001-00004521", fecha="12/10/2026",
         items=[("1", "1/4 POLLO A LA BRASA", 22.90), ("1", "PAPAS FRITAS PORCION", 8.50),
                ("2", "GASEOSA INCA COLA 500ML", 7.00), ("1", "ENSALADA CLASICA", 5.00)],
         cliente="DNI: 45879632"),
    dict(archivo="02_grifo.png", empresa="GRIFO SAN MARTIN E.I.R.L.", ruc="2054987321",
         tipo="FACTURA ELECTRONICA", serie="F002-00012877", fecha="05/10/2026",
         items=[("8.50", "GASOHOL 90 PLUS GALONES", 140.25)],
         cliente="RUC: <2010020030>|CLIENTE: TRANSPORTES ANDINOS S.A.C."),
    dict(archivo="03_botica.png", empresa="BOTICA SALUD Y VIDA S.R.L.", ruc="2071122334",
         tipo="BOLETA DE VENTA ELECTRONICA", serie="B003-00000987", fecha="18/09/2026",
         items=[("2", "PARACETAMOL 500MG TABLETA X10", 6.00), ("1", "JARABE PARA LA TOS 120ML", 18.90),
                ("1", "VITAMINA C 1G X 10", 12.50), ("1", "ALCOHOL 70% 250ML", 4.50)],
         cliente="DNI: 70123456"),
    dict(archivo="04_libreria.png", empresa="LIBRERIA LA MOLINA S.A.C.", ruc="2060987654",
         tipo="BOLETA DE VENTA ELECTRONICA", serie="B001-00031245", fecha="02/10/2026",
         items=[("3", "CUADERNO A4 CUADRICULADO", 19.50), ("5", "LAPICERO AZUL", 7.50),
                ("50", "FOTOCOPIAS B/N", 5.00), ("1", "ANILLADO", 4.00), ("1", "FOLDER MANILA", 1.00)],
         cliente="DNI: 72456123"),
    dict(archivo="05_tecnologia.png", empresa="COMPUTO PERU TECH S.A.C.", ruc="2055566677",
         tipo="FACTURA ELECTRONICA", serie="F001-00000312", fecha="20/10/2026",
         items=[("1", "MOUSE INALAMBRICO USB", 45.00), ("1", "TECLADO USB ESPANOL", 59.90),
                ("1", "MEMORIA USB 64GB", 32.00), ("1", "CABLE HDMI 2M", 19.90)],
         cliente="RUC: <2060000001>|CLIENTE: UNIVERSIDAD EJEMPLO"),
    dict(archivo="06_ferreteria.png", empresa="FERRETERIA EL MAESTRO S.R.L.", ruc="2050011122",
         tipo="BOLETA DE VENTA ELECTRONICA", serie="B002-00005566", fecha="15/10/2026",
         items=[("2", "CEMENTO SOL BOLSA 42.5KG", 58.00), ("1", "PINTURA LATEX GALON", 45.50),
                ("1", "BROCHA 3 PULGADAS", 9.00), ("1", "CLAVOS 2 PULG KG", 8.00)],
         cliente="DNI: 41236587"),
]


def render(caso: dict) -> Image.Image:
    f, fb = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT_B, 22)
    W, lh = 620, 30
    lines: list[tuple[str, ImageFont.FreeTypeFont, str]] = []
    add = lambda t, font=f, al="c": lines.append((t, font, al))
    ruc = ruc_con_dv(caso["ruc"])
    add(caso["empresa"], fb); add(f"RUC: {ruc}"); add("AV. LA MOLINA 1234 - LIMA"); add("")
    add(caso["tipo"], fb); add(caso["serie"], fb); add("-" * 46)
    add(f"FECHA EMISION: {caso['fecha']}  HORA: 13:45", al="l")
    import re
    cli = re.sub(r"<(\d{10})>", lambda m: ruc_con_dv(m.group(1)), caso["cliente"])
    for part in cli.split("|"):
        add(part, al="l")
    add("MONEDA: SOLES", al="l"); add("-" * 46)
    add("CANT  DESCRIPCION                   IMPORTE", al="l")
    total = 0.0
    for q, d, imp in caso["items"]:
        add(f"{q:<5} {d[:26]:<26} {imp:>9.2f}", al="l"); total += imp
    total = round(total, 2)
    grav = round(total / 1.18, 2); igv = round(total - grav, 2)
    add("-" * 46)
    add(f"{'OP. GRAVADA':<30} S/ {grav:>9.2f}", al="l")
    add(f"{'IGV (18%)':<30} S/ {igv:>9.2f}", al="l")
    add(f"{'IMPORTE TOTAL':<30} S/ {total:>9.2f}", fb, "l")
    add(""); add("DOCUMENTO DE PRUEBA - DATOS FICTICIOS"); add("GRACIAS POR SU COMPRA")
    H = lh * (len(lines) + 2)
    img = Image.new("RGB", (W, H), (250, 248, 240))
    dr = ImageDraw.Draw(img)
    for i, (t, font, al) in enumerate(lines):
        w = dr.textlength(t, font=font)
        x = (W - w) / 2 if al == "c" else 20
        dr.text((x, 20 + i * lh), t, font=font, fill=(40, 40, 40))
    return img


def degradar(img: Image.Image, nivel: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR).astype(np.float32)
    h, w = a.shape[:2]
    # Sombra/iluminación irregular (como foto con celular)
    grad = np.tile(np.linspace(1.0, 0.55 + 0.1 * (3 - nivel), w), (h, 1))[..., None]
    a *= grad
    a = cv2.addWeighted(a, 0.8, np.full_like(a, 90), 0.2, 0)  # bajo contraste
    a += rng.normal(0, 6 + 4 * nivel, a.shape)                 # ruido
    a = np.clip(a, 0, 255).astype(np.uint8)
    a = cv2.GaussianBlur(a, (3, 3), 0.6 + 0.3 * nivel)
    # Inclinación
    ang = rng.uniform(-4, 4)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    a = cv2.warpAffine(a, M, (w, h), borderValue=(70, 70, 70))
    # Resolución baja
    return cv2.resize(a, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    random.seed(7)
    for i, caso in enumerate(CASOS):
        img = degradar(render(caso), nivel=1 + i % 3, seed=i)
        cv2.imwrite(str(OUT / caso["archivo"]), img)
        print("ok", caso["archivo"])
