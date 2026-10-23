"""
Limpieza de texto y análisis NLP para comprobantes de pago peruanos (SUNAT).

Técnicas implementadas:
1. Limpieza: minúsculas, quitar caracteres especiales, tokenización,
   eliminación de stopwords y corrección de espacios/símbolos típicos del OCR.
2. Extracción de entidades (NER basado en reglas): tipo de comprobante,
   serie-número, RUC emisor y cliente, DNI, razón social, fecha, moneda,
   op. gravada, IGV y total.
3. Validación: dígito verificador del RUC y cuadre del IGV (18 %).
4. Clasificación temática del gasto por diccionario de palabras clave.
5. Extracción de palabras clave (frecuencia y TF-IDF) y nube de palabras.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# 1. LIMPIEZA Y PREPROCESAMIENTO DEL TEXTO
# ---------------------------------------------------------------------------
STOPWORDS_ES = set("""
a al algo algunas algunos ante antes como con contra cual cuando de del desde donde
durante e el ella ellas ellos en entre era es esa esas ese eso esos esta estas este
esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mis mucho muy
nada ni no nos o os otra otro para pero poco por porque que quien se ser si sin sobre
son su sus tambien te tiene tu un una uno unos y ya yo usted ustedes sr sra
""".split())

# Palabras que aparecen en casi todos los comprobantes y no aportan al análisis
STOPWORDS_DOMINIO = set("""
ruc boleta factura venta electronica electronico comprobante representacion impresa
sunat consulte documento www com pe http https gracias compra cliente senor senores
dni fecha emision hora total subtotal igv op gravada gravadas exonerada inafecta
importe pagar pago efectivo vuelto son soles s cant cantidad descripcion precio unit
unitario valor nro numero n serie direccion dir av jr calle telf tel cel caja cajero
ticket items item und unidad uni nit codigo cod hash resumen autorizado mediante
resolucion intendencia sac eirl srl sa lima peru moneda forma condicion contado
""".split())

STOPWORDS = STOPWORDS_ES | STOPWORDS_DOMINIO


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_ocr_text(text: str) -> str:
    """Corrige errores frecuentes del OCR sin perder mayúsculas ni números.
    Este texto normalizado es el que se usa para extraer entidades."""
    t = text.replace("\r", "\n")
    t = re.sub(r"[‘’´`]", "'", t)
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[—–]", "-", t)
    # "S /", "S/." , "S/ ." -> "S/"
    t = re.sub(r"\bS\s*/\s*\.?", "S/ ", t)
    # Letras confundidas dentro de números: 1O.5O -> 10.50 ; 2l -> 21
    def _fix_digits(m: re.Match) -> str:
        return m.group(0).translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}))
    t = re.sub(r"(?<=\d)[OolI|]+|[OolI|]+(?=\d)", _fix_digits, t)
    # Serie de boleta: el OCR suele leer la "B" como "8" (8001-123 -> B001-123)
    t = re.sub(r"(?m)^\s*8(\d{3})\s*-\s*(\d{3,8})\b", r"B\1-\2", t)
    # ...o le agrega un carácter de más (B8003-123 -> B003-123)
    t = re.sub(r"\b([FB])[8B](\d{3})\s*-\s*(\d{3,8})\b", r"\1\2-\3", t)
    # Coma decimal pegada a espacio: "12 .50" -> "12.50"
    t = re.sub(r"(\d)\s+([.,])\s*(\d{2})\b", r"\1\2\3", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def clean_text(text: str) -> str:
    """Minúsculas, sin tildes, sin caracteres especiales ni números sueltos."""
    t = strip_accents(text.lower())
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-zñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(clean: str) -> list[str]:
    return re.findall(r"[a-zñ]+", clean)


def remove_stopwords(tokens: list[str], min_len: int = 3) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= min_len]


def text_pipeline(raw: str) -> dict:
    normalized = normalize_ocr_text(raw)
    clean = clean_text(normalized)
    tokens = tokenize(clean)
    filtered = remove_stopwords(tokens)
    return {
        "normalizado": normalized,
        "limpio": clean,
        "tokens": tokens,
        "tokens_filtrados": filtered,
    }


# ---------------------------------------------------------------------------
# 2. EXTRACCIÓN DE ENTIDADES
# ---------------------------------------------------------------------------
RE_AMOUNT = r"(\d{1,3}(?:,\d{3})+\.\d{2}|\d+[.,]\d{2})(?!\d)"
RE_SERIE = re.compile(r"\b([FBE][A-Z0-9]{3})\s*[-N°º]+\s*0*(\d{1,8})\b")
RE_RUC = re.compile(r"R\.?\s*U\.?\s*C\.?\s*[:N°º.#]*\s*((?:10|15|17|20)\d{9})")
RE_RUC_LOOSE = re.compile(r"\b((?:10|15|17|20)\d{9})\b")
RE_DNI = re.compile(r"D\.?\s*N\.?\s*I\.?\s*[:N°º.#]*\s*(\d{8})\b")
RE_DATE = re.compile(
    r"\b(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b|\b(\d{4})-(\d{2})-(\d{2})\b"
)
COMPANY_SUFFIX = re.compile(r"\b(S\.?\s?A\.?\s?C\.?|S\.?\s?A\.?\s?A\.?|E\.?\s?I\.?\s?R\.?\s?L\.?|S\.?\s?R\.?\s?L\.?|S\.?\s?A\.?)\s*$", re.I)


def ruc_valido(ruc: str) -> bool:
    """Valida el dígito verificador del RUC (módulo 11, algoritmo SUNAT)."""
    if not re.fullmatch(r"\d{11}", ruc or ""):
        return False
    pesos = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    suma = sum(int(d) * p for d, p in zip(ruc[:10], pesos))
    dv = 11 - (suma % 11)
    dv = {10: 0, 11: 1}.get(dv, dv)
    return dv == int(ruc[10])


def _to_float(s: str) -> float | None:
    s = s.replace(" ", "")
    # "1,234.50" -> 1234.50 ; "1.234,50" -> 1234.50 ; "12,50" -> 12.50
    if re.fullmatch(r"\d{1,3}(,\d{3})+\.\d{2}", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{2}", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _amount_on_line(line: str) -> float | None:
    found = re.findall(RE_AMOUNT, line)
    return _to_float(found[-1]) if found else None


def _find_amount(lines: list[str], patterns: list[str], exclude: str | None = None) -> float | None:
    """Busca el monto en la primera línea que coincida con algún patrón (en orden
    de prioridad). Si la línea no trae el número, mira la siguiente."""
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for i, line in enumerate(lines):
            plain = strip_accents(line)
            if rx.search(plain) and not (exclude and re.search(exclude, plain, re.I)):
                val = _amount_on_line(line)
                # A veces el OCR separa la etiqueta y el monto en otra línea
                if val is None and i + 1 < len(lines):
                    val = _amount_on_line(lines[i + 1])
                if val is None and i > 0:
                    val = _amount_on_line(lines[i - 1])
                if val is not None:
                    return val
    return None


def _parse_date(text: str) -> str | None:
    for m in RE_DATE.finditer(text):
        if m.group(4):
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        else:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
        if 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2035:
            return f"{d:02d}/{mo:02d}/{y}"
    return None


def _razon_social(lines: list[str], ruc_line_idx: int | None) -> str | None:
    head = lines[: (ruc_line_idx if ruc_line_idx is not None else 6) or 6]
    for line in head[:8]:
        if COMPANY_SUFFIX.search(line.strip()) and len(line.strip()) > 5:
            return line.strip()
    # Si no hay sufijo societario, la primera línea con varias letras
    for line in head[:4]:
        if len(re.findall(r"[A-Za-zÁÉÍÓÚÑ]", line)) >= 5:
            return line.strip()
    return None


@dataclass
class Comprobante:
    tipo: str | None = None
    serie: str | None = None
    numero: str | None = None
    ruc_emisor: str | None = None
    ruc_emisor_valido: bool | None = None
    razon_social: str | None = None
    ruc_cliente: str | None = None
    dni_cliente: str | None = None
    fecha_emision: str | None = None
    moneda: str | None = None
    op_gravada: float | None = None
    igv: float | None = None
    total: float | None = None
    igv_cuadra: bool | None = None
    campos_encontrados: int = 0
    alertas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alertas"] = "; ".join(self.alertas)
        return d


def extract_entities(normalized: str) -> Comprobante:
    c = Comprobante()
    lines = [l for l in normalized.split("\n") if l.strip()]
    upper = strip_accents(normalized.upper())

    # Tipo de comprobante
    if "NOTA DE CREDITO" in upper:
        c.tipo = "Nota de crédito"
    elif "FACTURA" in upper:
        c.tipo = "Factura"
    elif "BOLETA" in upper:
        c.tipo = "Boleta de venta"
    elif "TICKET" in upper:
        c.tipo = "Ticket"

    # Serie y número (F001-123, B002-0004567, EB01-12)
    m = RE_SERIE.search(normalized.upper())
    if m:
        c.serie, c.numero = m.group(1), m.group(2).zfill(8)
        if c.tipo is None:
            c.tipo = {"F": "Factura", "B": "Boleta de venta"}.get(c.serie[0], c.tipo)

    # RUC: el primero es del emisor; si hay un segundo distinto, es del cliente
    rucs, ruc_idx = [], None
    for i, line in enumerate(lines):
        for r in RE_RUC.findall(line.upper()):
            if r not in rucs:
                rucs.append(r)
                ruc_idx = i if ruc_idx is None else ruc_idx
    if not rucs:
        rucs = list(dict.fromkeys(RE_RUC_LOOSE.findall(normalized)))
    if rucs:
        c.ruc_emisor = rucs[0]
        c.ruc_emisor_valido = ruc_valido(rucs[0])
        if len(rucs) > 1:
            c.ruc_cliente = rucs[1]

    m = RE_DNI.search(normalized.upper())
    if m:
        c.dni_cliente = m.group(1)

    c.razon_social = _razon_social(lines, ruc_idx)
    c.fecha_emision = _parse_date(normalized)

    if re.search(r"US\s*\$|DOLAR|USD", upper):
        c.moneda = "USD"
    elif re.search(r"S/|SOLES|PEN\b", upper):
        c.moneda = "PEN"

    # Montos
    c.op_gravada = _find_amount(lines, [r"OP\.?\s*GRAVADA", r"GRAVAD[AO]", r"VALOR\s+(DE\s+)?VENTA", r"SUB\s*-?\s*TOTAL"])
    c.igv = _find_amount(lines, [r"I\.?\s*G\.?\s*V\.?"])
    c.total = _find_amount(
        lines,
        [r"IMPORTE\s+TOTAL", r"TOTAL\s+A\s+PAGAR", r"TOTAL\s+VENTA", r"\bTOTAL\b"],
        exclude=r"SUB\s*-?\s*TOTAL|GRAVAD|EXONERAD|INAFECT|DESCUENTO|ITEMS|CANT",
    )

    # Validaciones
    if c.ruc_emisor and not c.ruc_emisor_valido:
        c.alertas.append("RUC emisor con dígito verificador inválido (posible error de OCR)")
    if c.igv is not None and c.op_gravada:
        esperado = round(c.op_gravada * 0.18, 2)
        c.igv_cuadra = abs(esperado - c.igv) <= 0.05 + 0.005 * c.igv
        if not c.igv_cuadra:
            c.alertas.append(f"IGV leído {c.igv:.2f} ≠ 18% de op. gravada ({esperado:.2f})")
    if c.total and c.op_gravada and c.igv is not None:
        if abs(c.op_gravada + c.igv - c.total) > 0.05:
            c.alertas.append("Op. gravada + IGV no coincide con el total")
    if c.total is None:
        c.alertas.append("No se encontró el importe total")
    if c.op_gravada is None and c.total is not None and c.igv is not None:
        c.op_gravada = round(c.total - c.igv, 2)

    campos = ["tipo", "serie", "ruc_emisor", "razon_social", "fecha_emision", "total", "igv"]
    c.campos_encontrados = sum(getattr(c, k) is not None for k in campos)
    return c


# ---------------------------------------------------------------------------
# 3. CLASIFICACIÓN TEMÁTICA DEL GASTO
# ---------------------------------------------------------------------------
CATEGORIAS: dict[str, list[str]] = {
    "Alimentos y restaurantes": [
        "pollo", "brasa", "menu", "almuerzo", "cena", "desayuno", "ceviche", "chifa", "pizza",
        "hamburguesa", "gaseosa", "inca", "cola", "chicha", "jugo", "cafe", "restaurant",
        "restaurante", "cevicheria", "polleria", "anticucho", "lomo", "saltado", "arroz",
        "papa", "papas", "pan", "postre", "mozo", "mesa", "propina", "consumo", "bebida",
        "agua", "cerveza", "sandwich", "empanada", "torta", "helado", "comida", "delivery",
    ],
    "Supermercado y bodega": [
        "leche", "yogurt", "aceite", "azucar", "fideos", "detergente", "papel", "higienico",
        "atun", "galletas", "huevos", "queso", "mantequilla", "jabon", "shampoo", "avena",
        "market", "minimarket", "supermercado", "bodega", "abarrotes", "kg", "bolsa",
    ],
    "Combustible": [
        "gasohol", "diesel", "petroleo", "glp", "gnv", "galones", "galon", "grifo",
        "combustible", "surtidor", "isla", "octanos", "premium", "regular", "estacion",
    ],
    "Transporte": [
        "pasaje", "taxi", "bus", "terminal", "asiento", "viaje", "destino", "origen",
        "peaje", "estacionamiento", "cochera", "movilidad", "flete", "encomienda", "courier",
    ],
    "Salud y farmacia": [
        "farmacia", "botica", "paracetamol", "ibuprofeno", "amoxicilina", "tableta",
        "tabletas", "capsula", "jarabe", "mg", "ml", "receta", "consulta", "medica",
        "medico", "clinica", "laboratorio", "analisis", "vitamina", "alcohol", "mascarilla",
    ],
    "Tecnología": [
        "laptop", "mouse", "teclado", "monitor", "usb", "memoria", "cable", "cargador",
        "celular", "audifonos", "impresora", "tinta", "toner", "router", "disco", "ssd",
        "software", "licencia", "computadora", "hdmi",
    ],
    "Útiles y librería": [
        "cuaderno", "lapicero", "lapiz", "borrador", "folder", "archivador", "copias",
        "fotocopias", "impresiones", "anillado", "libreria", "libro", "papel", "bond",
        "resaltador", "plumon", "engrapador", "sobre",
    ],
    "Ferretería y construcción": [
        "cemento", "fierro", "clavos", "pintura", "brocha", "tubo", "pvc", "ladrillo",
        "arena", "tornillo", "ferreteria", "martillo", "cinta", "pegamento", "codo",
        "llave", "candado", "foco", "cable",
    ],
    "Servicios": [
        "luz", "energia", "kwh", "agua", "potable", "alcantarillado", "internet", "telefonia",
        "plan", "recibo", "suministro", "consumo", "mantenimiento", "servicio", "servicios",
        "alquiler", "arrendamiento", "honorarios",
    ],
    "Hospedaje": ["hotel", "hostal", "habitacion", "noche", "noches", "hospedaje", "check", "huesped"],
}


def classify_expense(tokens: list[str]) -> dict:
    """Clasifica el comprobante sumando coincidencias por categoría.
    Devuelve la categoría ganadora, su confianza y el puntaje de todas."""
    counts = Counter(tokens)
    scores = {cat: sum(counts[w] for w in set(words)) for cat, words in CATEGORIAS.items()}
    total = sum(scores.values())
    if total == 0:
        return {"categoria": "Otros", "confianza": 0.0, "puntajes": scores, "evidencia": []}
    best = max(scores, key=scores.get)
    evidencia = sorted({w for w in tokens if w in CATEGORIAS[best]})
    return {
        "categoria": best,
        "confianza": round(scores[best] / total, 2),
        "puntajes": scores,
        "evidencia": evidencia,
    }


# ---------------------------------------------------------------------------
# 4. PALABRAS CLAVE
# ---------------------------------------------------------------------------
def top_keywords(tokens: list[str], k: int = 10) -> list[tuple[str, int]]:
    return Counter(tokens).most_common(k)


def tfidf_keywords(docs_tokens: list[list[str]], k: int = 5) -> list[list[tuple[str, float]]]:
    """TF-IDF calculado a mano (sin sklearn) para un lote de comprobantes."""
    n = len(docs_tokens)
    df = Counter(w for toks in docs_tokens for w in set(toks))
    out = []
    for toks in docs_tokens:
        tf = Counter(toks)
        total = sum(tf.values()) or 1
        scores = {w: (c / total) * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in tf.items()}
        out.append(sorted(((w, round(s, 3)) for w, s in scores.items()), key=lambda x: -x[1])[:k])
    return out


# ---------------------------------------------------------------------------
# Análisis completo de un documento
# ---------------------------------------------------------------------------
def analyze(raw_text: str) -> dict:
    tp = text_pipeline(raw_text)
    comp = extract_entities(tp["normalizado"])
    cls = classify_expense(tp["tokens_filtrados"])
    return {
        **tp,
        "comprobante": comp,
        "clasificacion": cls,
        "keywords": top_keywords(tp["tokens_filtrados"]),
    }
