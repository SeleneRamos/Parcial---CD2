"""
OCR + NLP de boletas y facturas peruanas
Ejecutar: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.nlp import CATEGORIAS, analyze, tfidf_keywords
from src.ocr import PSM_OPTIONS, available_langs, run_ocr
from src.preprocess import PreprocessConfig, load_file, preprocess, to_png_bytes

st.set_page_config(page_title="OCR Boletas Perú", page_icon="🧾", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"


# ---------------------------------------------------------------------------
# Barra lateral: parámetros
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")
    st.subheader("Preprocesamiento de imagen")
    do_resize = st.checkbox("Redimensionar", True)
    target_width = st.slider("Ancho objetivo (px)", 800, 2400, 1600, 100, disabled=not do_resize)
    do_denoise = st.checkbox("Eliminar ruido", True)
    denoise_method = st.selectbox("Método de filtrado", ["nlmeans", "mediana", "gaussiano"],
                                  disabled=not do_denoise,
                                  format_func={"nlmeans": "Non-local means", "mediana": "Mediana",
                                               "gaussiano": "Gaussiano"}.get)
    denoise_strength = st.slider("Intensidad del filtro", 3, 25, 10, disabled=not do_denoise)
    do_contrast = st.checkbox("Mejorar contraste (CLAHE)", True)
    clahe_clip = st.slider("Límite CLAHE", 1.0, 5.0, 2.0, 0.5, disabled=not do_contrast)
    do_deskew = st.checkbox("Corregir inclinación", True)
    do_binarize = st.checkbox("Binarizar", True)
    binarize_method = st.radio("Método", ["adaptativa", "otsu"], horizontal=True, disabled=not do_binarize)
    block_size = st.slider("Tamaño de bloque", 11, 81, 41, 2,
                           disabled=not (do_binarize and binarize_method == "adaptativa"))
    c_value = st.slider("Constante C", 0, 30, 10,
                        disabled=not (do_binarize and binarize_method == "adaptativa"))

    st.subheader("OCR (Tesseract)")
    langs = available_langs()
    lang = st.selectbox("Idioma", [l for l in ["spa", "spa+eng", "eng"] if l.split("+")[0] in langs] or ["eng"])
    psm = st.selectbox("Modo de segmentación (PSM)", list(PSM_OPTIONS), format_func=PSM_OPTIONS.get)
    compare_raw = st.checkbox("Comparar con OCR sin preprocesar", True,
                              help="Corre el OCR también sobre la imagen original para medir la mejora.")

cfg = PreprocessConfig(
    do_resize=do_resize, target_width=target_width, do_denoise=do_denoise,
    denoise_method=denoise_method, denoise_strength=denoise_strength,
    do_contrast=do_contrast, clahe_clip=clahe_clip, do_binarize=do_binarize,
    binarize_method=binarize_method, block_size=block_size, c_value=c_value,
    do_deskew=do_deskew,
)


# ---------------------------------------------------------------------------
# Procesamiento (con caché para no repetir el OCR al mover la interfaz)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def process_file(data: bytes, name: str, cfg_dict: dict, lang: str, psm: int, compare_raw: bool) -> list[dict]:
    cfg = PreprocessConfig(**cfg_dict)
    results = []
    pages = load_file(data, name)
    for p, img in enumerate(pages, start=1):
        pp = preprocess(img, cfg)
        ocr = run_ocr(pp.final, lang=lang, psm=psm)
        analysis = analyze(ocr.text)
        raw = None
        if compare_raw:
            r = run_ocr(img, lang=lang, psm=psm)
            raw = {"text": r.text, "confidence": r.confidence, "n_words": r.n_words,
                   "campos": analyze(r.text)["comprobante"].campos_encontrados}
        results.append({
            "archivo": name if len(pages) == 1 else f"{name} (pág. {p})",
            "steps": {k: to_png_bytes(v) for k, v in pp.steps.items()},
            "angle": pp.angle,
            "ocr_text": ocr.text,
            "ocr_conf": ocr.confidence,
            "ocr_words": ocr.n_words,
            "raw": raw,
            "analysis": analysis,
        })
    return results


def wordcloud_png(tokens: list[str]) -> bytes | None:
    if not tokens:
        return None
    from wordcloud import WordCloud

    wc = WordCloud(width=900, height=420, background_color="white", colormap="viridis",
                   collocations=False, random_state=1).generate(" ".join(tokens))
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


def fmt_money(v):
    return "—" if v is None else f"S/ {v:,.2f}"


def yes_no(v):
    return "—" if v is None else ("Sí" if v else "No")


# ---------------------------------------------------------------------------
# Encabezado y carga de archivos
# ---------------------------------------------------------------------------
st.title("🧾 OCR y NLP de boletas y facturas peruanas")
st.caption(
    "Sube fotos o PDFs escaneados de comprobantes de pago. La app limpia la imagen, extrae el texto "
    "con Tesseract, identifica los campos SUNAT (RUC, serie, fecha, IGV, total), valida el RUC y el IGV, "
    "y clasifica el tipo de gasto."
)

col_up, col_demo = st.columns([3, 1])
with col_up:
    uploads = st.file_uploader("Cargar imágenes o documentos", type=["jpg", "jpeg", "png", "pdf"],
                               accept_multiple_files=True)
with col_demo:
    st.write("")
    use_samples = st.toggle("Usar boletas de ejemplo", value=not uploads,
                            help="Boletas sintéticas con datos ficticios, solo para probar.")

files: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in (uploads or [])]
if use_samples and SAMPLES_DIR.exists():
    files += [(p.name, p.read_bytes()) for p in sorted(SAMPLES_DIR.glob("*.png"))]

if not files:
    st.info("Sube al menos una imagen (JPG, PNG) o PDF para empezar.")
    st.stop()

docs: list[dict] = []
progress = st.progress(0.0, text="Procesando…")
for i, (name, data) in enumerate(files):
    progress.progress(i / len(files), text=f"Procesando {name}…")
    docs.extend(process_file(data, name, cfg.__dict__, lang, psm, compare_raw))
progress.empty()

# TF-IDF del lote (palabras características de cada comprobante)
tfidf = tfidf_keywords([d["analysis"]["tokens_filtrados"] for d in docs], k=5)
for d, kw in zip(docs, tfidf):
    d["tfidf"] = kw

# Tabla resumen
rows = []
for d in docs:
    c = d["analysis"]["comprobante"]
    cl = d["analysis"]["clasificacion"]
    rows.append({
        "archivo": d["archivo"],
        "tipo": c.tipo,
        "serie_numero": f"{c.serie}-{c.numero}" if c.serie else None,
        "ruc_emisor": c.ruc_emisor,
        "ruc_valido": c.ruc_emisor_valido,
        "razon_social": c.razon_social,
        "fecha": c.fecha_emision,
        "moneda": c.moneda,
        "op_gravada": c.op_gravada,
        "igv": c.igv,
        "total": c.total,
        "igv_cuadra": c.igv_cuadra,
        "categoria": cl["categoria"],
        "conf_categoria": cl["confianza"],
        "conf_ocr": d["ocr_conf"],
        "conf_ocr_sin_prep": d["raw"]["confidence"] if d["raw"] else None,
        "campos_encontrados": c.campos_encontrados,
        "palabras_clave": ", ".join(w for w, _ in d["tfidf"]),
        "alertas": "; ".join(c.alertas),
    })
df = pd.DataFrame(rows)

tab_doc, tab_lote, tab_desc = st.tabs(["Documento", "Resumen del lote", "Descargas"])

# ---------------------------------------------------------------------------
# Pestaña 1: análisis de un documento
# ---------------------------------------------------------------------------
with tab_doc:
    sel = st.selectbox("Documento", range(len(docs)), format_func=lambda i: docs[i]["archivo"])
    d = docs[sel]
    a = d["analysis"]
    c = a["comprobante"]
    cl = a["clasificacion"]

    m1, m2, m3, m4 = st.columns(4)
    delta = None
    if d["raw"]:
        delta = f"{d['ocr_conf'] - d['raw']['confidence']:+.1f} vs. sin preprocesar"
    m1.metric("Confianza OCR", f"{d['ocr_conf']:.1f} %", delta)
    m2.metric("Campos extraídos", f"{c.campos_encontrados} / 7")
    m3.metric("Importe total", fmt_money(c.total))
    m4.metric("Categoría de gasto", cl["categoria"], f"{cl['confianza']:.0%} de coincidencias", delta_color="off")

    s1, s2, s3, s4 = st.tabs(["1. Preprocesamiento", "2. Texto extraído", "3. Texto limpio", "4. Análisis NLP"])

    with s1:
        steps = list(d["steps"].items())
        cols = st.columns(min(4, len(steps)))
        for j, (label, png) in enumerate(steps):
            cols[j % len(cols)].image(png, caption=label, width="stretch")
        if d["angle"]:
            st.caption(f"Inclinación corregida: {d['angle']:+.2f}°")

    with s2:
        if d["raw"]:
            left, right = st.columns(2)
            left.markdown(f"**Sin preprocesar** · confianza {d['raw']['confidence']:.1f} % · "
                          f"{d['raw']['campos']} campos")
            left.code(d["raw"]["text"] or "(sin texto)", language=None)
            right.markdown(f"**Con preprocesamiento** · confianza {d['ocr_conf']:.1f} % · "
                           f"{c.campos_encontrados} campos")
            right.code(d["ocr_text"] or "(sin texto)", language=None)
        else:
            st.code(d["ocr_text"] or "(sin texto)", language=None)

    with s3:
        left, right = st.columns(2)
        left.markdown("**Texto normalizado** (corrección de espacios y símbolos del OCR)")
        left.code(a["normalizado"], language=None)
        right.markdown("**Texto limpio** (minúsculas, sin tildes ni caracteres especiales)")
        right.code(a["limpio"], language=None)
        st.markdown(f"**Tokens:** {len(a['tokens'])} · **sin stopwords:** {len(a['tokens_filtrados'])}")
        st.write(" · ".join(a["tokens_filtrados"]) or "—")

    with s4:
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("**Entidades extraídas**")
            ent = pd.DataFrame([
                ("Tipo de comprobante", c.tipo),
                ("Serie - número", f"{c.serie}-{c.numero}" if c.serie else None),
                ("RUC emisor", c.ruc_emisor),
                ("RUC válido (dígito verificador)", yes_no(c.ruc_emisor_valido)),
                ("Razón social", c.razon_social),
                ("RUC cliente", c.ruc_cliente),
                ("DNI cliente", c.dni_cliente),
                ("Fecha de emisión", c.fecha_emision),
                ("Moneda", c.moneda),
                ("Op. gravada", fmt_money(c.op_gravada)),
                ("IGV", fmt_money(c.igv)),
                ("Importe total", fmt_money(c.total)),
                ("IGV cuadra con 18 %", yes_no(c.igv_cuadra)),
            ], columns=["Campo", "Valor"])
            ent["Valor"] = ent["Valor"].fillna("—").astype(str)
            st.dataframe(ent, hide_index=True, width="stretch")
            for al in c.alertas:
                st.warning(al)
            if not c.alertas:
                st.success("Sin alertas de validación.")
        with right:
            st.markdown("**Clasificación del gasto**")
            sc = pd.DataFrame({"categoría": list(cl["puntajes"]), "coincidencias": list(cl["puntajes"].values())})
            sc = sc[sc["coincidencias"] > 0].sort_values("coincidencias")
            if len(sc):
                st.bar_chart(sc, x="categoría", y="coincidencias", horizontal=True, height=220)
                st.caption("Palabras que la respaldan: " + ", ".join(cl["evidencia"]))
            else:
                st.caption("No se encontraron palabras de ninguna categoría → «Otros».")
            st.markdown("**Palabras clave**")
            kc1, kc2 = st.columns(2)
            kc1.dataframe(pd.DataFrame(a["keywords"], columns=["frecuencia", "n"]),
                          hide_index=True, width="stretch")
            kc2.dataframe(pd.DataFrame(d["tfidf"], columns=["TF-IDF", "peso"]),
                          hide_index=True, width="stretch")
        png = wordcloud_png(a["tokens_filtrados"])
        if png:
            st.markdown("**Nube de palabras**")
            st.image(png, width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 2: resumen del lote
# ---------------------------------------------------------------------------
with tab_lote:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Comprobantes", len(df))
    k2.metric("Gasto total identificado", fmt_money(df["total"].sum(min_count=1)))
    k3.metric("Confianza OCR promedio", f"{df['conf_ocr'].mean():.1f} %",
              f"{df['conf_ocr'].mean() - df['conf_ocr_sin_prep'].mean():+.1f} vs. sin preprocesar"
              if df["conf_ocr_sin_prep"].notna().any() else None)
    k4.metric("Campos extraídos (prom.)", f"{df['campos_encontrados'].mean():.1f} / 7")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Gasto por categoría (S/)**")
        by_cat = df.groupby("categoria", as_index=False)["total"].sum().sort_values("total")
        st.bar_chart(by_cat, x="categoria", y="total", horizontal=True, height=260)
    with g2:
        st.markdown("**Confianza del OCR: sin vs. con preprocesamiento**")
        if df["conf_ocr_sin_prep"].notna().any():
            comp = df[["archivo", "conf_ocr_sin_prep", "conf_ocr"]].rename(
                columns={"conf_ocr_sin_prep": "Sin preprocesar", "conf_ocr": "Con preprocesamiento"})
            st.bar_chart(comp.set_index("archivo"), stack=False, height=260)
        else:
            st.caption("Activa «Comparar con OCR sin preprocesar» en la barra lateral.")

    st.markdown("**Tabla de comprobantes**")
    st.dataframe(df, hide_index=True, width="stretch")

    all_tokens = [t for d in docs for t in d["analysis"]["tokens_filtrados"]]
    png = wordcloud_png(all_tokens)
    if png:
        st.markdown("**Nube de palabras del lote**")
        st.image(png, width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 3: descargas
# ---------------------------------------------------------------------------
with tab_desc:
    st.markdown("Descarga los resultados del lote procesado.")
    st.download_button("Tabla de comprobantes (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       "comprobantes.csv", "text/csv", width="stretch")

    full = []
    for d in docs:
        a = d["analysis"]
        full.append({
            "archivo": d["archivo"],
            "texto_ocr": d["ocr_text"],
            "texto_limpio": a["limpio"],
            "tokens_filtrados": a["tokens_filtrados"],
            "entidades": a["comprobante"].to_dict(),
            "clasificacion": {k: v for k, v in a["clasificacion"].items()},
            "palabras_clave": a["keywords"],
            "tfidf": d["tfidf"],
            "confianza_ocr": d["ocr_conf"],
        })
    st.download_button("Resultados completos (JSON)",
                       json.dumps(full, ensure_ascii=False, indent=2).encode("utf-8"),
                       "resultados.json", "application/json", width="stretch")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in docs:
            stem = Path(d["archivo"]).stem.replace(" ", "_")
            zf.writestr(f"{stem}_ocr.txt", d["ocr_text"])
            zf.writestr(f"{stem}_limpio.txt", d["analysis"]["limpio"])
    st.download_button("Textos extraídos y limpios (ZIP)", buf.getvalue(), "textos.zip",
                       "application/zip", width="stretch")

    with st.expander("Categorías y palabras clave usadas para clasificar"):
        st.json(CATEGORIAS)
