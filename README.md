# OCR y NLP de boletas y facturas peruanas

Aplicación en Streamlit que lee fotos o PDFs de comprobantes de pago peruanos (boletas, facturas, tickets), extrae el texto con OCR y lo analiza con técnicas de NLP para obtener los datos que pide la SUNAT y clasificar el tipo de gasto.

**Problema:** en el Perú muchas personas y pymes registran sus gastos a mano a partir de boletas impresas en papel térmico, que se borran con el tiempo y suelen fotografiarse con poca luz o torcidas. Digitalizar esos datos (RUC, fecha, IGV, total) toma tiempo y genera errores al deducir gastos o al hacer la conciliación contable.

## Qué hace

| Etapa | Técnica |
|---|---|
| Preprocesamiento | Redimensionamiento, escala de grises, eliminación de ruido (non-local means / mediana / gaussiano), contraste CLAHE, corrección de inclinación por perfil de proyección, binarización adaptativa u Otsu |
| OCR | Tesseract 5 (`pytesseract`) en español, con confianza promedio por palabra |
| Limpieza del texto | Corrección de errores típicos del OCR (O→0, B→8 en la serie, `S /`→`S/`), minúsculas, quitar tildes y caracteres especiales, tokenización, stopwords generales y del dominio |
| NLP | Extracción de entidades por reglas (tipo, serie-número, RUC, razón social, DNI, fecha, moneda, op. gravada, IGV, total), validación del RUC (dígito verificador módulo 11) y del IGV (18 %), clasificación temática del gasto, palabras clave por frecuencia y TF-IDF, nube de palabras |
| App | Carga múltiple (JPG, PNG, PDF), vista de cada paso, texto crudo vs. limpio, resultados NLP, resumen del lote y descarga en CSV, JSON y ZIP |

## Estructura

```
├── app.py                  # Aplicación Streamlit
├── src/
│   ├── preprocess.py       # Preprocesamiento de imágenes
│   ├── ocr.py              # OCR con Tesseract
│   └── nlp.py              # Limpieza de texto, entidades, clasificación, keywords
├── tools/
│   ├── evaluar.py          # Evaluación por lotes -> resultados/evaluacion.csv
│   └── generar_muestras.py # Genera boletas sintéticas de prueba
├── data/raw/               # Aquí van las fotos reales del dataset
├── samples/                # Boletas sintéticas (datos ficticios)
├── requirements.txt        # Librerías de Python
└── packages.txt            # Tesseract para Streamlit Cloud
```

## Ejecutar en local

```bash
# 1. Instalar Tesseract con el idioma español
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki (marcar "Spanish" al instalar)
#    Ubuntu:  sudo apt install tesseract-ocr tesseract-ocr-spa
#    macOS:   brew install tesseract tesseract-lang

# 2. Instalar librerías y correr
pip install -r requirements.txt
streamlit run app.py
```

En Windows, si sale `TesseractNotFoundError`, agrega al inicio de `src/ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio público de GitHub.
2. Entra a https://share.streamlit.io → **Create app** → elige el repositorio, rama `main` y archivo `app.py`.
3. `packages.txt` instala Tesseract y el paquete de español automáticamente.

## Dataset

Pon las fotos en `data/raw/` con nombres descriptivos (`polleria_01.jpg`, `grifo_02.jpg`, …). Conviene variar el rubro (restaurante, farmacia, grifo, librería, supermercado, ferretería), el tipo (boleta y factura) y la calidad de la foto (buena luz, sombra, inclinada, papel arrugado o desgastado). Antes de subirlas al repositorio, tapa el DNI o nombre del cliente si son tuyas o de alguien más.

Luego corre la evaluación:

```bash
python tools/evaluar.py data/raw
```

## Resultados con las boletas sintéticas (6 imágenes)

| Métrica | Sin preprocesar | Con preprocesamiento |
|---|---|---|
| Confianza OCR promedio | 85.9 % | 91.5 % |
| Campos extraídos (de 7) | 3.7 | 7.0 |

Los parámetros por defecto (binarización adaptativa, bloque 41, C = 10) salieron de comparar seis configuraciones; Otsu y la binarización sin ajustar dieron peores resultados.

## Limitaciones

- La extracción usa reglas (expresiones regulares), así que un formato de boleta muy distinto puede necesitar nuevos patrones.
- La clasificación del gasto usa un diccionario de palabras; boletas con descripciones genéricas («producto 1», «varios») caen en «Otros».
- Tesseract falla con papel térmico muy borrado, letra manuscrita y fotos muy borrosas.
- Mejoras posibles: probar EasyOCR o PaddleOCR, leer el código QR de la boleta electrónica para validar los datos, entrenar un clasificador con más boletas etiquetadas.
