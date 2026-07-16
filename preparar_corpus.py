# preparar_corpus.py
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
import re


def cargar_y_fusionar_corpus(split="train"):
    """
    Carga el dataset de imágenes de Crossing Minds y el texto original de Amazon ESCI,
    fusionándolos en un solo DataFrame.
    """
    print("Cargando URLs de imágenes (Crossing Minds)...")
    ds_urls = load_dataset(
        "crossingminds/shopping-queries-image-dataset",
        name="product_image_urls",
        split=split,
    ).to_pandas()

    print("Cargando textos originales (Amazon ESCI)...")
    ds_texto = load_dataset(
        "milistu/amazon-esci-data", name="products", split=split
    ).to_pandas()

    # Cruzamos la información usando product_id
    ds_texto = ds_texto.drop_duplicates(subset=["product_id"])
    df_fusionado = pd.merge(ds_urls, ds_texto, on="product_id", how="inner")

    return df_fusionado


def procesar_texto(texto):
    """
    Procesa el texto cuando sea necesario para estandarizar el contexto del RAG.
    """
    if not isinstance(texto, str) or pd.isna(texto):
        return ""

    texto = texto.lower()
    # Mantenemos caracteres alfanuméricos y puntuación básica
    texto = re.sub(r"[^a-záéíóúñ0-9\s.,-]", "", texto)
    return " ".join(texto.split())


def descargar_imagen(url):
    """
    Descarga una imagen desde su URL y la convierte a un objeto PIL.
    """
    try:
        respuesta = requests.get(url, timeout=5)
        respuesta.raise_for_status()
        return Image.open(BytesIO(respuesta.content)).convert("RGB")
    except Exception:
        # Retorna None si hay error 404 o timeout
        return None


def asociar_multimodal(df_fusionado, num_muestras=None):
    """
    Asocia correctamente cada imagen con la información textual correspondiente.
    """
    if num_muestras:
        df_fusionado = df_fusionado.head(num_muestras)

    datos_estructurados = []
    print(f"Descargando y asociando {len(df_fusionado)} muestras...")

    for _, fila in df_fusionado.iterrows():
        # Combinamos título y descripción para tener un contexto enriquecido para el RAG
        texto_crudo = (
            f"{fila.get('product_title', '')}. {fila.get('product_description', '')}"
        )
        texto_limpio = procesar_texto(texto_crudo)

        imagen_pil = descargar_imagen(fila["image_url"])

        # Solo guardamos el documento si cuenta con imagen válida y texto útil
        if imagen_pil and len(texto_limpio) > 5:
            datos_estructurados.append(
                {
                    "product_id": fila["product_id"],
                    "texto": texto_limpio,
                    "imagen": imagen_pil,
                }
            )

    return datos_estructurados
