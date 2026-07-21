# construir_embeddings.py
"""
Construcción de representaciones vectoriales (embeddings) multimodales
usando CLIP.

Responsabilidades:
  - Generar embeddings para los documentos del corpus (texto + imagen).
  - Generar embeddings para las consultas del usuario (texto).
  - Almacenar las representaciones vectoriales (índice FAISS + metadatos)
    para su recuperación posterior.

"""

import os
import pickle
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    import faiss
except ImportError:
    faiss = None


class ConstructorEmbeddings:
    def __init__(
        self, modelo_nombre: str = "openai/clip-vit-base-patch32", device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo CLIP '{modelo_nombre}' en {self.device}...")
        self.modelo = CLIPModel.from_pretrained(modelo_nombre).to(self.device)
        self.procesador = CLIPProcessor.from_pretrained(modelo_nombre)
        self.modelo.eval()

        self.indice_faiss = None
        self.metadatos: List[Dict] = []

    # ---------- Generación de embeddings ----------

    @staticmethod
    def _extraer_tensor(salida):
        """Compatibilidad entre versiones de transformers: algunas devuelven
        el tensor de embeddings directamente, otras un objeto ModelOutput
        (ej. BaseModelOutputWithPooling) que contiene el tensor adentro."""
        if torch.is_tensor(salida):
            return salida
        for atributo in (
            "pooler_output",
            "text_embeds",
            "image_embeds",
            "last_hidden_state",
        ):
            if hasattr(salida, atributo):
                valor = getattr(salida, atributo)
                if valor is not None:
                    return valor
        raise TypeError(
            f"No se pudo extraer el tensor de embeddings de: {type(salida)}"
        )

    @torch.no_grad()
    def generar_embeddings_texto(self, textos: List[str]) -> np.ndarray:
        """Genera embeddings normalizados (L2) para una lista de textos."""
        entradas = self.procesador(
            text=textos, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        salida = self.modelo.get_text_features(**entradas)
        emb = self._extraer_tensor(salida)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    @torch.no_grad()
    def generar_embeddings_imagen(self, imagenes: List[Image.Image]) -> np.ndarray:
        """Genera embeddings normalizados (L2) para una lista de imágenes PIL."""
        entradas = self.procesador(images=imagenes, return_tensors="pt").to(self.device)
        salida = self.modelo.get_image_features(**entradas)
        emb = self._extraer_tensor(salida)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    def generar_embedding_consulta(self, consulta: str) -> np.ndarray:
        """Genera el embedding normalizado de una consulta textual del usuario."""
        return self.generar_embeddings_texto([consulta])[0]

    # ---------- Generación de la matriz de embeddings del corpus ----------

    def generar_matriz_corpus(
        self, datos_estructurados: List[Dict], batch_size: int = 16
    ):
        """
        Recorre el corpus (lista de dicts con 'product_id', 'texto', 'imagen',
        tal como los entrega preparar_corpus.asociar_multimodal) y genera un
        vector multimodal por documento (promedio texto+imagen, renormalizado).

        No indexa ni almacena nada: solo devuelve (matriz_embeddings, metadatos).
        El indexado/almacenamiento/recuperación es responsabilidad del módulo
        base_datos_vectorial.py.
        """
        vectores = []
        metadatos = []
        total = len(datos_estructurados)
        print(f"Generando embeddings para {total} documentos...")

        for inicio in range(0, total, batch_size):
            lote = datos_estructurados[inicio : inicio + batch_size]
            textos = [d["texto"] for d in lote]
            imagenes = [d["imagen"] for d in lote]

            emb_texto = self.generar_embeddings_texto(textos)
            emb_imagen = self.generar_embeddings_imagen(imagenes)

            emb_multimodal = (emb_texto + emb_imagen) / 2.0
            emb_multimodal = emb_multimodal / np.linalg.norm(
                emb_multimodal, axis=1, keepdims=True
            )
            vectores.append(emb_multimodal)

            for d in lote:
                metadatos.append(
                    {
                        "product_id": d["product_id"],
                        "texto": d["texto"],
                        "image_url": d.get("image_url", ""),
                    }
                )

            print(f"  {min(inicio + batch_size, total)}/{total} procesados")

        matriz = np.vstack(vectores).astype("float32")
        return matriz, metadatos

    # ---------- Construcción del índice (compatibilidad) ----------

    def construir_base_vectorial(
        self, datos_estructurados: List[Dict], batch_size: int = 16
    ):
        """
        [Se mantiene por compatibilidad con notebooks anteriores]
        Genera la matriz de embeddings y construye además un índice FAISS
        interno. Para el flujo recomendado (con la base de datos vectorial
        como componente separado), usa generar_matriz_corpus() junto con
        la clase BaseDatosVectorial de base_datos_vectorial.py.
        """
        if faiss is None:
            raise ImportError("Falta faiss. Instala con: pip install faiss-cpu")

        matriz, self.metadatos = self.generar_matriz_corpus(
            datos_estructurados, batch_size=batch_size
        )
        dimension = matriz.shape[1]

        # Producto interno == similitud coseno porque los vectores están normalizados
        self.indice_faiss = faiss.IndexFlatIP(dimension)
        self.indice_faiss.add(matriz)

        print(
            f"Índice FAISS construido: {self.indice_faiss.ntotal} vectores, dimensión {dimension}."
        )
        return self.indice_faiss

    # ---------- Persistencia ----------

    def guardar(self, ruta_directorio: str = "vector_store"):
        """Guarda el índice FAISS y los metadatos asociados en disco."""
        if self.indice_faiss is None:
            raise ValueError(
                "No hay índice para guardar. Ejecuta construir_base_vectorial() primero."
            )
        os.makedirs(ruta_directorio, exist_ok=True)
        faiss.write_index(
            self.indice_faiss, os.path.join(ruta_directorio, "indice.faiss")
        )
        with open(os.path.join(ruta_directorio, "metadatos.pkl"), "wb") as f:
            pickle.dump(self.metadatos, f)
        print(f"Guardado en '{ruta_directorio}/' (indice.faiss + metadatos.pkl).")

    def cargar(self, ruta_directorio: str = "vector_store"):
        """Carga un índice FAISS y sus metadatos previamente guardados."""
        if faiss is None:
            raise ImportError("Falta faiss. Instala con: pip install faiss-cpu")
        self.indice_faiss = faiss.read_index(
            os.path.join(ruta_directorio, "indice.faiss")
        )
        with open(os.path.join(ruta_directorio, "metadatos.pkl"), "rb") as f:
            self.metadatos = pickle.load(f)
        print(f"Índice cargado: {self.indice_faiss.ntotal} vectores.")

    # ---------- Búsqueda (recuperación) ----------

    def buscar(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """Dada una consulta de texto, devuelve los top_k documentos más similares."""
        if self.indice_faiss is None:
            raise ValueError(
                "No hay índice cargado. Usa construir_base_vectorial() o cargar()."
            )

        emb_consulta = self.generar_embedding_consulta(consulta).reshape(1, -1)
        distancias, indices = self.indice_faiss.search(emb_consulta, top_k)

        resultados = []
        for score, idx in zip(distancias[0], indices[0]):
            if idx == -1:
                continue
            resultados.append({"score": float(score), **self.metadatos[idx]})
        return resultados


if __name__ == "__main__":
    # Ejecución standalone: prepara el corpus y construye el índice de una vez.
    from preparar_corpus import cargar_y_fusionar_corpus, asociar_multimodal

    df = cargar_y_fusionar_corpus(split="train")
    datos = asociar_multimodal(df, num_muestras=200)

    constructor = ConstructorEmbeddings()
    constructor.construir_base_vectorial(datos)
    constructor.guardar("vector_store")
