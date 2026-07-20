# base_datos_vectorial.py
"""
Base de datos vectorial: indexado y recuperación de embeddings usando FAISS.

Responsabilidades:
  - Indexar el corpus (a partir de la matriz de embeddings ya generada por
    construir_embeddings.ConstructorEmbeddings).
  - Recuperar los documentos más similares para una consulta.
  - Obtener un ranking Top-k de resultados.

Este módulo es independiente del modelo de embeddings: solo trabaja con
vectores numpy (ya normalizados) y metadatos, así que en teoría podría
usarse con cualquier encoder, no solo CLIP.

Se usa FAISS (IndexFlatIP, equivalente a similitud coseno con vectores
normalizados) como motor de indexado. Si se prefiriera ChromaDB en vez
de FAISS, la interfaz pública (indexar_corpus / recuperar_top_k / guardar
/ cargar) se mantendría igual, cambiando solo la implementación interna.

Pensado para ser importado desde Proyecto.ipynb:

    from construir_embeddings import ConstructorEmbeddings
    from base_datos_vectorial import BaseDatosVectorial

    constructor = ConstructorEmbeddings()
    matriz, metadatos = constructor.generar_matriz_corpus(datos_estructurados)

    bd_vectorial = BaseDatosVectorial()
    bd_vectorial.indexar_corpus(matriz, metadatos)
    bd_vectorial.guardar("vector_store")

    emb_consulta = constructor.generar_embedding_consulta("zapatillas rojas")
    ranking = bd_vectorial.recuperar_top_k(emb_consulta, top_k=5)
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class BaseDatosVectorial:
    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension
        self.indice = None
        self.metadatos: List[Dict] = []

    # ---------- Indexado ----------

    def indexar_corpus(self, matriz_embeddings: np.ndarray, metadatos: List[Dict]):
        """
        Indexa el corpus completo a partir de su matriz de embeddings
        (forma: [num_documentos, dimension]) y una lista de metadatos
        (uno por documento, en el mismo orden que las filas de la matriz).
        """
        if faiss is None:
            raise ImportError("Falta faiss. Instala con: pip install faiss-cpu")
        if matriz_embeddings.ndim != 2:
            raise ValueError("matriz_embeddings debe tener forma [num_documentos, dimension].")
        if len(metadatos) != matriz_embeddings.shape[0]:
            raise ValueError(
                f"Cantidad de metadatos ({len(metadatos)}) no coincide con "
                f"cantidad de vectores ({matriz_embeddings.shape[0]})."
            )

        matriz_embeddings = np.ascontiguousarray(matriz_embeddings, dtype="float32")
        self.dimension = matriz_embeddings.shape[1]

        # IndexFlatIP = producto interno; con vectores normalizados equivale
        # a similitud coseno. Suficiente para corpus pequeños/medianos.
        self.indice = faiss.IndexFlatIP(self.dimension)
        self.indice.add(matriz_embeddings)
        self.metadatos = list(metadatos)

        print(f"Corpus indexado: {self.indice.ntotal} documentos, dimensión {self.dimension}.")
        return self.indice

    def agregar_documentos(self, matriz_embeddings: np.ndarray, metadatos: List[Dict]):
        """Agrega nuevos documentos a un índice ya existente (indexado incremental)."""
        if self.indice is None:
            raise ValueError("No hay índice creado todavía. Usa indexar_corpus() primero.")
        matriz_embeddings = np.ascontiguousarray(matriz_embeddings, dtype="float32")
        self.indice.add(matriz_embeddings)
        self.metadatos.extend(metadatos)
        print(f"Documentos agregados. Total actual: {self.indice.ntotal}.")

    # ---------- Recuperación / ranking Top-k ----------

    def recuperar_top_k(self, vector_consulta: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Dado el embedding de una consulta (ya generado, por ejemplo con
        ConstructorEmbeddings.generar_embedding_consulta), devuelve una
        lista ordenada (ranking) con los top_k documentos más similares.

        Cada elemento del ranking incluye: 'rank' (posición, empieza en 1),
        'score' (similitud coseno) y los metadatos del documento
        (product_id, texto, etc).
        """
        if self.indice is None:
            raise ValueError("El índice no ha sido creado. Usa indexar_corpus() o cargar() primero.")

        vector_consulta = np.ascontiguousarray(vector_consulta, dtype="float32").reshape(1, -1)
        distancias, indices = self.indice.search(vector_consulta, top_k)

        ranking = []
        for posicion, (score, idx) in enumerate(zip(distancias[0], indices[0]), start=1):
            if idx == -1:  # FAISS devuelve -1 cuando hay menos de top_k resultados
                continue
            ranking.append({"rank": posicion, "score": float(score), **self.metadatos[idx]})
        return ranking

    # ---------- Persistencia ----------

    def guardar(self, ruta_directorio: str = "vector_store"):
        """Guarda el índice FAISS y los metadatos asociados en disco."""
        if self.indice is None:
            raise ValueError("No hay índice para guardar. Ejecuta indexar_corpus() primero.")
        os.makedirs(ruta_directorio, exist_ok=True)
        faiss.write_index(self.indice, os.path.join(ruta_directorio, "indice.faiss"))
        with open(os.path.join(ruta_directorio, "metadatos.pkl"), "wb") as f:
            pickle.dump(self.metadatos, f)
        print(f"Base vectorial guardada en '{ruta_directorio}/' (indice.faiss + metadatos.pkl).")

    def cargar(self, ruta_directorio: str = "vector_store"):
        """Carga un índice FAISS y sus metadatos previamente guardados."""
        if faiss is None:
            raise ImportError("Falta faiss. Instala con: pip install faiss-cpu")
        self.indice = faiss.read_index(os.path.join(ruta_directorio, "indice.faiss"))
        with open(os.path.join(ruta_directorio, "metadatos.pkl"), "rb") as f:
            self.metadatos = pickle.load(f)
        self.dimension = self.indice.d
        print(f"Base vectorial cargada: {self.indice.ntotal} documentos, dimensión {self.dimension}.")

    def __len__(self):
        return 0 if self.indice is None else self.indice.ntotal


if __name__ == "__main__":
    # Ejecución standalone: prepara corpus, genera embeddings, indexa y
    # prueba una recuperación de ejemplo.
    from preparar_corpus import cargar_y_fusionar_corpus, asociar_multimodal
    from construir_embeddings import ConstructorEmbeddings

    df = cargar_y_fusionar_corpus(split="train")
    datos = asociar_multimodal(df, num_muestras=200)

    constructor = ConstructorEmbeddings()
    matriz, metadatos = constructor.generar_matriz_corpus(datos)

    bd_vectorial = BaseDatosVectorial()
    bd_vectorial.indexar_corpus(matriz, metadatos)
    bd_vectorial.guardar("vector_store")

    consulta = "zapatillas rojas para correr"
    emb_consulta = constructor.generar_embedding_consulta(consulta)
    ranking = bd_vectorial.recuperar_top_k(emb_consulta, top_k=5)

    print(f"\nTop-5 resultados para: '{consulta}'")
    for r in ranking:
        print(f"  #{r['rank']}  score={r['score']:.3f}  {r['product_id']}  {r['texto'][:60]}")