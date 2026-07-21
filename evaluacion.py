"""
Módulo de evaluación del sistema de recuperación de información.

Responsabilidades:
  - Calcular métricas estándar de recuperación: Precision@k, Recall@k y NDCG@k.
  - Ejecutar la evaluación por lotes utilizando un diccionario de juicios de relevancia (qrels).
"""

import numpy as np
from typing import Dict, List


def precision_at_k(relevantes: List[str], recuperados: List[str], k: int) -> float:
    """Calcula la proporción de documentos relevantes en el Top-k."""
    if not relevantes:
        return 0.0
    recuperados_k = recuperados[:k]
    relevantes_recuperados = [doc for doc in recuperados_k if doc in relevantes]
    return len(relevantes_recuperados) / k


def recall_at_k(relevantes: List[str], recuperados: List[str], k: int) -> float:
    """Calcula la proporción del total de documentos relevantes que fueron recuperados en el Top-k."""
    if not relevantes:
        return 0.0
    recuperados_k = recuperados[:k]
    relevantes_recuperados = [doc for doc in recuperados_k if doc in relevantes]
    return len(relevantes_recuperados) / len(relevantes)


def ndcg_at_k(relevantes: List[str], recuperados: List[str], k: int) -> float:
    """Calcula el Normalized Discounted Cumulative Gain (NDCG) en el Top-k."""
    if not relevantes:
        return 0.0

    # DCG: suma de la relevancia descontada por la posición logarítmica
    dcg = 0.0
    for i, doc in enumerate(recuperados[:k]):
        if doc in relevantes:
            dcg += 1.0 / np.log2(
                i + 2
            )  # +2 porque el índice empieza en 0 y log2(1) es 0

    # IDCG: el DCG ideal si todos los documentos relevantes estuvieran al principio
    idcg = 0.0
    for i in range(min(len(relevantes), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluar_sistema(
    qrels: Dict[str, List[str]], bd_vectorial, constructor, k: int = 5
) -> Dict[str, float]:
    """
    Evalúa el sistema procesando un conjunto de consultas contra sus documentos relevantes esperados.
    Retorna el promedio (Mean) de Precision, Recall y NDCG para el conjunto de pruebas.
    """
    metricas = {"Precision@k": [], "Recall@k": [], "NDCG@k": []}

    for query, docs_relevantes in qrels.items():
        # Generar embedding de la consulta
        emb_consulta = constructor.generar_embedding_consulta(query)

        # Recuperar Top-k de FAISS
        resultados = bd_vectorial.recuperar_top_k(emb_consulta, top_k=k)
        recuperados_ids = [r["product_id"] for r in resultados]

        # Calcular y almacenar métricas
        metricas["Precision@k"].append(
            precision_at_k(docs_relevantes, recuperados_ids, k)
        )
        metricas["Recall@k"].append(recall_at_k(docs_relevantes, recuperados_ids, k))
        metricas["NDCG@k"].append(ndcg_at_k(docs_relevantes, recuperados_ids, k))

    # Calcular el promedio de todas las consultas
    return {nombre: float(np.mean(valores)) for nombre, valores in metricas.items()}
