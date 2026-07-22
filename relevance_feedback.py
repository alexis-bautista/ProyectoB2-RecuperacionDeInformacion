# relevance_feedback.py
"""
Módulo de Relevance Feedback usando el Algoritmo de Rocchio.

Responsabilidades:
  - Registrar las calificaciones ("Me gusta" / "No me gusta") de los usuarios para documentos específicos.
  - Ajustar vectorialmente las futuras consultas del usuario (Query Drift / Query Shift):
    acercando la consulta a los documentos relevantes (liked) y alejándola de los irrelevantes (disliked).
  - Implementar la clase SistemaRAGConRocchio (subclase de SistemaRAG) para
    integrar este comportamiento de forma limpia y transparente sin modificar los archivos base.
"""

from typing import Dict, List, Optional
import numpy as np
from sistema_rag import SistemaRAG


class RelevanceFeedbackSystem:
    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.25):
        """
        Inicializa el sistema de feedback con los pesos del Algoritmo de Rocchio.

        Formula: q_new = alpha * q_old + beta * avg(D_relevant) - gamma * avg(D_non_relevant)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Almacenamiento en memoria del feedback del usuario
        # Estructura: {consulta: {"likes": [doc_ids], "dislikes": [doc_ids]}}
        self.historial_feedback: Dict[str, Dict[str, List[str]]] = {}

    def registrar_feedback(self, consulta: str, product_id: str, es_relevante: bool):
        """
        Registra el feedback del usuario ('like' o 'dislike') para un producto y consulta específicos.
        """
        consulta_normalizada = consulta.strip().lower()
        if consulta_normalizada not in self.historial_feedback:
            self.historial_feedback[consulta_normalizada] = {
                "likes": [],
                "dislikes": [],
            }

        lista_destino = "likes" if es_relevante else "dislikes"
        lista_opuesta = "dislikes" if es_relevante else "likes"

        # Evitar duplicados
        if (
            product_id
            not in self.historial_feedback[consulta_normalizada][lista_destino]
        ):
            self.historial_feedback[consulta_normalizada][lista_destino].append(
                product_id
            )

        # Remover de la lista opuesta si existía previamente
        if product_id in self.historial_feedback[consulta_normalizada][lista_opuesta]:
            self.historial_feedback[consulta_normalizada][lista_opuesta].remove(
                product_id
            )

        print(
            f"[Feedback] Registrado {'LIKE' if es_relevante else 'DISLIKE'} para {product_id} bajo la consulta '{consulta_normalizada}'"
        )

    def obtener_vectores_documentos(
        self, doc_ids: List[str], bd_vectorial
    ) -> List[np.ndarray]:
        """
        Recupera los embeddings vectoriales originales de los documentos calificados.
        """
        vectores = []
        if not bd_vectorial.indice:
            return vectores

        for doc_id in doc_ids:
            # Buscar el índice físico del documento en los metadatos de la BD vectorial
            for idx, meta in enumerate(bd_vectorial.metadatos):
                if meta["product_id"] == doc_id:
                    # FAISS nos permite reconstruir el vector a partir de su ID de índice
                    vec = bd_vectorial.indice.reconstruct(idx)
                    vectores.append(vec)
                    break
        return vectores

    def ajustar_vector_consulta(
        self, consulta: str, vector_original: np.ndarray, bd_vectorial
    ) -> np.ndarray:
        """
        Aplica la fórmula de Rocchio para desplazar el vector de consulta original.
        """
        consulta_normalizada = consulta.strip().lower()
        feedback = self.historial_feedback.get(consulta_normalizada)

        # Si no hay feedback para esta consulta, devolvemos el vector original sin cambios
        if not feedback or (not feedback["likes"] and not feedback["dislikes"]):
            return vector_original

        q_new = self.alpha * vector_original

        # Componente Positivo (Likes)
        if feedback["likes"]:
            vecs_likes = self.obtener_vectores_documentos(
                feedback["likes"], bd_vectorial
            )
            if vecs_likes:
                avg_likes = np.mean(vecs_likes, axis=0)
                q_new = q_new + self.beta * avg_likes

        # Componente Negativo (Dislikes)
        if feedback["dislikes"]:
            vecs_dislikes = self.obtener_vectores_documentos(
                feedback["dislikes"], bd_vectorial
            )
            if vecs_dislikes:
                avg_dislikes = np.mean(vecs_dislikes, axis=0)
                q_new = q_new - self.gamma * avg_dislikes

        # Normalizamos el nuevo vector ajustado (L2 norm) para mantener compatibilidad en la búsqueda coseno
        norma = np.linalg.norm(q_new)
        if norma > 0:
            q_new = q_new / norma

        return q_new.astype("float32")


class SistemaRAGConRocchio(SistemaRAG):
    def __init__(
        self,
        constructor_embeddings,
        base_datos_vectorial,
        api_key: Optional[str] = None,
        modelo_llm: str = "gemini-3.1-flash-lite",
        feedback_system: Optional[RelevanceFeedbackSystem] = None,
    ):
        """
        Subclase de SistemaRAG que incorpora Relevance Feedback basado en Rocchio.
        """
        super().__init__(
            constructor_embeddings=constructor_embeddings,
            base_datos_vectorial=base_datos_vectorial,
            api_key=api_key,
            modelo_llm=modelo_llm,
        )
        self.feedback_system = feedback_system or RelevanceFeedbackSystem()

    def recuperar_evidencias(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """
        Sobrescribe la recuperación. Genera el embedding, lo ajusta usando Rocchio
        si hay feedback registrado, y recupera de FAISS.
        """
        emb_original = self.constructor.generar_embedding_consulta(consulta)

        # Ajustamos el vector de consulta según las calificaciones previas
        emb_ajustado = self.feedback_system.ajustar_vector_consulta(
            consulta, emb_original, self.bd_vectorial
        )

        evidencias = self.bd_vectorial.recuperar_top_k(emb_ajustado, top_k=top_k)
        return evidencias
