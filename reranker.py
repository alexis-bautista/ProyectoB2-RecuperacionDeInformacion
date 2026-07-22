# reranker.py
"""
Módulo de Re-ranking para el sistema RAG Multimodal.

Responsabilidades:
  - Refinar el ranking inicial obtenido por búsqueda vectorial (FAISS + CLIP)
    usando un modelo cross-encoder más preciso.
  - Contiene la clase ReRanker para reordenar candidatos.
  - Contiene la clase SistemaRAGConReranking (subclase de SistemaRAG) para
    integrarse al RAG sin modificar el archivo sistema_rag.py original.
"""

from typing import Dict, List, Optional
from sentence_transformers import CrossEncoder
from sistema_rag import SistemaRAG


class ReRanker:
    def __init__(
        self,
        modelo_nombre: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Carga el modelo cross-encoder para re-ranking.
        """
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cargando modelo de re-ranking '{modelo_nombre}' en {self.device}...")
        self.modelo = CrossEncoder(modelo_nombre, device=self.device)
        print("Modelo de re-ranking listo.")

    def rerank(
        self,
        consulta: str,
        candidatos: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Re-ordena la lista de candidatos usando el cross-encoder.
        """
        if not candidatos:
            return []

        pares = [(consulta, c["texto"]) for c in candidatos]
        scores = self.modelo.predict(pares)

        ranking = sorted(zip(scores, candidatos), key=lambda x: x[0], reverse=True)

        if top_k is not None:
            ranking = ranking[:top_k]

        resultados = []
        for posicion, (score, candidato) in enumerate(ranking, start=1):
            resultado = {**candidato}
            resultado["rank"] = posicion
            resultado["score_original"] = candidato.get("score", 0.0)
            resultado["score_reranking"] = float(score)
            resultados.append(resultado)

        return resultados


class SistemaRAGConReranking(SistemaRAG):
    def __init__(
        self,
        constructor_embeddings,
        base_datos_vectorial,
        api_key: Optional[str] = None,
        modelo_llm: str = "gemini-2.5-flash",
        reranker: Optional[ReRanker] = None,
        factor_candidatos: int = 4,
    ):
        """
        Subclase de SistemaRAG que implementa Re-ranking.
        Hereda todo el comportamiento del RAG original, pero añade soporte
        para refinar la búsqueda antes de enviar el contexto al LLM.
        """
        # Inicializamos la clase base original
        super().__init__(
            constructor_embeddings=constructor_embeddings,
            base_datos_vectorial=base_datos_vectorial,
            api_key=api_key,
            modelo_llm=modelo_llm,
        )
        self.reranker = reranker
        self.factor_candidatos = factor_candidatos

    def recuperar_evidencias(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """
        Sobrescribe recuperar_evidencias de la clase base.
        Recupera factor_candidatos * top_k de FAISS y luego aplica Re-ranking.
        """
        emb_consulta = self.constructor.generar_embedding_consulta(consulta)

        if self.reranker is not None:
            # Recuperamos más candidatos de FAISS para tener de dónde re-ordenar
            n_candidatos = top_k * self.factor_candidatos
            candidatos = self.bd_vectorial.recuperar_top_k(
                emb_consulta, top_k=n_candidatos
            )
            # Aplicamos re-ranking con el cross-encoder
            evidencias = self.reranker.rerank(consulta, candidatos, top_k=top_k)
        else:
            # Fallback al comportamiento original si no hay reranker
            evidencias = self.bd_vectorial.recuperar_top_k(emb_consulta, top_k=top_k)

        return evidencias
