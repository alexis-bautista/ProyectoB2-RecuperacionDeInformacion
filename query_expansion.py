# query_expansion.py
"""
Módulo de Expansión de Consultas (Query Expansion) para el sistema RAG.

Responsabilidades:
  - Expandir o reformular la consulta inicial del usuario utilizando un LLM (Gemini)
    para generar variaciones, sinónimos y términos técnicos (en inglés, ya que el corpus está en inglés).
  - Realizar una búsqueda "Multi-Query": recuperar candidatos en la base vectorial para cada
    variación generada, fusionar los resultados, eliminar duplicados y ordenar por relevancia.
  - Implementar la clase SistemaRAGConExpansion (subclase de SistemaRAG) para
    integrar esta funcionalidad de forma limpia y transparente sin modificar los archivos originales.
"""

import os
import re
from typing import Dict, List, Optional
import google.generativeai as genai
from sistema_rag import SistemaRAG


class QueryExpander:
    def __init__(
        self, api_key: Optional[str] = None, modelo_llm: str = "gemini-3.1-flash-lite"
    ):
        """
        Inicializa el expansor de consultas mediante Gemini.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.modelo_nombre = modelo_llm
        self.modelo = None

        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.modelo = genai.GenerativeModel(self.modelo_nombre)
        else:
            print(
                "Aviso: No hay GEMINI_API_KEY para Query Expansion. Se usarán reglas básicas."
            )

    def expandir_consulta(self, consulta: str) -> List[str]:
        """
        Usa el LLM para generar 3 variaciones de la consulta original en inglés,
        incluyendo sinónimos y traducciones en caso de que la consulta venga en español.
        """
        consultas = [consulta]  # Siempre incluimos la consulta original

        if not self.modelo:
            # Fallback simple si no hay API key: devolver original
            return consultas

        prompt = f"""You are an assistant specialized in e-commerce search query reformulation.
Given the user's search query, generate exactly 3 alternative search queries in English. 
Focus on:
- Translating it to English if it is in Spanish.
- Using synonyms, hypernyms, or related product terms.
- Keeping them concise (2-5 words).

Output ONLY the 3 queries, one per line. Do not number them, do not add introductory or concluding text.

Original Query: {consulta}

Alternative Queries:"""

        try:
            respuesta = self.modelo.generate_content(prompt)
            lineas = respuesta.text.strip().split("\n")
            for linea in lineas:
                linea_limpia = re.sub(
                    r"^\d+[\.\-\)]\s*", "", linea
                ).strip()  # Limpia numeración
                linea_limpia = linea_limpia.replace('"', "").replace("'", "")
                if linea_limpia and len(linea_limpia) > 2:
                    consultas.append(linea_limpia)
        except Exception as e:
            print(f"Error en expansión de consultas: {e}")

        # Retornamos consultas únicas (máximo 4: original + 3 expansiones)
        return list(dict.fromkeys(consultas))[:4]


class SistemaRAGConExpansion(SistemaRAG):
    def __init__(
        self,
        constructor_embeddings,
        base_datos_vectorial,
        api_key: Optional[str] = None,
        modelo_llm: str = "gemini-3.1-flash-lite",
        expansor: Optional[QueryExpander] = None,
    ):
        """
        Subclase de SistemaRAG que implementa Query Expansion (Multi-Query Retrieval).
        """
        super().__init__(
            constructor_embeddings=constructor_embeddings,
            base_datos_vectorial=base_datos_vectorial,
            api_key=api_key,
            modelo_llm=modelo_llm,
        )
        self.expansor = expansor or QueryExpander(
            api_key=self.api_key, modelo_llm=self.modelo_llm_nombre
        )

    def recuperar_evidencias(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """
        Sobrescribe recuperar_evidencias de la clase base.
        Expande la consulta, busca en la base de datos para cada variación,
        fusiona los resultados (deduplicando) y devuelve el top_k global.
        """
        # 1. Obtener las consultas expandidas
        consultas_expandidas = self.expansor.expandir_consulta(consulta)
        print(f"[Query Expansion] Consultas generadas: {consultas_expandidas}")

        candidatos_fusionados = {}

        # 2. Búsqueda vectorial para cada variación
        for q in consultas_expandidas:
            emb_q = self.constructor.generar_embedding_consulta(q)
            resultados_q = self.bd_vectorial.recuperar_top_k(emb_q, top_k=top_k)

            for res in resultados_q:
                prod_id = res["product_id"]
                # Si el producto ya fue recuperado por otra consulta, nos quedamos con el mejor score
                if prod_id in candidatos_fusionados:
                    if res["score"] > candidatos_fusionados[prod_id]["score"]:
                        candidatos_fusionados[prod_id] = res
                else:
                    candidatos_fusionados[prod_id] = res

        # 3. Ordenar todos los candidatos fusionados por score
        ranking_global = sorted(
            candidatos_fusionados.values(), key=lambda x: x["score"], reverse=True
        )

        # 4. Ajustar el número de ranking (posiciones 1 a top_k)
        evidencias_finales = []
        for posicion, candidato in enumerate(ranking_global[:top_k], start=1):
            doc = {**candidato}
            doc["rank"] = posicion
            evidencias_finales.append(doc)

        return evidencias_finales
