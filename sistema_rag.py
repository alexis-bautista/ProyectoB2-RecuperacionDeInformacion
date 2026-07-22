"""
Sistema RAG (Retrieval-Augmented Generation) Multimodal.

Responsabilidades:
  1. Recibir una consulta textual del usuario.
  2. Recuperar los Top-k documentos más relevantes de la BaseDatosVectorial.
  3. Construir automáticamente el contexto textual formateado para el LLM.
  4. Generar una respuesta coherente utilizando dicho contexto.
  5. Empaquetar y devolver la respuesta junto con las evidencias (Top-k, imágenes, scores)
     para garantizar la transparencia y trazabilidad del sistema.
"""

import os
from typing import Dict, List, Optional
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()


class SistemaRAG:
    def __init__(
        self,
        constructor_embeddings,
        base_datos_vectorial,
        api_key: Optional[str] = None,
        modelo_llm: str = "gemini-3.1-flash-lite",
    ):
        """
        Inicializa el pipeline RAG.
        """
        self.constructor = constructor_embeddings
        self.bd_vectorial = base_datos_vectorial
        self.modelo_llm_nombre = modelo_llm

        # Configuración de la API del LLM (usando Gemini)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.modelo_llm = genai.GenerativeModel(self.modelo_llm_nombre)
        else:
            self.modelo_llm = None
            print(
                "Aviso: No se proporcionó GEMINI_API_KEY. El sistema funcionará en modo simulación."
            )

    def recuperar_evidencias(self, consulta: str, top_k: int = 5) -> List[Dict]:
        """
        Paso 1 y 2: Recibe la consulta y recupera los Top-k documentos más similares.
        """
        emb_consulta = self.constructor.generar_embedding_consulta(consulta)
        evidencias = self.bd_vectorial.recuperar_top_k(emb_consulta, top_k=top_k)
        return evidencias

    def construir_contexto(self, evidencias: List[Dict]) -> str:
        """
        Paso 3: Construye automáticamente el bloque de contexto estructurado.
        """
        contexto_lineas = []
        for i, ev in enumerate(evidencias, start=1):
            id_prod = ev.get("product_id", "N/A")
            texto = ev.get("texto", "")
            score = ev.get("score", 0.0)
            contexto_lineas.append(
                f"Documento [{i}] (ID: {id_prod}, Similitud: {score:.3f}):\n{texto}\n"
            )

        return "\n".join(contexto_lineas)

    def generar_respuesta(self, consulta: str, contexto: str) -> str:
        """
        Paso 4: Genera la respuesta mediante el LLM basándose únicamente en el contexto.
        """
        prompt = f"""Eres un asistente virtual de compras especializado en recomendación de productos.
Responde a la consulta del usuario utilizando ÚNICAMENTE la información proporcionada en el siguiente contexto.
Si la información no es suficiente para responder con certeza, indícalo educadamente.

Contexto de productos recuperados:
{contexto}

Consulta del usuario: {consulta}

Respuesta clara y detallada:"""

        if self.modelo_llm:
            try:
                response = self.modelo_llm.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error al llamar al modelo de lenguaje: {e}"
        else:
            # Respuesta simulada en caso de no contar con API KEY configurada
            return (
                f"[Modo Simulación] Basado en los {len(contexto.split('Documento'))-1} productos recuperados, "
                f"encontré opciones relevantes para '{consulta}'. Revisa las evidencias mostradas a continuación."
            )

    def responder_consulta(self, consulta: str, top_k: int = 5) -> Dict:
        """
        Ejecuta el pipeline RAG completo y devuelve tanto la respuesta como las evidencias.
        """
        # 1. Recuperar
        evidencias = self.recuperar_evidencias(consulta, top_k=top_k)

        # 2. Construir contexto
        contexto = self.construir_contexto(evidencias)

        # 3. Generar respuesta
        respuesta = self.generar_respuesta(consulta, contexto)

        return {
            "consulta": consulta,
            "respuesta": respuesta,
            "contexto_utilizado": contexto,
            "evidencias": evidencias,
        }
