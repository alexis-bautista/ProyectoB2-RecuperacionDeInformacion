# memoria.py
"""
Módulo de Memoria Conversacional para el sistema RAG.

Responsabilidades:
  - Guardar el historial de interacciones (preguntas del usuario y respuestas del asistente).
  - Reformular la consulta actual del usuario en base al contexto histórico (Query Condensation):
    si el usuario hace una pregunta de seguimiento (ej. "tienes en talla 9?"), el LLM la reformula
    usando la memoria (ej. "zapatillas rojas de correr talla 9") antes de buscar en la base vectorial.
  - Generar respuestas del LLM inyectando el hilo conversacional para mantener la coherencia.
  - Implementar la clase SistemaRAGConMemoria (subclase de SistemaRAG) de forma limpia y aislada.
"""

import os
from typing import Dict, List, Optional
import google.generativeai as genai
from sistema_rag import SistemaRAG


class ConversationalMemory:
    def __init__(self):
        """Inicializa un historial de conversación vacío."""
        self.historial: List[Dict[str, str]] = []

    def agregar_mensaje(self, rol: str, contenido: str):
        """Agrega un mensaje al historial ('user' o 'assistant')."""
        self.historial.append({"role": rol, "content": contenido})

    def obtener_historial_formateado(self) -> str:
        """Formatea el historial de conversación en texto para el prompt."""
        lineas = []
        for msg in self.historial:
            rol_tag = "Usuario" if msg["role"] == "user" else "Asistente"
            lineas.append(f"{rol_tag}: {msg['content']}")
        return "\n".join(lineas)

    def vaciar(self):
        """Limpia el historial de conversación."""
        self.historial = []


class SistemaRAGConMemoria(SistemaRAG):
    def __init__(
        self,
        constructor_embeddings,
        base_datos_vectorial,
        api_key: Optional[str] = None,
        modelo_llm: str = "gemini-2.5-flash",
        memoria: Optional[ConversationalMemory] = None,
    ):
        """
        Subclase de SistemaRAG que incorpora memoria conversacional y reformulación de consultas.
        """
        super().__init__(
            constructor_embeddings=constructor_embeddings,
            base_datos_vectorial=base_datos_vectorial,
            api_key=api_key,
            modelo_llm=modelo_llm
        )
        self.memoria = memoria or ConversationalMemory()

    def reformular_consulta(self, consulta_actual: str) -> str:
        """
        Usa Gemini para condensar la consulta actual basándose en el historial de chat.
        Devuelve una consulta de búsqueda stand-alone en inglés lista para FAISS.
        """
        historial_str = self.memoria.obtener_historial_formateado()
        
        # Si el historial está vacío, no hay nada que condensar, pero la traducimos a inglés
        if not historial_str:
            prompt_traduccion = f"Translate the following search query to English. Output ONLY the translation: {consulta_actual}"
            if self.modelo_llm:
                try:
                    res = self.modelo_llm.generate_content(prompt_traduccion)
                    return res.text.strip()
                except Exception:
                    pass
            return consulta_actual

        prompt = f"""You are a search assistant. Given the following conversation history and a new follow-up query from the user, rewrite the follow-up query into a standalone, single search query in English that contains all necessary context from the history.

Conversation History:
{historial_str}

Follow-up Query: {consulta_actual}

Standalone Search Query (English):"""

        if self.modelo_llm:
            try:
                res = self.modelo_llm.generate_content(prompt)
                consulta_condensada = res.text.strip().replace('"', '').replace("'", "")
                print(f"[Memoria] Consulta reformulada: '{consulta_condensada}' (Original: '{consulta_actual}')")
                return consulta_condensada
            except Exception as e:
                print(f"Error al reformular consulta con memoria: {e}")
                
        return consulta_actual

    def responder_consulta(self, consulta: str, top_k: int = 5) -> Dict:
        """
        Sobrescribe responder_consulta para:
          1. Reformular la consulta según el historial.
          2. Recuperar evidencias usando la consulta reformulada.
          3. Generar la respuesta inyectando tanto el historial de chat como el nuevo contexto.
          4. Guardar la interacción actual en el historial de memoria.
        """
        # 1. Reformular consulta usando el historial
        consulta_busqueda = self.reformular_consulta(consulta)

        # 2. Recuperar evidencias
        evidencias = self.recuperar_evidencias(consulta_busqueda, top_k=top_k)
        contexto = self.construir_contexto(evidencias)

        # 3. Construir Prompt que incluya memoria y contexto
        historial_str = self.memoria.obtener_historial_formateado()
        prompt_rag = f"""Eres un asistente de compras conversacional con memoria. 
Utiliza el historial de conversación anterior y el nuevo contexto de productos para responder detalladamente a la consulta.

Historial de Conversación anterior:
{historial_str}

Nuevo Contexto de Productos recuperados:
{contexto}

Nueva Consulta del Usuario: {consulta}

Respuesta del Asistente:"""

        # Generar respuesta
        if self.modelo_llm:
            try:
                response = self.modelo_llm.generate_content(prompt_rag)
                respuesta = response.text
            except Exception as e:
                respuesta = f"Error al generar respuesta con memoria: {e}"
        else:
            respuesta = f"[Modo Simulación con Memoria] Entendido. Guardé en memoria tu consulta sobre '{consulta_busqueda}'."

        # 4. Registrar turno actual en el historial de memoria
        self.memoria.agregar_mensaje("user", consulta)
        self.memoria.agregar_mensaje("assistant", respuesta)

        return {
            "consulta_original": consulta,
            "consulta_busqueda": consulta_busqueda,
            "respuesta": respuesta,
            "contexto_utilizado": contexto,
            "evidencias": evidencias,
        }
