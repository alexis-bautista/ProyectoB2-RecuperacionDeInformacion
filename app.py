"""
Interfaz conversacional web para el sistema RAG Multimodal.

Responsabilidades:
  - Proveer una interfaz web básica tipo chat.
  - Recibir consultas conversacionales del usuario.
  - Mostrar la respuesta generada por el asistente.
  - Visualizar los documentos e imágenes utilizados como contexto.
"""

import streamlit as st
from construir_embeddings import ConstructorEmbeddings
from base_datos_vectorial import BaseDatosVectorial
from sistema_rag import SistemaRAG

# Configuración de la página
st.set_page_config(page_title="Asistente RAG Multimodal", page_icon="🛍️", layout="wide")


@st.cache_resource
def inicializar_sistema():
    """Carga los modelos y la base de datos solo una vez en caché."""
    with st.spinner("Cargando modelos y base de datos vectorial..."):
        constructor = ConstructorEmbeddings()
        bd_vectorial = BaseDatosVectorial()
        bd_vectorial.cargar("vector_store")
        sistema = SistemaRAG(constructor, bd_vectorial)
        return sistema


# Inicializar el backend
sistema_rag = inicializar_sistema()

st.title("Asistente de Compras Multimodal")
st.markdown(
    "Consulta nuestro catálogo y te recomendaré los mejores productos basándome en su texto e imágenes."
)

# Inicializar el historial de chat en el estado de la sesión
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Renderizar el historial de mensajes
for mensaje in st.session_state.mensajes:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

        # Si el mensaje del asistente tiene evidencias, las mostramos
        if "evidencias" in mensaje and mensaje["evidencias"]:
            with st.expander(
                "Visualizar contexto y evidencias utilizadas", expanded=False
            ):
                for ev in mensaje["evidencias"]:
                    st.markdown("---")
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if ev.get("image_url"):
                            st.image(ev["image_url"], use_container_width=True)
                    with col2:
                        st.markdown(
                            f"**Ranking:** #{ev['rank']} | **Score:** {ev['score']:.4f}"
                        )
                        st.markdown(f"**ID del Producto:** `{ev['product_id']}`")
                        st.markdown(f"{ev['texto']}")

# Capturar nueva consulta del usuario
if consulta := st.chat_input(
    "Escribe tu consulta conversacional aquí (ej. I am looking for red running shoes)..."
):
    # Mostrar y guardar el mensaje del usuario
    st.chat_message("user").markdown(consulta)
    st.session_state.mensajes.append({"rol": "user", "contenido": consulta})

    # Mostrar "escribiendo..." mientras se procesa
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el catálogo y generando respuesta..."):
            # Ejecutar el pipeline RAG
            resultado = sistema_rag.responder_consulta(consulta, top_k=3)

            # Mostrar la respuesta principal
            st.markdown(resultado["respuesta"])

            # Mostrar las evidencias inmediatamente debajo
            st.markdown("### Evidencias Recuperadas")
            with st.expander(
                "Inspeccionar documentos e imágenes del contexto", expanded=True
            ):
                for ev in resultado["evidencias"]:
                    st.markdown("---")
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if ev.get("image_url"):
                            st.image(ev["image_url"], use_container_width=True)
                    with col2:
                        st.markdown(
                            f"**Ranking:** #{ev['rank']} | **Score:** {ev['score']:.4f}"
                        )
                        st.markdown(f"**ID del Producto:** `{ev['product_id']}`")
                        st.markdown(f"{ev['texto']}")

    # Guardar la respuesta y sus evidencias en el historial
    st.session_state.mensajes.append(
        {
            "rol": "assistant",
            "contenido": resultado["respuesta"],
            "evidencias": resultado["evidencias"],
        }
    )
