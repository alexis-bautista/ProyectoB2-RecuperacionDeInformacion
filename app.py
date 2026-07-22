"""
Interfaz conversacional web para el sistema RAG Multimodal.

Responsabilidades:
  - Re-ranking (cross-encoder ms-marco-MiniLM-L-6-v2)
  - Query Expansion con LLM (Gemini)
  - Relevance Feedback con algoritmo de Rocchio
  - Memoria conversacional (Query Condensation)
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from construir_embeddings import ConstructorEmbeddings
from base_datos_vectorial import BaseDatosVectorial
from sistema_rag import SistemaRAG
from reranker import ReRanker
from query_expansion import QueryExpander
from relevance_feedback import RelevanceFeedbackSystem
from memoria import ConversationalMemory

# Configuracion de la pagina
st.set_page_config(
    page_title="Asistente RAG Multimodal",
    page_icon="shopping_bags",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.badge-active {
    display: inline-block; background: linear-gradient(135deg,#6366f1,#8b5cf6);
    color: white; font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; margin: 2px; vertical-align: middle;
}
.badge-inactive {
    display: inline-block; background: rgba(100,116,139,0.3); color: #94a3b8;
    font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; margin: 2px; vertical-align: middle;
}
.score-pill {
    display: inline-block; background: rgba(16,185,129,0.15); color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.3); font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin-right: 6px;
}
.rank-pill {
    display: inline-block; background: rgba(99,102,241,0.15); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.3); font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin-right: 6px;
}
.rerank-pill {
    display: inline-block; background: rgba(245,158,11,0.15); color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.3); font-size: 0.72rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin-right: 6px;
}
.expanded-query {
    background: rgba(139,92,246,0.1); border-left: 3px solid #8b5cf6;
    padding: 4px 10px; margin: 3px 0; border-radius: 0 6px 6px 0;
    font-size: 0.85rem; color: #c4b5fd;
}
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size: 2rem; font-weight: 700; margin-bottom: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_recursos_base():
    constructor = ConstructorEmbeddings()
    bd_vectorial = BaseDatosVectorial()

    # Construir la ruta absoluta de manera dinámica
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_vector_store = os.path.join(ruta_base, "vector_store")

    # Cargar usando la ruta completa (ej: /mount/src/tu-repo/vector_store)
    bd_vectorial.cargar(ruta_vector_store)

    return constructor, bd_vectorial


@st.cache_resource
def cargar_reranker():
    return ReRanker(modelo_nombre="cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuracion del Sistema")
    st.markdown("---")

    st.markdown("### 🎯 Re-ranking")
    usar_reranking = st.toggle("Activar Re-ranking", value=True, key="toggle_reranking")
    if usar_reranking:
        st.caption("Cross-encoder `ms-marco-MiniLM-L-6-v2`")
        factor_candidatos = st.slider(
            "Factor de candidatos",
            2,
            8,
            4,
            key="factor_cand",
            help="top_k x factor candidatos antes de re-ordenar",
        )
    else:
        factor_candidatos = 4
    st.markdown("---")

    st.markdown("### 🔍 Expansion de Consultas")
    usar_expansion = st.toggle(
        "Activar Query Expansion", value=True, key="toggle_expansion"
    )
    if usar_expansion:
        st.caption("Genera 3 variaciones con Gemini (Multi-Query)")
    st.markdown("---")

    st.markdown("### 🧠 Memoria Conversacional")
    usar_memoria = st.toggle("Activar Memoria", value=True, key="toggle_memoria")
    if usar_memoria:
        st.caption("Reformula preguntas de seguimiento con contexto.")
    if st.button("🗑️ Limpiar historial", key="btn_limpiar"):
        st.session_state.mensajes = []
        if "memoria_obj" in st.session_state:
            st.session_state.memoria_obj.vaciar()
        st.rerun()
    st.markdown("---")

    st.markdown("### 📌 Relevance Feedback (Rocchio)")
    usar_rocchio = st.toggle("Activar Rocchio", value=True, key="toggle_rocchio")
    if usar_rocchio:
        st.caption("Ajusta la consulta vectorialmente con 👍/👎.")
        alpha = st.slider(
            "α (peso consulta original)", 0.0, 1.5, 1.0, 0.05, key="alpha"
        )
        beta = st.slider("β (documentos relevantes)", 0.0, 1.5, 0.75, 0.05, key="beta")
        gamma = st.slider(
            "γ (documentos no relevantes)", 0.0, 1.0, 0.25, 0.05, key="gamma"
        )
    else:
        alpha, beta, gamma = 1.0, 0.75, 0.25
    st.markdown("---")

    st.markdown("### 🔧 Busqueda")
    top_k = st.slider("Top-K resultados", 1, 10, 5, key="top_k")
    st.markdown("---")

    if "feedback_obj" in st.session_state:
        total_fb = sum(
            len(v["likes"]) + len(v["dislikes"])
            for v in st.session_state.feedback_obj.historial_feedback.values()
        )
        st.markdown(f"**Feedback registrado:** `{total_fb}` calificaciones")
        if total_fb > 0 and st.button("🔄 Resetear feedback", key="btn_reset_fb"):
            st.session_state.feedback_obj = RelevanceFeedbackSystem(
                alpha=alpha, beta=beta, gamma=gamma
            )
            st.rerun()


# ── Inicializar estado de sesion ──────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "feedback_obj" not in st.session_state:
    st.session_state.feedback_obj = RelevanceFeedbackSystem(
        alpha=alpha, beta=beta, gamma=gamma
    )

if "memoria_obj" not in st.session_state:
    st.session_state.memoria_obj = ConversationalMemory()


# ── Sistema compuesto ─────────────────────────────────────────────────────────
class SistemaCompuesto(SistemaRAG):
    def __init__(
        self,
        constructor,
        bd_vectorial,
        api_key,
        reranker,
        factor_candidatos,
        usar_expansion,
        usar_rocchio,
        feedback_system,
        usar_memoria,
        memoria,
    ):
        super().__init__(constructor, bd_vectorial, api_key, "gemini-3.5-flash-lite")
        self.reranker = reranker
        self.factor_candidatos = factor_candidatos
        self.usar_expansion = usar_expansion
        self.usar_rocchio = usar_rocchio
        self.feedback_system = feedback_system
        self.usar_memoria = usar_memoria
        self.memoria = memoria
        self._expandidas = []
        if usar_expansion:
            self._expansor = QueryExpander(
                api_key=self.api_key, modelo_llm=self.modelo_llm_nombre
            )

    def recuperar_evidencias(self, consulta, top_k=5):
        # 1. Embedding base
        emb = self.constructor.generar_embedding_consulta(consulta)

        # 2. Rocchio: ajustar vector
        if self.usar_rocchio and self.feedback_system:
            emb = self.feedback_system.ajustar_vector_consulta(
                consulta, emb, self.bd_vectorial
            )

        n_cand = (top_k * self.factor_candidatos) if self.reranker else top_k

        # 3. Query Expansion
        self._expandidas = []
        if self.usar_expansion:
            consultas_exp = self._expansor.expandir_consulta(consulta)
            self._expandidas = consultas_exp[1:]
            fusionados = {}
            for res in self.bd_vectorial.recuperar_top_k(emb, top_k=n_cand):
                pid = res["product_id"]
                if pid not in fusionados or res["score"] > fusionados[pid]["score"]:
                    fusionados[pid] = res
            for q in consultas_exp[1:]:
                emb_q = self.constructor.generar_embedding_consulta(q)
                for res in self.bd_vectorial.recuperar_top_k(emb_q, top_k=n_cand):
                    pid = res["product_id"]
                    if pid not in fusionados or res["score"] > fusionados[pid]["score"]:
                        fusionados[pid] = res
            candidatos = sorted(
                fusionados.values(), key=lambda x: x["score"], reverse=True
            )
        else:
            candidatos = self.bd_vectorial.recuperar_top_k(emb, top_k=n_cand)

        # 4. Re-ranking
        if self.reranker and candidatos:
            return self.reranker.rerank(consulta, candidatos, top_k=top_k)
        else:
            return [{**c, "rank": i + 1} for i, c in enumerate(candidatos[:top_k])]

    def responder_consulta(self, consulta, top_k=5):
        consulta_busqueda = consulta

        # Condensacion con memoria
        if self.usar_memoria and self.memoria and self.modelo_llm:
            hist = self.memoria.obtener_historial_formateado()
            if hist:
                try:
                    prompt_cond = f"""Rewrite the follow-up query as a standalone English search query using the conversation history.

Conversation History:
{hist}

Follow-up Query: {consulta}

Standalone Search Query:"""
                    r = self.modelo_llm.generate_content(prompt_cond)
                    consulta_busqueda = r.text.strip().replace('"', "").replace("'", "")
                except Exception as e:
                    print(f"Error reformulando: {e}")

        evidencias = self.recuperar_evidencias(consulta_busqueda, top_k=top_k)
        contexto = self.construir_contexto(evidencias)

        if self.usar_memoria and self.memoria:
            hist = self.memoria.obtener_historial_formateado()
            prompt = f"""Eres un asistente de compras conversacional con memoria.
Usa el historial y el contexto de productos para responder.

Historial:
{hist}

Contexto de Productos:
{contexto}

Nueva Consulta: {consulta}
Respuesta:"""
        else:
            prompt = f"""Eres un asistente de compras. Responde usando UNICAMENTE el contexto dado.

Contexto:
{contexto}

Consulta: {consulta}
Respuesta:"""

        respuesta = ""
        if self.modelo_llm:
            try:
                respuesta = self.modelo_llm.generate_content(prompt).text
            except Exception as e:
                respuesta = f"Error: {e}"
        else:
            respuesta = f"[Simulacion] {len(evidencias)} productos encontrados para '{consulta_busqueda}'."

        if self.usar_memoria and self.memoria:
            self.memoria.agregar_mensaje("user", consulta)
            self.memoria.agregar_mensaje("assistant", respuesta)

        return {
            "consulta": consulta,
            "consulta_busqueda": consulta_busqueda,
            "respuesta": respuesta,
            "evidencias": evidencias,
            "consultas_expandidas": self._expandidas,
        }


# ── Cargar recursos ───────────────────────────────────────────────────────────
constructor, bd_vectorial = cargar_recursos_base()
reranker_obj = cargar_reranker() if usar_reranking else None

api_key = os.environ.get("GEMINI_API_KEY")

sistema_rag = SistemaCompuesto(
    constructor=constructor,
    bd_vectorial=bd_vectorial,
    api_key=api_key,
    reranker=reranker_obj,
    factor_candidatos=factor_candidatos,
    usar_expansion=usar_expansion,
    usar_rocchio=usar_rocchio,
    feedback_system=st.session_state.feedback_obj,
    usar_memoria=usar_memoria,
    memoria=st.session_state.memoria_obj,
)

# ── Encabezado ────────────────────────────────────────────────────────────────
col_title, col_badges = st.columns([3, 2])
with col_title:
    st.markdown(
        '<p class="main-header">🛍️ Asistente RAG Multimodal</p>', unsafe_allow_html=True
    )
    st.caption(
        "CLIP · FAISS · Gemini · Re-ranking · Query Expansion · Rocchio · Memoria"
    )
with col_badges:
    b = ""
    b += f'<span class="{"badge-active" if usar_reranking else "badge-inactive"}">🎯 Re-ranking</span>'
    b += f'<span class="{"badge-active" if usar_expansion else "badge-inactive"}">🔍 Expansion</span>'
    b += f'<span class="{"badge-active" if usar_rocchio else "badge-inactive"}">📌 Rocchio</span>'
    b += f'<span class="{"badge-active" if usar_memoria else "badge-inactive"}">🧠 Memoria</span>'
    st.markdown(f"<div style='padding-top:18px'>{b}</div>", unsafe_allow_html=True)

st.markdown("---")


# ── Historial ─────────────────────────────────────────────────────────────────
def render_evidencias(evidencias, consulta_original, prefix):
    with st.expander(f"📄 {len(evidencias)} evidencias recuperadas", expanded=False):
        for j, ev in enumerate(evidencias):
            col_img, col_info = st.columns([1, 4])
            with col_img:
                if ev.get("image_url"):
                    st.image(ev["image_url"], use_container_width=True)
            with col_info:
                rank = ev.get("rank", j + 1)
                score_orig = ev.get("score_original", ev.get("score", 0.0))
                score_rr = ev.get("score_reranking")
                pills = f'<span class="rank-pill">#{rank}</span>'
                pills += f'<span class="score-pill">CLIP: {score_orig:.4f}</span>'
                if score_rr is not None:
                    pills += f'<span class="rerank-pill">Re-rank: {score_rr:.3f}</span>'
                st.markdown(pills, unsafe_allow_html=True)
                st.markdown(f"**ID:** `{ev['product_id']}`")
                st.markdown(
                    ev["texto"][:250] + ("..." if len(ev["texto"]) > 250 else "")
                )
                if usar_rocchio:
                    fb1, fb2, _ = st.columns([1, 1, 3])
                    with fb1:
                        if st.button("👍", key=f"{prefix}_like_{j}"):
                            st.session_state.feedback_obj.registrar_feedback(
                                consulta_original, ev["product_id"], es_relevante=True
                            )
                            st.toast(f"✅ Like para `{ev['product_id']}`")
                    with fb2:
                        if st.button("👎", key=f"{prefix}_dislike_{j}"):
                            st.session_state.feedback_obj.registrar_feedback(
                                consulta_original, ev["product_id"], es_relevante=False
                            )
                            st.toast(f"❌ Dislike para `{ev['product_id']}`")
            st.markdown("---")


for i, mensaje in enumerate(st.session_state.mensajes):
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if mensaje["rol"] == "assistant":
            if mensaje.get("consultas_expandidas"):
                with st.expander("🔍 Consultas expandidas", expanded=False):
                    for qe in mensaje["consultas_expandidas"]:
                        st.markdown(
                            f'<div class="expanded-query">↳ {qe}</div>',
                            unsafe_allow_html=True,
                        )
            if mensaje.get("evidencias"):
                render_evidencias(
                    mensaje["evidencias"], mensaje.get("consulta", ""), f"hist_{i}"
                )


# ── Nueva consulta ────────────────────────────────────────────────────────────
if consulta := st.chat_input(
    "Escribe tu consulta (ej: red running shoes for women)..."
):
    st.chat_message("user").markdown(consulta)
    st.session_state.mensajes.append({"rol": "user", "contenido": consulta})

    with st.chat_message("assistant"):
        with st.spinner("🔎 Procesando con todos los modulos activos..."):
            resultado = sistema_rag.responder_consulta(consulta, top_k=top_k)

        consultas_exp = resultado.get("consultas_expandidas", [])
        if consultas_exp:
            with st.expander("🔍 Consultas expandidas por Gemini", expanded=True):
                for qe in consultas_exp:
                    st.markdown(
                        f'<div class="expanded-query">↳ {qe}</div>',
                        unsafe_allow_html=True,
                    )

        cb = resultado.get("consulta_busqueda", consulta)
        if cb != consulta:
            st.info(f"🧠 **Memoria reformulo la consulta:** _{cb}_")

        st.markdown(resultado["respuesta"])

        evidencias = resultado["evidencias"]
        with st.expander(f"📄 {len(evidencias)} evidencias recuperadas", expanded=True):
            for j, ev in enumerate(evidencias):
                col_img, col_info = st.columns([1, 4])
                with col_img:
                    if ev.get("image_url"):
                        st.image(ev["image_url"], use_container_width=True)
                with col_info:
                    rank = ev.get("rank", j + 1)
                    score_orig = ev.get("score_original", ev.get("score", 0.0))
                    score_rr = ev.get("score_reranking")
                    pills = f'<span class="rank-pill">#{rank}</span>'
                    pills += f'<span class="score-pill">CLIP: {score_orig:.4f}</span>'
                    if score_rr is not None:
                        pills += (
                            f'<span class="rerank-pill">Re-rank: {score_rr:.3f}</span>'
                        )
                    st.markdown(pills, unsafe_allow_html=True)
                    st.markdown(f"**ID:** `{ev['product_id']}`")
                    st.markdown(
                        ev["texto"][:250] + ("..." if len(ev["texto"]) > 250 else "")
                    )
                    if usar_rocchio:
                        fb1, fb2, _ = st.columns([1, 1, 3])
                        with fb1:
                            if st.button("👍", key=f"new_like_{j}"):
                                st.session_state.feedback_obj.registrar_feedback(
                                    consulta, ev["product_id"], es_relevante=True
                                )
                                st.toast(f"✅ Like para `{ev['product_id']}`")
                        with fb2:
                            if st.button("👎", key=f"new_dislike_{j}"):
                                st.session_state.feedback_obj.registrar_feedback(
                                    consulta, ev["product_id"], es_relevante=False
                                )
                                st.toast(f"❌ Dislike para `{ev['product_id']}`")
                st.markdown("---")

    st.session_state.mensajes.append(
        {
            "rol": "assistant",
            "contenido": resultado["respuesta"],
            "evidencias": resultado["evidencias"],
            "consulta": consulta,
            "consultas_expandidas": consultas_exp,
        }
    )
