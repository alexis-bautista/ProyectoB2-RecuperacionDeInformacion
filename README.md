# 🛍️ Sistema de Recuperación de Información Multimodal con RAG

**Proyecto Final - Recuperación de Información**  
**Escuela Politécnica Nacional**  
_Desarrollado por: Bautista Alexis, Correa Adrian, Yunga Bryan_

Este repositorio contiene la implementación de un **Sistema de Recuperación de Información Multimodal** basado en la arquitectura **RAG (Retrieval-Augmented Generation)**. El sistema es capaz de responder consultas conversacionales en lenguaje natural sobre un catálogo de productos de Amazon (compuesto por texto e imágenes) utilizando modelos de embeddings multimodales (CLIP) y una base de datos vectorial (FAISS).

---

## 🛠️ Arquitectura General del Sistema

El sistema sigue un diseño modular y desacoplado para mantener el orden, facilitar el mantenimiento y evitar la congestión de código en el notebook principal:

```text
📁 ProyectoB2-RecuperacionDeInformacion
 ├── 📓 Proyecto.ipynb               # Notebook Principal de presentación de resultados.
 ├── 🐍 preparar_corpus.py           # Descarga y pre-procesamiento del dataset multimodal.
 ├── 🐍 construir_embeddings.py      # Generación de embeddings L2 con CLIP.
 ├── 🐍 base_datos_vectorial.py      # Indexación y búsqueda Top-K rápida con FAISS.
 ├── 🐍 sistema_rag.py               # Pipeline RAG integrado con el LLM (Gemini API).
 ├── 🐍 reranker.py                  # Excelencia 1: Re-ranking con Cross-Encoder.
 ├── 🐍 query_expansion.py           # Excelencia 2: Expansión de Consultas (Multi-Query).
 ├── 🐍 relevance_feedback.py        # Excelencia 3: Feedback con Algoritmo de Rocchio.
 ├── 🐍 memoria.py                   # Excelencia 4: Memoria conversacional.
 ├── 🐍 app.py                       # Interfaz de usuario interactiva web (Streamlit).
 └── 📄 requirements.txt             # Dependencias del proyecto.

```

---

## 🚀 Requisitos e Instalación

### 1. Clonar el repositorio

```bash
git clone <https://github.com/alexis-bautista/ProyectoB2-RecuperacionDeInformacion.git>
cd ProyectoB2-RecuperacionDeInformacion

```

### 2. Instalar dependencias

Asegúrate de contar con **Python 3.11** instalado. Se recomienda utilizar un entorno virtual (`venv` o `conda`). Ejecuta el siguiente comando para instalar todas las librerías necesarias:

```bash
pip install -r requirements.txt

```

_(Nota: El archivo de requerimientos está configurado para instalar la versión de CPU de PyTorch por defecto)._

### 3. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto para alojar tus credenciales de forma segura. El archivo `.gitignore` ya está configurado para excluirlo del control de versiones:

```env
GEMINI_API_KEY=tu_api_key_real_aqui

```

---

## 💻 Ejecución del Proyecto

### Flujo en Jupyter Notebook

El punto de entrada principal para revisar la lógica y los resultados paso a paso es [`Proyecto.ipynb`]. El notebook estructura la ejecución en las siguientes fases:

1. **Preparación del Corpus:** Fusión de los datasets _Amazon ESCI_ y _Crossing Minds_ para alinear textos descriptivos e imágenes.
2. **Generación de Embeddings:** Uso del modelo CLIP pre-entrenado para proyectar el catálogo multimodal.
3. **Búsqueda Vectorial:** Indexación de vectores en FAISS y ejecución de búsquedas semánticas eficientes.
4. **Sistema RAG:** Integración de la recuperación con Gemini para sintetizar respuestas fundamentadas, exponiendo las evidencias visuales.
5. **Funcionalidades de Excelencia:** Pruebas modulares e interactivas de cada componente avanzado.
6. **Evaluación Experimental:** Ejecución de la validación del sistema sobre el conjunto de relevancia (_qrels_) diseñado para el análisis del pipeline.

### Ejecución de la Interfaz Web (Streamlit)

Para interactuar con el sistema a través de un chat web completo, que permite inspeccionar visualmente los documentos recuperados:

```bash
streamlit run app.py

```

Una vez iniciado, abre en tu navegador: **http://localhost:8501**

---

## ✨ Funcionalidades de Excelencia Implementadas

El sistema supera los requisitos base incorporando **4 funcionalidades de excelencia** para optimizar la recuperación y la experiencia del usuario:

### 1. Re-ranking

- **Módulo:** [`reranker.py`]
- **Descripción:** Utiliza el modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` de _Sentence-Transformers_ para reordenar los candidatos iniciales de CLIP. Evalúa semánticamente el par (consulta, documento) con capas de atención cruzada, refinando significativamente la precisión en el Top-K.

### 2. Query Expansion

- **Módulo:** [`query_expansion.py`]
- **Descripción:** Emplea el LLM para reformular la consulta inicial, generando variaciones y traduciéndola al inglés de ser necesario. Ejecuta una búsqueda vectorial _Multi-Query_, fusionando y deduplicando resultados para maximizar los aciertos.

### 3. Relevance Feedback

- **Módulo:** [`relevance_feedback.py`]
- **Descripción:** Implementa el **Algoritmo de Rocchio** en el espacio multidimensional. Ajusta dinámicamente el vector de la consulta en base a las calificaciones del usuario ("Me gusta" / "No me gusta"), desplazándolo hacia el centroide de los documentos relevantes y alejándolo del ruido semántico.

### 4. Memoria Conversacional

- **Módulo:** [`memoria.py`]
- **Descripción:** Integra el historial de la sesión mediante **Query Condensation**. Ante preguntas de seguimiento ambiguas (ej. _"¿tienes ese en talla 9?"_), el LLM reconstruye una consulta independiente con todo el contexto necesario antes de impactar contra la base de datos vectorial.
