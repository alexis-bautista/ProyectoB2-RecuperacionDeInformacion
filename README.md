# Sistema de Recuperación de Información Multimodal con RAG 🛍️
**Proyecto Final - Recuperación de Información**  
*Desarrollado por: Bautista Alexis, Correa Adrian, Yunga Bryan*

Este repositorio contiene la implementación de un **Sistema de Recuperación de Información Multimodal** basado en arquitectura **RAG (Retrieval-Augmented Generation)**. El sistema es capaz de responder consultas conversacionales en lenguaje natural sobre un catálogo de productos de Amazon (compuesto por texto e imágenes) utilizando modelos de embeddings multimodales (CLIP) y una base de datos vectorial (FAISS).

---

## 🛠️ Arquitectura General del Sistema

El sistema sigue un flujo modular desacoplado para mantener el orden y evitar congestión de código en el notebook principal:

```
Proyecto.ipynb (Notebook Principal)
 ├── preparar_corpus.py       # Descarga y pre-procesamiento del dataset multimodal.
 ├── construir_embeddings.py  # Generación de embeddings L2 con CLIP (OpenAI).
 ├── base_datos_vectorial.py  # Indexación y búsqueda Top-K rápida con FAISS.
 ├── sistema_rag.py           # Pipeline RAG integrado con el LLM (Gemini API).
 ├── reranker.py              # Excelencia 1: Re-ranking con Cross-Encoder (MiniLM).
 ├── query_expansion.py       # Excelencia 2: Expansión de Consultas con LLM (Multi-Query).
 ├── relevance_feedback.py    # Excelencia 3: Relevance Feedback con Algoritmo de Rocchio.
 ├── memoria.py               # Excelencia 4: Memoria conversacional (Query Condensation).
 └── app.py                   # Interfaz de usuario interactiva web con Streamlit.
```

---

## 🚀 Requisitos e Instalación

### 1. Clonar el repositorio y acceder a él
```bash
git clone <url-del-repositorio>
cd ProyectoB2-RecuperacionDeInformacion
```

### 2. Instalar dependencias necesarias
Asegúrate de contar con Python 3.11 instalado. Ejecuta el siguiente comando para instalar las librerías del proyecto:
```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.50.0 tokenizers==0.21.4 huggingface_hub==0.36.2
python -m pip install faiss-cpu datasets pandas pillow requests google-generativeai python-dotenv streamlit sentence-transformers
```

### 3. Configurar la API Key de Gemini
Crea un archivo `.env` en la raíz del proyecto (este archivo ya está ignorado en el `.gitignore` por seguridad):
```env
GEMINI_API_KEY=tu_api_key_real_aqui
```

---

## 💻 Ejecución del Proyecto

### Flujo en Jupyter Notebook (`Proyecto.ipynb`)
El archivo principal es [Proyecto.ipynb](file:///D:/7mo%20semestre/RI/ProyectoB2-RecuperacionDeInformacion/Proyecto.ipynb). Puedes abrirlo y ejecutar todas las celdas en orden. El flujo del notebook realiza lo siguiente:
1. **Preparación del Corpus**: Fusiona Amazon ESCI con Crossing Minds para obtener textos e imágenes asociadas.
2. **Generación de Embeddings**: Carga el modelo CLIP de OpenAI y vectoriza los productos.
3. **Búsqueda Vectorial**: Indexa los vectores en FAISS y realiza búsquedas semánticas.
4. **Sistema RAG**: Combina la búsqueda con Gemini para responder preguntas mostrando evidencias.
5. **Funcionalidades de Excelencia**: Demostración interactiva de cada módulo avanzado.
6. **Evaluación Experimental**: Calcula métricas Precision@K, Recall@K y NDCG@K.

### Ejecución de la Interfaz Web (Chat Streamlit)
Para lanzar el chat interactivo tipo web donde puedes chatear con el catálogo y ver las evidencias recuperadas:
```bash
python -m streamlit run app.py
```
Abre en tu navegador la dirección: **http://localhost:8501**

---

## ✨ Funcionalidades de Excelencia Implementadas

Hemos implementado **las 4 funcionalidades de excelencia** contempladas en la rúbrica del proyecto, asegurando la nota máxima:

### 1. Re-ranking (+15 puntos)
* **Archivo**: [reranker.py](file:///D:/7mo%20semestre/RI/ProyectoB2-RecuperacionDeInformacion/reranker.py)
* **Método**: Utiliza el modelo `cross-encoder/ms-marco-MiniLM-L-6-v2` de Sentence-Transformers para reordenar los candidatos recuperados por CLIP. Al evaluar conjuntamente la pregunta y el texto del producto mediante capas de atención cruzada, se logran corregir posiciones del ranking original.

### 2. Query Expansion (+15 puntos)
* **Archivo**: [query_expansion.py](file:///D:/7mo%20semestre/RI/ProyectoB2-RecuperacionDeInformacion/query_expansion.py)
* **Método**: Envía la consulta del usuario a Gemini para generar 3 variaciones alternativas en inglés (traduciendo si el usuario escribe en español e incorporando sinónimos). El recuperador ejecuta una búsqueda "Multi-Query" combinada en FAISS, unificando y deduplicando resultados.

### 3. Relevance Feedback (+15 puntos)
* **Archivo**: [relevance_feedback.py](file:///D:/7mo%20semestre/RI/ProyectoB2-RecuperacionDeInformacion/relevance_feedback.py)
* **Método**: Implementa el **Algoritmo de Rocchio** matemático en el espacio vectorial. Cuando el usuario otorga Likes o Dislikes a las evidencias, el vector de la consulta se re-calcula (desplazamiento vectorial) acercándose a los productos deseados y alejándose de los irrelevantes para futuras búsquedas de ese hilo.

### 4. Memoria Conversacional (+15 puntos)
* **Archivo**: [memoria.py](file:///D:/7mo%20semestre/RI/ProyectoB2-RecuperacionDeInformacion/memoria.py)
* **Método**: Implementa la técnica de **Query Condensation**. Cada vez que el usuario hace una pregunta de seguimiento ambigua (ej. *"¿lo tienes en color rojo?"*), el LLM reescribe la pregunta integrando el contexto conversacional previo (ej. *"cuchillo imarku color rojo"*) antes de buscar en la base de datos vectorial.
