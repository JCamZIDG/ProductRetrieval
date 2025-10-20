# Retrieval & RAG Service — Product Search Evaluation

Proyecto entregable: evaluación e implementación de un sistema de recuperación de productos (retrieval) con mejora por reranking y un microservicio RAG (opcional LLM local).  
Contiene: (1) notebook exploratorio donde se desarrolló/ evaluó la solución; (2) microservicio FastAPI que expone búsqueda, reranking y RAG.

---

## Resumen

Este proyecto implementa y evalúa pipelines de búsqueda de productos sobre el dataset WANDS:

- **Baseline lexical**: TF-IDF + lematización (spaCy).
- **Baseline semántico**: embeddings (Sentence-Transformers `all-MiniLM-L6-v2`) + FAISS.
- **Reranking**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- **RAG (opcional)**: Integración con LLM local usando `gpt4all` para generar explicaciones / justificar la selección (no obligatorio para que el servicio funcione; si el LLM no está disponible, se usa fallback determinístico).
- Métricas de evaluación incluidas: **MAP@10**, **Weighted MAP@10**, **NDCG@10**, **Precision@10**, **Recall@10**, plus bootstrap paired test.

Objetivo de la entrega: mejorar MAP@10 sobre el baseline y refactorizar el pipeline como microservicio listo para pruebas.

---

## Estructura del repositorio

```
RetrievalBestProductsMatch/
├─ app/
│ ├─ init.py
│ ├─ config.py # carga de env vars (pydantic)
│ ├─ logger.py # configuración logging
│ ├─ models/
│ │ ├─ retriever.py # clase Retriever (FAISS + embeddings)
│ │ ├─ reranker.py # clase Reranker (CrossEncoder)
│ │ └─ rag.py # clase RAGService (build context + call LLM/fallback)
│ ├─ clients/
│ │ ├─ gpt4all_client.py # Adaptador para gpt4all
│ │ └─ llm_base.py # Interfaz LLMClientProtocol
│ ├─ utils/
│ │ └─ llm_loader.py # Loader que instancia GPT4All con descarga opcional
│ ├─ data/ # No incluir datos privados en el repo
│ │ ├─ faiss.index # índice FAISS (generado por el notebook)
│ │ ├─ label.csv # (opcional local para pruebas)
│ │ ├─ query.csv
│ │ └─ product.csv
│ ├─ schemas.py # pydantic request/response models
│ └─ utils.py # helpers (metrics, parsing)
├─ tests/ # tests unitarios (opcional)
├─ main.py # FastAPI app + endpoints
```

notebook.ipynb # notebook con todo el desarrollo/evaluación
requirements.txt
README.md

## Requisitos

El proyecto fue desarrollado con las dependencias listadas en `requirements.txt`. Para instalar:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Notebook: reproducir experimentos

1. Abre `AIEngineer_retrieval_assignment_JuanCamarena.ipynb` en Jupyter / Colab (preferible ejecutar local con GPU si la tienes).

2. Asegúrate de colocar `product.csv`, `query.csv`, `label.csv` en `app/data/` o actualizar las rutas.

3. Ejecuta todas las celdas (restart & run all) para:
   - generar embeddings,
   - construir y guardar `data/faiss.index`,
   - evaluar baselines (TF-IDF) y pipelines semántico / rerank,
   - ejecutar grid-search para `partial_w` y bootstrap paired test.

El notebook incluye celdas Markdown con justificación, resultados y visualizaciones (gráficas comparativas, top queries, etc).

**Importante:** el notebook genera `data/faiss.index`. Si vas a levantar el microservicio, asegúrate de que `FAISS_INDEX_PATH` apunte a ese archivo (o genera el índice desde el notebook primero).

---

## Microservicio (FastAPI)

### Ejecutar localmente
```bash
uvicorn app.main:app --reload 
```

## 🚀 Ejecución del Microservicio (FastAPI)

Por defecto el servidor arranca en:  
👉 **http://127.0.0.1:8000**


## 🧠 Configurar y usar LLM local (gpt4all)

Si deseas usar RAG con un LLM local:

- Coloca el archivo `.gguf` en una ruta accesible, **o**  
- Deja `LLM_ALLOW_DOWNLOAD=true` para que **gpt4all** lo descargue al caché del usuario en la primera ejecución.

⚠️ **Advertencia:**  
Los modelos grandes (7B, 8B) ocupan varios GB y la descarga puede tardar o fallar en redes lentas.  
También requieren bastante RAM/GPU.

Si **gpt4all** no puede cargar el modelo, el servicio seguirá funcionando **sin LLM**.

---

## 📊 Resultados clave (experimentales)

Comparativa entre **Semantic (FAISS)** y **Rerank (FAISS → Cross-Encoder)**:

### 🔹 Semantic (FAISS top-10) — promedios
| Métrica | Valor |
|----------|--------|
| MAP@10 (exact only) | 0.3353 |
| Weighted MAP@10 | 0.5342 |
| NDCG@10 (graded) | 0.8713 |
| Precision@10 | 0.3223 |
| Recall@10 | 0.1729 |

---

### 🔹 Rerank (FAISS → CrossEncoder → top-10) — promedios
| Métrica | Valor |
|----------|--------|
| MAP@10 (exact only) | 0.4411 |
| Weighted MAP@10 | 0.5971 |
| NDCG@10 (graded) | 0.8942 |
| Precision@10 | 0.3769 |
| Recall@10 | 0.2213 |

---

**Bootstrap paired (Weighted MAP):**  
- mean difference = **0.062882**  
- 95% CI = **(0.048660, 0.077632)** → mejora estadísticamente significativa ✅

---

## ⚙️ Grid Search `partial_w`

Valores probados: `[0.2, 0.4, 0.5, 0.7, 0.9]`  
→ Se usó `partial_w = 0.5` por defecto como compromiso razonable.

---

## 💡 ¿Por qué incluir *Weighted MAP* (Partial matches)?

El dataset etiqueta coincidencias como **Exact** y **Partial**.  
Tratar `Partial` como irrelevante penaliza demasiado el sistema, cuando el resultado puede ser útil aunque no sea idéntico.

**Weighted MAP** permite reconocer grados de relevancia:

- Exact = 1.0  
- Partial = `partial_w`

Esto da una evaluación más realista.

## Proximo Pasos y Desarrollo del Proyecto

Dado el tiempo limitado, el proceso del desarrollo del microservicio fue en su mayoría un proceso de vibe coding en el que se llevó lo creado lo del notebook al microservicio lo más rápido posible sin perder los resultados obtenidos. De lo cual, desde mi punto de vista personal se logró el objetivo.

Al microservicio de todas maneras se le podrían hacer muchas mejoras de todas maneras, los logs podrían ser aún mejores; así como también el manejo de errores se podría ver en una mayor magnitud implementando tests o mocking para ver realmente que tan efectivo es el manejo de errores del producto final.

En cuanto al modelo, dado que ya se logró probar la efectividad con un modelo con mas contexto, tal vez el siguiente camino podría ser usar un modelo de NLP más pesado o más enfocado al tipo de requerimiento tenemos, y no uno tan "general"; dado que este producto estaba enfocado precisamente en la comparación de productos a la venta. Tal vez, mejorar el preprocesamiento podría llevarnos a aún mejores resultados también.

Muchas gracias por la evaluación, fue muy divertida de realizar.

Juan Camarena

PD: Terminé modificando las versiones de algunas librerías (pandas) porque demorarme en temas de compatibilidad por la limitación del tiempo podría haberme quitado tiempo valioso que invertir en ideas.