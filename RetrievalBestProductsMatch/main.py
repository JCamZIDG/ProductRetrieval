# app/main.py
from fastapi import FastAPI, HTTPException
import logging

from app.logger import setup_logging
from app.config import settings
from app.models.retriever import Retriever
from app.models.reranker import Reranker
from app.models.rag import RAGService
from app.schemas import SearchRequest, SearchResponse
from app.utils.llm_loader import load_local_gpt4all_adapter  # tu loader aquí

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Retrieval RAG Service")

# Instantiate model objects (but don't perform heavy loads yet)
retriever = Retriever(settings.FAISS_INDEX_PATH, settings.PRODUCT_CSV, settings.EMBED_MODEL)
reranker = Reranker(settings.RERANKER_MODEL)
rag_service: RAGService | None = None
llm_client = None

@app.on_event("startup")
def startup_event():
    global rag_service, llm_client
    try:
        logger.info("Loading retriever...")
        retriever.load()
        logger.info("Loading reranker...")
        reranker.load()

        # Load LLM adapter (non-fatal if fails). Use settings values.
        llm_client = None
        if settings.LLM_MODEL_NAME and str(settings.LLM_MODEL_NAME).lower() != "none":
            logger.info("Attempting to load LLM adapter for model name: %s", settings.LLM_MODEL_NAME)
            # allow_download comes from config (True to allow download)
            llm_client = load_local_gpt4all_adapter(settings.LLM_MODEL_NAME, allow_download=settings.LLM_ALLOW_DOWNLOAD)
            if llm_client is None:
                logger.warning("LLM adapter not available; /rag will fallback to deterministic responses.")
            else:
                logger.info("LLM adapter loaded.")
        else:
            logger.info("No LLM model configured; skipping LLM load.")

        rag_service = RAGService(retriever=retriever, reranker=reranker, llm_client=llm_client)
        logger.info("Service started.")
    except Exception as e:
        logger.exception("Startup failed: %s", e)
        raise

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        idxs, dists = retriever.retrieve(req.query, top_k=req.top_k)
        if req.use_rerank:
            candidate_texts = [
                retriever.product_df.iloc[i]['product_name'] + " - " + str(retriever.product_df.iloc[i]['product_description'])
                for i in idxs
            ]
            idxs, scores = reranker.rerank(req.query, candidate_texts, idxs, top_m=req.rerank_m)
            dists = scores
        results = []
        for idx, score in zip(idxs[:req.rerank_m], dists[:req.rerank_m]):
            p = retriever.get_product(idx)
            results.append({
                "product_id": str(p['product_id']),
                "product_name": p['product_name'],
                "product_description": p['product_description'],
                "score": float(score)
            })
        return {"query": req.query, "results": results}
    except Exception as e:
        logger.exception("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
def rag(req: SearchRequest):
    try:
        answer = rag_service.answer(req.query, top_k=req.top_k, rerank_top=req.rerank_m)
        return answer
    except Exception as e:
        logger.exception("RAG error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))