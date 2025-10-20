# app/models/rag.py
import logging
import json
from typing import List, Dict, Any, Optional
from app.clients.llm_base import LLMClientProtocol

logger = logging.getLogger(__name__)


class RAGService:
    """
    Servicio RAG orquestador:
      - retriever: objeto con .retrieve(query, top_k) -> (indices, distances) y .get_product(idx)
      - reranker: objeto con .rerank(query, candidate_texts, candidate_indices, top_m)
      - llm_client: adaptador que cumple LLMClientProtocol (opcional)
    """
    def __init__(self, retriever, reranker=None, llm_client: Optional[LLMClientProtocol] = None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm_client

    def build_context(self, indices: List[int], max_chars: int = 3000) -> str:
        parts = []
        total = 0
        for idx in indices:
            p = self.retriever.get_product(idx)
            text = f"product_id: {p.get('product_id')}\nname: {p.get('product_name')}\n{p.get('product_description')}\n\n"
            if total + len(text) > max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n".join(parts)

    def _normalize_llm_response(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza la respuesta del LLM a la forma:
          { best_product_id: int|null, reasons: [str,...], top_candidates: [{product_id:int, score:float}, ...] }
        Si resp contiene 'raw', intentamos extraer JSON; si no, devolvemos fallback.
        """
        if not resp:
            return {"best_product_id": None, "reasons": [], "top_candidates": []}

        if "best_product_id" in resp and "reasons" in resp and "top_candidates" in resp:
            return resp

        # if adapter returned raw text, try to parse json inside it
        raw = resp.get("raw")
        if raw:
            import re
            m = re.search(r'\{.*\}', raw, flags=re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    return parsed
                except Exception:
                    logger.warning("Could not parse JSON from raw LLM output.")
        # fallback minimal structured response
        return {"best_product_id": None, "reasons": ["no structured LLM output"], "top_candidates": []}

    def answer(self, query: str, top_k: int = 50, rerank_top: int = 5) -> Dict[str, Any]:
        """
        Ejecuta pipeline:
          1. retrieve top_k candidatos
          2. (opcional) rerank -> toma rerank_top
          3. construir contexto y llamar al LLM adapter si existe
          4. parsear respuesta y devolver estructura consistente
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        # 1) retrieve
        idxs, dists = self.retriever.retrieve(query, top_k=top_k)
        if not idxs:
            return {"best_product_id": None, "reasons": ["no candidates"], "top_candidates": []}

        # 2) rerank (si existe)
        if self.reranker:
            candidate_texts = [
                f"{self.retriever.product_df.iloc[i]['product_name']} - {self.retriever.product_df.iloc[i]['product_description']}"
                for i in idxs
            ]
            try:
                idxs, scores = self.reranker.rerank(query, candidate_texts, idxs, top_m=rerank_top)
            except Exception as e:
                logger.exception("Reranker failed, continuing with original order: %s", e)

        # 3) context
        context = self.build_context(idxs[:rerank_top], max_chars=3500)

        # 4) call LLM if present
        if self.llm:
            try:
                raw_resp = self.llm.generate_answer(query=query, context=context)
                normalized = self._normalize_llm_response(raw_resp)
                # Try to enrich best_product metadata if best_product_id is an index into product_df
                try:
                    bpid = normalized.get("best_product_id")
                    if bpid is not None:
                        # If product_id in dataset is an int stored as string, convert and search
                        # We assume best_product_id corresponds to product_id (not row idx), try to find row
                        df = self.retriever.product_df
                        match = df.loc[df['product_id'].astype(str) == str(bpid)]
                        if not match.empty:
                            row = match.iloc[0]
                            normalized["_best_product_meta"] = {
                                "product_id": row['product_id'],
                                "product_name": row.get('product_name'),
                                "product_description": row.get('product_description')
                            }
                except Exception:
                    logger.debug("Could not enrich LLM best_product metadata (non-fatal).")
                return normalized
            except Exception as e:
                logger.exception("LLM generation failed; falling back: %s", e)

        # fallback deterministic response using top candidate indices
        top_idx = idxs[0]
        best = self.retriever.get_product(top_idx)
        top_candidates = []
        for i in idxs[:rerank_top]:
            try:
                pid = int(self.retriever.product_df.iloc[i]['product_id'])
            except Exception:
                pid = self.retriever.product_df.iloc[i]['product_id']
            top_candidates.append({"product_id": pid, "score": None})
        return {"best_product_id": best.get("product_id"), "reasons": ["fallback: top match"], "top_candidates": top_candidates}
