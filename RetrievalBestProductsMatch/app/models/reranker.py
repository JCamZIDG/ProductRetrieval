from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    def load(self):
        logger.info("Cargando reranker %s", self.model_name)
        self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidate_texts: list, candidate_indices: list, top_m: int = 10):
        inputs = [[query, txt] for txt in candidate_texts]
        scores = self.model.predict(inputs)
        ranked = sorted(range(len(candidate_indices)), key=lambda i: scores[i], reverse=True)
        ranked_indices = [candidate_indices[i] for i in ranked[:top_m]]
        ranked_scores = [scores[i] for i in ranked[:top_m]]
        return ranked_indices, ranked_scores
