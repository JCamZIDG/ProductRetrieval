import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_path: str, product_csv: str, embed_model_name: str):
        self.index_path = index_path
        self.product_csv = product_csv
        self.embed_model_name = embed_model_name
        self.index = None
        self.product_df = None
        self.embedding_model = None

    def load(self):
        logger.info("Cargando productos desde %s", self.product_csv)
        self.product_df = pd.read_csv(self.product_csv, sep='\t', dtype={'product_id': object})
        logger.info("Cargando modelo de embeddings %s", self.embed_model_name)
        self.embedding_model = SentenceTransformer(self.embed_model_name)
        logger.info("Cargando FAISS index desde %s", self.index_path)
        self.index = faiss.read_index(self.index_path)

    def retrieve(self, query: str, top_k: int = 50):
        q_emb = self.embedding_model.encode([query], convert_to_tensor=False)
        q_np = np.array(q_emb).astype('float32')
        faiss.normalize_L2(q_np)
        D, I = self.index.search(q_np.reshape(1, -1), top_k)
        return I.flatten().tolist(), D.flatten().tolist()

    def get_product(self, idx: int):
        row = self.product_df.iloc[idx]
        return {"product_id": row['product_id'], "product_name": row.get('product_name',''), "product_description": row.get('product_description','')}
