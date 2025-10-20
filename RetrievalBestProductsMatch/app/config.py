from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FAISS_INDEX_PATH: str = "data/faiss.index"
    PRODUCT_CSV: str = "data/product.csv"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    LLM_MODEL_NAME: str = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

    LLM_ALLOW_DOWNLOAD: bool = True
    
    TOP_K_RETRIEVER: int = 50
    RERANK_TOP_M: int = 10

    class Config:
        env_file = ".env"

settings = Settings()