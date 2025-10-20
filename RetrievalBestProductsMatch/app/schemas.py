from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 50
    rerank_m: Optional[int] = 10
    use_rerank: Optional[bool] = True

class SearchResult(BaseModel):
    product_id: str
    product_name: str
    product_description: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
