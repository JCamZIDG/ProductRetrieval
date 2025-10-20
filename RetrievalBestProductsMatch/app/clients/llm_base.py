# app/clients/llm_base.py
from typing import Dict, Any, Protocol


class LLMClientProtocol(Protocol):
    """
    Protocolo que define la interfaz mínima que RAGService espera.
    Implementaciones deben proporcionar generate_answer(query, context, **kwargs) -> Dict[str, Any].
    """
    def generate_answer(self, query: str, context: str, **kwargs) -> Dict[str, Any]:
        ...