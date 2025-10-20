# app/clients/gpt4all_client.py
import logging
import json
import re
from typing import Dict, Any
from app.clients.llm_base import LLMClientProtocol

logger = logging.getLogger(__name__)


class GPT4AllClient(LLMClientProtocol):
    """
    Adaptador para gpt4all usando el patrón que usas en el notebook:
      gpt = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
      with gpt.chat_session():
          raw = gpt.generate(prompt, max_tokens=400)

    Este adaptador encapsula esa lógica y devuelve un dict (si el LLM produce JSON)
    o {"raw": "<texto>"} en caso contrario.
    """
    def __init__(self, gpt4all_instance):
        self._gpt = gpt4all_instance

    def _extract_json_from_text(self, text: str):
        """Extrae el primer bloque JSON del texto, si existe."""
        try:
            m = re.search(r'\{.*\}', text, flags=re.DOTALL)
            if not m:
                return None
            payload = json.loads(m.group(0))
            return payload
        except Exception as e:
            logger.debug("Failed to parse JSON from LLM output: %s", e)
            return None

    def _unpack_raw_output(self, raw_out):
        """
        raw_out puede ser str o dict dependiendo de la versión del SDK.
        Este helper retorna siempre texto (str) para intentar parseo.
        """
        # Si viene dict, intenta encontrar campos comunes
        if isinstance(raw_out, dict):
            # common keys: 'response', 'text', 'choices'
            if 'response' in raw_out and isinstance(raw_out['response'], str):
                return raw_out['response']
            if 'text' in raw_out and isinstance(raw_out['text'], str):
                return raw_out['text']
            if 'choices' in raw_out and isinstance(raw_out['choices'], list) and raw_out['choices']:
                first = raw_out['choices'][0]
                # choices item puede ser dict {'text': '...'}
                if isinstance(first, dict) and 'text' in first:
                    return first['text']
                return str(first)
            # fallback a string
            try:
                return json.dumps(raw_out)
            except Exception:
                return str(raw_out)
        else:
            return str(raw_out)

    def generate_answer(self, query: str, context: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Genera una respuesta a partir de query+context. Devuelve dict con la respuesta parseada
        (busca JSON embebido) o {'raw': '<texto completo>'} si no logra extraer JSON.
        """
        prompt = (
            "Context:\n"
            f"{context}\n\n"
            f"User question: {query}\n\n"
            "Using ONLY the information in Context, return a JSON object with keys:\n"
            "  - best_product_id: (integer or null),\n"
            "  - reasons: (list of why you think is the best match),\n"
            "  - top_candidates: (list of objects {product_id:int, score:float})\n\n"
            "Return JSON ONLY. No extra commentary.\n"
        )

        try:
            # Usa chat_session si existe (tal como en tu notebook)
            if hasattr(self._gpt, "chat_session"):
                try:
                    with self._gpt.chat_session():
                        raw_out = self._gpt.generate(prompt, max_tokens=max_tokens)
                except TypeError:
                    # algunas versiones pueden querer (prompt, max_tokens) distinto, fallback:
                    raw_out = self._gpt.generate(prompt)
            else:
                # fallback directo a generate
                raw_out = self._gpt.generate(prompt, max_tokens=max_tokens)

            text = self._unpack_raw_output(raw_out)
            parsed = self._extract_json_from_text(text)
            logger.debug("Raw LLM output: %s", text[:8000]) 
            if parsed is not None:
                return parsed
            else:
                logger.warning("gpt4all returned no JSON. Returning raw text for debugging.")
                return {"raw": text}
        except Exception as e:
            logger.exception("gpt4all generation error: %s", e)
            raise
