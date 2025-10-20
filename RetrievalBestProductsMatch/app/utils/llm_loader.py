# app/utils/llm_loader.py
import logging
import os
from typing import Optional
from app.clients.gpt4all_client import GPT4AllClient

logger = logging.getLogger(__name__)

def load_gpt4all_instance_with_download_support(model_identifier: str, allow_download: bool = True):
    """
    Instancia GPT4All a partir de un nombre de modelo o ruta. Si allow_download=True,
    el SDK puede descargar el modelo si no está localmente.
    """
    try:
        from gpt4all import GPT4All
    except Exception as e:
        logger.exception("gpt4all package not available: %s", e)
        raise

    # Intentar formas comunes de instanciación; dejamos que la librería gestione el cache/download.
    # 1) Intento simple por model name (esto normalmente hará la descarga en el cache si falta).
    try:
        logger.info("Instanciando GPT4All con model_identifier=%s allow_download=%s", model_identifier, allow_download)
        # Algunas versiones aceptan (model_name) y otras (model_name, model_path=..., allow_download=...)
        try:
            return GPT4All(model_identifier, allow_download=allow_download)
        except TypeError:
            # otra firma posible
            model_name = os.path.basename(model_identifier)
            model_dir = os.path.dirname(model_identifier) or None
            if model_dir:
                return GPT4All(model_name=model_name, model_path=model_dir, allow_download=allow_download)
            else:
                return GPT4All(model_name=model_name, allow_download=allow_download)
    except Exception as e:
        logger.exception("Instanciación GPT4All fallida para %s: %s", model_identifier, e)
        raise

def load_local_gpt4all_adapter(model_identifier: str, allow_download: bool = True) -> Optional[GPT4AllClient]:
    """
    Devuelve un GPT4AllClient (adaptador) o None si no se consiguió instanciar.
    - model_identifier: nombre del modelo reconocible por gpt4all (o ruta .gguf).
    - allow_download: si True, permite que el SDK descargue el modelo.
    """
    if not model_identifier or str(model_identifier).lower() == "none":
        logger.info("LLM model identifier empty or 'none' -> skipping load.")
        return None

    try:
        gpt_inst = load_gpt4all_instance_with_download_support(model_identifier, allow_download=allow_download)
        client = GPT4AllClient(gpt_inst)
        logger.info("gpt4all adapter initialized for model: %s", model_identifier)
        return client
    except Exception as e:
        # No fallamos el startup, solo devolvemos None y logueamos el problema.
        logger.exception("Failed to initialize gpt4all adapter for %s: %s", model_identifier, e)
        return None