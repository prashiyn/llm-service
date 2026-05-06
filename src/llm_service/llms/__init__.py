from dotenv import load_dotenv

from llm_service.config import get_config_dir

load_dotenv()
_api_keys_env = get_config_dir() / "api_keys.env"
if _api_keys_env.exists():
    load_dotenv(_api_keys_env)

from llm_service.llms.client import LLMClient
from llm_service.llms.config import get_llm_config
from llm_service.llms.embeddings import EmbeddingClient

__all__ = ["LLMClient", "EmbeddingClient", "get_llm_config"]
