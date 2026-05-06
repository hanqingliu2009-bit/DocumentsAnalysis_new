"""Backend configuration settings."""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Always load backend/.env regardless of process cwd (uvicorn may be started from repo root).
_BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "info"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Chat LLM (OpenAI-compatible chat.completions), e.g. Alibaba DashScope:
    # LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None
    LLM_MODEL: str = ""
    # chat.completions max_tokens limits the assistant reply length (tokens), not upload size.
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.0

    # Remote OpenAI-compatible /embeddings (optional separate key/URL; else LLM_* is used).
    EMBEDDING_OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_OPENAI_BASE_URL: Optional[str] = None

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Path("./data")
    VECTOR_DB_PATH: Path = Path("./data/vector_db")
    DOCUMENT_STORE_PATH: Path = Path("./data/documents")
    FILE_STORAGE_PATH: Path = Path("./data/files")

    # Processing Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    # Embeddings: "local" = sentence-transformers; "openai" = HTTP POST .../embeddings (OpenAI-compatible).
    EMBEDDING_BACKEND: str = "local"
    # Default local model: BGE Chinese (same family as Splite ingest). Change dimension if you switch models.
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    # True: call .../embeddings/multimodal with multimodal input blocks; False: standard /embeddings text.
    EMBEDDING_USE_MULTIMODAL_API: bool = False
    # Must match EMBEDDING_MODEL output (1024 for bge-large-zh-v1.5).
    EMBEDDING_DIMENSION: int = 1024
    # Change collection or clear vector_db when switching model/dimension.
    CHROMADB_COLLECTION: str = "document_chunks"
    # If set: directory is created and HF_HOME defaults here for Hub / sentence-transformers cache.
    EMBEDDING_CACHE_DIR: Optional[Path] = None
    TRANSFORMERS_OFFLINE: bool = False

    def llm_openai_api_key(self) -> str:
        key = self.LLM_API_KEY
        if key is not None and str(key).strip():
            return str(key).strip()
        return ""

    def llm_openai_base_url(self) -> str:
        if self.LLM_BASE_URL is not None and str(self.LLM_BASE_URL).strip():
            return str(self.LLM_BASE_URL).strip().rstrip("/")
        return ""

    def embedding_openai_api_key(self) -> str:
        """API key for OpenAI-compatible /embeddings; dedicated key overrides LLM_API_KEY."""
        for key in (self.EMBEDDING_OPENAI_API_KEY, self.LLM_API_KEY):
            if key is not None and str(key).strip():
                return str(key).strip()
        return ""

    def embedding_openai_base_url(self) -> str:
        """Base URL for /embeddings; dedicated URL overrides LLM_BASE_URL."""
        if self.EMBEDDING_OPENAI_BASE_URL is not None and str(self.EMBEDDING_OPENAI_BASE_URL).strip():
            return str(self.EMBEDDING_OPENAI_BASE_URL).strip().rstrip("/")
        return self.llm_openai_base_url()

    # PDF Parser Configuration
    PDF_PARSER: str = "pypdf"  # Options: "pypdf", "opendataloader"

    # RAG backend: "chromadb" = local embeddings + Chroma; "external_graph" = supplier graph HTTP + LLM;
    # "hybrid" = parallel graph HTTP + splite BGE Chroma, merged context, then LLM.
    RAG_BACKEND: str = "chromadb"
    EXTERNAL_GRAPH_API_URL: str = "http://14.103.133.160:8022/graph/info"
    EXTERNAL_GRAPH_DOMAIN: str = "南兴装备"
    EXTERNAL_GRAPH_SEARCH_TYPE: str = "triplet"
    EXTERNAL_GRAPH_TIMEOUT: float = 60.0

    # Hybrid RAG: splite corpus in a dedicated Chroma collection (sentence-transformers, local only).
    HYBRID_SPLITE_COLLECTION: str = "splite_bge_zh_v15"
    HYBRID_SPLITE_EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    HYBRID_VECTOR_TOP_K: int = 5
    HYBRID_CONTEXT_MAX_CHARS: int = 24000

    # Search Configuration
    TOP_K_RETRIEVAL: int = 15
    SIMILARITY_THRESHOLD: float = 0.7

    # Upload Limits
    MAX_FILE_SIZE: int = 104_857_600  # 100MB
    MAX_FILES_PER_UPLOAD: int = 10

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        self.DOCUMENT_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self.FILE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        if self.EMBEDDING_CACHE_DIR is not None:
            cache = Path(self.EMBEDDING_CACHE_DIR)
            cache = cache.resolve() if cache.is_absolute() else (self.BASE_DIR / cache).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, "EMBEDDING_CACHE_DIR", cache)
            os.environ.setdefault("HF_HOME", str(cache))

        if self.TRANSFORMERS_OFFLINE:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)


# Global settings instance
settings = Settings()
