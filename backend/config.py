"""Backend configuration settings."""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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

    # LLM via OpenAI-compatible SDK (openai.OpenAI). Configure in backend/.env.
    # 火山方舟：VOLCENGINE_BASE_URL 一般为 OpenAI 兼容地址；LLM_MODEL 为接入点 ID（常为 ep-...）或控制台给出的 model 名。
    VOLCENGINE_API_KEY: Optional[str] = None
    VOLCENGINE_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    LLM_MODEL: str = ""
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.0
    # When True (default): /api/chat may call the LLM even when retrieval returns no chunks (general chat).
    # /api/query always stays document-only when nothing is retrieved. Set False to disable chat fallback.
    CHAT_ALLOW_GENERAL_WITHOUT_DOCS: bool = True

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Path("./data")
    VECTOR_DB_PATH: Path = Path("./data/vector_db")
    DOCUMENT_STORE_PATH: Path = Path("./data/documents")
    FILE_STORAGE_PATH: Path = Path("./data/files")

    # Processing Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    # If set: this directory is created and HF_HOME defaults here (setdefault) so Hub / sentence-transformers
    # weights live under a known path for backup and offline copy. Relative paths are under backend/ (BASE_DIR).
    EMBEDDING_CACHE_DIR: Optional[Path] = None
    # When True: export TRANSFORMERS_OFFLINE and HF_HUB_OFFLINE to os.environ so Hugging Face / transformers
    # honor offline mode (use with a populated EMBEDDING_CACHE_DIR / HF_HOME copy). Declared here so backend/.env applies.
    TRANSFORMERS_OFFLINE: bool = False

    # PDF Parser Configuration
    PDF_PARSER: str = "pypdf"  # Options: "pypdf", "opendataloader"

    # Search Configuration
    TOP_K_RETRIEVAL: int = 5
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
            # Mutate so introspection and logs show the resolved path
            object.__setattr__(self, "EMBEDDING_CACHE_DIR", cache)
            # Hugging Face Hub + transformers use HF_HOME; sentence-transformers follows the same cache tree.
            os.environ.setdefault("HF_HOME", str(cache))

        if self.TRANSFORMERS_OFFLINE:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)


# Global settings instance
settings = Settings()
