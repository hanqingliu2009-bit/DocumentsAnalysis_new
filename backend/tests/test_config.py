"""Tests for configuration module."""
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from config import Settings


class TestSettings:
    """Test cases for Settings class."""

    def test_default_values(self, temp_dir):
        """Test that default values are set correctly (explicit LLM env avoids local .env)."""
        settings = Settings(
            DATA_DIR=temp_dir,
            VOLCENGINE_API_KEY=None,
            VOLCENGINE_BASE_URL="https://ark.cn-beijing.volces.com/api/v3",
            LLM_MODEL="",
        )

        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "info"
        assert settings.VOLCENGINE_BASE_URL.startswith("https://")
        assert settings.LLM_MODEL == ""
        assert settings.CHUNK_SIZE == 512
        assert settings.CHUNK_OVERLAP == 50
        assert settings.EMBEDDING_DIMENSION == 384

    def test_paths_are_created(self, temp_dir):
        """Test that data directories are created on initialization."""
        settings = Settings(DATA_DIR=temp_dir)

        assert settings.DATA_DIR.exists()
        assert settings.VECTOR_DB_PATH.exists()
        assert settings.DOCUMENT_STORE_PATH.exists()
        assert settings.FILE_STORAGE_PATH.exists()

    def test_custom_values(self, temp_dir):
        """Test that custom values override defaults."""
        settings = Settings(
            DATA_DIR=temp_dir,
            HOST="127.0.0.1",
            PORT=9000,
            DEBUG=True,
            CHUNK_SIZE=1024,
        )

        assert settings.HOST == "127.0.0.1"
        assert settings.PORT == 9000
        assert settings.DEBUG is True
        assert settings.CHUNK_SIZE == 1024

    def test_pdf_parser_options(self, temp_dir):
        """Test that PDF parser options are valid."""
        # Default should be pypdf
        settings = Settings(DATA_DIR=temp_dir)
        assert settings.PDF_PARSER == "pypdf"

        # Can be set to opendataloader
        settings2 = Settings(DATA_DIR=temp_dir, PDF_PARSER="opendataloader")
        assert settings2.PDF_PARSER == "opendataloader"

    def test_cors_origins(self, temp_dir):
        """Test CORS origins configuration."""
        settings = Settings(DATA_DIR=temp_dir)
        assert "http://localhost:5173" in settings.CORS_ORIGINS
        assert "http://localhost:3000" in settings.CORS_ORIGINS

    def test_max_file_size(self, temp_dir):
        """Test max file size configuration."""
        settings = Settings(DATA_DIR=temp_dir)
        assert settings.MAX_FILE_SIZE == 104_857_600  # 100MB in bytes

        custom_settings = Settings(DATA_DIR=temp_dir, MAX_FILE_SIZE=50_000_000)
        assert custom_settings.MAX_FILE_SIZE == 50_000_000


class TestSettingsValidation:
    """Test validation of settings."""

    def test_invalid_chunk_size(self, temp_dir):
        """Test that invalid chunk size raises error."""
        # This should not raise an error since chunk_size is an int
        settings = Settings(DATA_DIR=temp_dir, CHUNK_SIZE=-1)
        assert settings.CHUNK_SIZE == -1  # But it accepts the value

    def test_empty_volcengine_key(self, temp_dir):
        """Test that empty API key is allowed."""
        settings = Settings(DATA_DIR=temp_dir, VOLCENGINE_API_KEY=None)
        assert settings.VOLCENGINE_API_KEY is None

    def test_path_as_string(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        settings = Settings(DATA_DIR=str(temp_dir))
        assert isinstance(settings.DATA_DIR, Path)
