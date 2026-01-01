"""
StudyMate Configuration Management
Centralized configuration for the entire application.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration management."""

    CURRENT_WORKING_DIR: str = os.getcwd()

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    ALLOWED_EXTENSIONS: list = os.getenv("ALLOWED_EXTENSIONS", "pdf").split(",")

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 150))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.8))
    TOP_K: int = int(os.getenv("TOP_K", 10))

    # Vector Store Configuration
    VECTOR_STORE_PATH: str = os.path.join(CURRENT_WORKING_DIR, "backend", "chroma_db")
    CHROMA_PATH: str = os.path.join(CURRENT_WORKING_DIR, "backend", "chroma_db")
    COLLECTION_PREFIX: str = os.getenv("COLLECTION_PREFIX", "studymate_")

    # Security Configuration
    ENABLE_GUARDRAILS: bool = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", 500))
    MAX_RESPONSE_LENGTH: int = int(os.getenv("MAX_RESPONSE_LENGTH", 2000))

    # UI Configuration
    PAGE_TITLE: str = os.getenv("PAGE_TITLE", "StudyMate - Your AI Study Buddy")
    PAGE_ICON: str = os.getenv("PAGE_ICON", "📚")
    THEME_PRIMARY_COLOR: str = os.getenv("THEME_PRIMARY_COLOR", "#1f77b4")

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status."""
        issues = []

        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is not set")

        if cls.MAX_FILE_SIZE_MB <= 0:
            issues.append("MAX_FILE_SIZE_MB must be positive")

        if cls.CHUNK_SIZE <= 0:
            issues.append("CHUNK_SIZE must be positive")

        if not (0 < cls.SIMILARITY_THRESHOLD <= 1):
            issues.append("SIMILARITY_THRESHOLD must be between 0 and 1")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": cls.to_dict()
        }

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "openai_api_key_set": bool(cls.OPENAI_API_KEY),
            "openai_model": cls.OPENAI_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "allowed_extensions": cls.ALLOWED_EXTENSIONS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "top_k": cls.TOP_K,
            "vector_store_path": cls.VECTOR_STORE_PATH,
            "enable_guardrails": cls.ENABLE_GUARDRAILS,
            "max_query_length": cls.MAX_QUERY_LENGTH,
            "max_response_length": cls.MAX_RESPONSE_LENGTH
        }

# Global configuration instance
config = Config()