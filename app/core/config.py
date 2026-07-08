import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "DocAnalyzer"
    
    # Security Settings
    SECRET_KEY: str = "supersecretkeychangethisinproduction1234567890!"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # LLM & Embedding Settings
    GROQ_API_KEY: str = "your_groq_api_key"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Storage & Database Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./doc_analyzer.db"
    UPLOAD_DIR: str = "uploads"
    INDEX_DIR: str = "indexes"
    
    # Rate Limiting Settings
    RATE_LIMIT_CALLS: int = 20  # calls per minute
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
