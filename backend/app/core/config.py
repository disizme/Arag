from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Adaptive RAG System"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "snowflake-arctic-embed2:latest"
    OLLAMA_DEFAULT_MODEL: str = "qwen3:latest"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_VECTOR_SIZE: int = 1024
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Upload Configuration
    UPLOAD_DIRECTORY: str = "uploads"
    
    class Config:
        env_file = ".env"

settings = Settings()