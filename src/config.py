from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = "data/chroma"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RESULTS: int = 4
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()


