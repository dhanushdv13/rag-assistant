from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER: str = "local"  # "local" for custom model, "hf" for standard
    CUSTOM_MODEL_PATH: str = "./models/enhanced-rag-model"
    BASE_MODEL_NAME: str = "microsoft/DialoGPT-medium"
    
    # Model Parameters
    MAX_NEW_TOKENS: int = 120
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    DO_SAMPLE: bool = True
    
    # Vector DB Configuration
    VECTOR_DB: str = "chroma"
    PERSIST_DIR: Path = Path("data/processed/chroma_store")
    
    # Text Processing Parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 6
    
    class Config:
        env_file = ".env"

settings = Settings()
