from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "NCERT Doubt Solver"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Windows-specific paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    NCERT_PDF_PATH: str = str(BASE_DIR / "backend" / "data" / "ncert_pdfs")
    PROCESSED_DATA_PATH: str = str(BASE_DIR / "backend" / "data" / "processed")
    VECTOR_DB_PATH: str = str(BASE_DIR / "backend" / "vector_db")
    MODEL_PATH: str = str(BASE_DIR / "backend" / "models")
    
    # Tesseract and Poppler paths (Windows)
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH: str = r"C:\Program Files\poppler-23.11.0\Library\bin"
    
    # Vector Database
    VECTOR_DB_TYPE: str = "chromadb"
    COLLECTION_NAME: str = "ncert_documents"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    
    # Groq API Settings
    USE_GROQ: bool = True
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    
    # LLM Settings
    LLM_TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 512
    
    # RAG Parameters
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # Language Support
    SUPPORTED_LANGUAGES: List[str] = [
        "english", "hindi", "urdu", "marathi", "tamil", 
        "telugu", "bengali", "gujarati", "kannada", "malayalam"
    ]
    
    # Grades
    SUPPORTED_GRADES: List[int] = [5, 6, 7, 8, 9, 10]
    
    # Performance
    MAX_LATENCY_SECONDS: float = 5.0
    BATCH_SIZE: int = 16
    
    # OCR
    OCR_LANGUAGE: str = "eng+hin+urd"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()

# Configure system paths for Windows
import pytesseract
pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
os.environ['PATH'] = settings.POPPLER_PATH + ';' + os.environ.get('PATH', '')