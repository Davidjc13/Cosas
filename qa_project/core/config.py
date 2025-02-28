import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Clase Singleton para manejar la configuración"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Inicializa los parámetros de configuración"""

        load_dotenv()  
        
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

        self.DEEPSEEK_MODEL = "deepseek-chat"
        self.TEMPERATURE = 0.3
        self.MAX_TOKENS = 1000
        
        self.DOCUMENTS_DIR = "qa_project/documentos"
        self.FILE_PATTERN = "**/*.txt"
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
