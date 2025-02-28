from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qa_project.core.qa_system import QASystem
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sistema de Q&A con DeepSeek",
    description="API para hacer preguntas sobre documentos usando RAG y DeepSeek",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_system = None

@app.on_event("startup")
async def startup_event():
    global qa_system
    try:
        qa_system = QASystem()
        logger.info("Sistema QA inicializado correctamente")
    except Exception as e:
        logger.error(f"Error inicializando el sistema QA: {str(e)}")
        raise


class QuestionRequest(BaseModel):
    question: str
    max_sources: int = 3

class SourceDocument(BaseModel):
    content: str
    source: str
    score: float

class QAResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    success: bool
    error: str = None

@app.post("/ask", response_model=QAResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint para hacer preguntas al sistema de Q&A
    
    Ejemplo:
    {
        "question": "¿Qué regulación propuso la UE?",
        "max_sources": 2
    }
    """
    try:
        if not qa_system:
            raise HTTPException(status_code=503, detail="Servicio no disponible")
        
        result = qa_system.query(request.question)
        
        sources = [
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata['source'],
                score=doc.metadata.get('score', 0.0)
            ) for doc in result['source_documents'][:request.max_sources]
        ]
        
        return QAResponse(
            answer=result['result'],
            sources=sources,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error procesando pregunta: {str(e)}")
        return QAResponse(
            answer="",
            sources=[],
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)