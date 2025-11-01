# backend\app\main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NCERT Doubt Solver",
    version="1.0.0",
    description="Multilingual RAG system for NCERT textbooks"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for services (initialized on startup)
rag_service = None
embedding_service = None
Language_service = None

# ==================== Pydantic Models ====================

class ChatRequest(BaseModel):
    query: str
    grade: Optional[int] = None
    subject: Optional[str] = None
    language: Optional[str] = None
    conversation_id: Optional[str] = None

class Citation(BaseModel):
    source_id: int
    text: str
    grade: Optional[int]
    subject: Optional[str]
    page_number: Optional[int]

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    language: str
    confidence: str
    latency_ms: float
    conversation_id: str

class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int  # 1-5
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str
    message: str

# ==================== Basic Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to NCERT Doubt Solver API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "metadata": "/metadata",
            "feedback": "/feedback",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "api": "running",
            "rag": "not_initialized" if rag_service is None else "initialized",
            "embedding": "not_initialized" if embedding_service is None else "initialized",
            "language": "not_initialized" if language_service is None else "initialized"
        }
    }
    return status

@app.get("/metadata")
async def get_metadata():
    """Get supported grades, subjects, and languages"""
    try:
        from backend.app.config import settings
        
        return {
            "grades": settings.SUPPORTED_GRADES,
            "subjects": ["math", "science", "social_science", "english", "hindi"],
            "languages": settings.SUPPORTED_LANGUAGES,
            "models": {
                "embedding": settings.EMBEDDING_MODEL,
                "llm": settings.LLM_MODEL
            }
        }
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return {
            "grades": [5, 6, 7, 8, 9, 10],
            "subjects": ["math", "science", "social_science", "english", "hindi"],
            "languages": ["english", "hindi", "urdu", "marathi", "tamil", "telugu"]
        }

# ==================== Service Initialization ====================

@app.on_event("startup")
async def startup_event():
    global rag_service, embedding_service, language_service
    
    logger.info("=" * 60)
    logger.info("🚀 Starting NCERT Doubt Solver API...")
    logger.info("=" * 60)
    
    try:
        from backend.app.config import settings
        from backend.services.embedding_service import EmbeddingService
        from backend.services.language_service import LanguageService
        
        logger.info("📦 Initializing services...")
        
        # Initialize Embedding Service
        logger.info("⏳ Loading embedding model...")
        embedding_service = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL,
            vector_db_path=settings.VECTOR_DB_PATH
        )
        embedding_service.create_collection(settings.COLLECTION_NAME)
        logger.info("✅ Embedding service ready")
        
        # Initialize Language Service
        language_service = LanguageService()
        logger.info("✅ Language service ready")
        
        # Initialize RAG Service with Groq
        if settings.USE_GROQ:
            logger.info("⏳ Initializing Groq API...")
            from backend.services.rag_service_groq import RAGServiceGroq
            
            rag_service = RAGServiceGroq(
                groq_api_key=settings.GROQ_API_KEY,
                groq_model=settings.GROQ_MODEL,
                embedding_service=embedding_service,
                language_service=language_service,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            logger.info(f"✅ RAG service ready with Groq ({settings.GROQ_MODEL})")
        else:
            # Fallback to local model
            logger.info("⏳ Loading local LLM model...")
            from backend.services.rag_service import RAGService
            
            rag_service = RAGService(
                model_name="google/flan-t5-small",
                embedding_service=embedding_service,
                language_service=language_service,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            logger.info("✅ RAG service ready with local model")
        
        logger.info("=" * 60)
        logger.info("🎉 All services initialized successfully!")
        logger.info("=" * 60)
        logger.info("📍 API available at: http://0.0.0.0:8000")
        logger.info("📚 API docs at: http://0.0.0.0:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERROR INITIALIZING SERVICES")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
# ==================== Main Chat Endpoint ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for question answering
    
    - **query**: The question to ask
    - **grade**: Optional grade filter (5-10)
    - **subject**: Optional subject filter (math, science, etc.)
    - **language**: Optional language preference
    - **conversation_id**: Optional ID to continue conversation
    """
    
    # Check if RAG service is initialized
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized. Please wait for startup to complete or check logs."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"📥 Received query: {request.query[:100]}...")
        logger.info(f"   Grade: {request.grade}, Subject: {request.subject}, Language: {request.language}")
        
        # Generate answer using RAG pipeline
        result = rag_service.generate_answer(
            query=request.query,
            grade=request.grade,
            subject=request.subject,
            language=request.language
        )
        
        # Format citations
        citations = []
        for i, ctx in enumerate(result["contexts"], 1):
            metadata = ctx["metadata"]
            citations.append(Citation(
                source_id=i,
                text=ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                grade=int(metadata.get("grade")) if metadata.get("grade") else None,
                subject=metadata.get("subject"),
                page_number=int(metadata.get("page_number")) if metadata.get("page_number") else None
            ))
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate conversation ID
        import uuid
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        logger.info(f"✅ Response generated in {latency_ms:.0f}ms")
        logger.info(f"   Citations: {len(citations)}, Confidence: {result['confidence']}")
        
        return ChatResponse(
            answer=result["answer"],
            citations=citations,
            language=result["language"],
            confidence=result["confidence"],
            latency_ms=latency_ms,
            conversation_id=conversation_id
        )
    
    except Exception as e:
        logger.error(f"❌ Error processing chat request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# ==================== Feedback Endpoint ====================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a conversation
    
    - **conversation_id**: ID of the conversation
    - **rating**: Rating from 1-5
    - **comment**: Optional comment
    """
    try:
        # Validate rating
        if feedback.rating < 1 or feedback.rating > 5:
            raise HTTPException(
                status_code=400,
                detail="Rating must be between 1 and 5"
            )
        
        logger.info(f"📝 Feedback received:")
        logger.info(f"   Conversation: {feedback.conversation_id}")
        logger.info(f"   Rating: {feedback.rating}/5")
        if feedback.comment:
            logger.info(f"   Comment: {feedback.comment[:100]}...")
        
        # TODO: Store feedback in database
        # For now, just log it
        
        return FeedbackResponse(
            status="success",
            message="Thank you for your feedback!"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error storing feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error storing feedback: {str(e)}"
        )

# ==================== Test/Debug Endpoints ====================

@app.get("/test/vector-db")
async def test_vector_db():
    """Test endpoint to check vector database status"""
    try:
        if embedding_service is None:
            return {"status": "not_initialized"}
        
        # Get collection info
        info = embedding_service.get_collection_info()
        
        if info:
            return {
                "status": "initialized",
                "collection_name": info["name"],
                "document_count": info["count"],
                "vector_db_path": info["path"]
            }
        else:
            return {"status": "error", "message": "Could not get collection info"}
            
    except Exception as e:
        logger.error(f"Error checking vector DB: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )