from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import traceback
import torch

from .vectorstore import load_pdfs, load_texts, chunk_docs, build_or_load_store, get_vectorstore_stats
from .rag_pipeline import build_qa_chain, get_device, clean_generated_answer
from .schemas import AskRequest, AskResponse, Source
from .config import settings

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="Enhanced RAG Assistant API",
    description="Custom-trained LLM with RAG capabilities - Production Ready",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Variables
VECTORSTORE = None
QA_CHAIN = None

# ============================================================================
# Application Lifecycle Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global VECTORSTORE, QA_CHAIN
    
    print("🚀 Starting Enhanced RAG Assistant API...")
    print(f"📁 Persist directory: {settings.PERSIST_DIR}")
    print(f"🤖 LLM Provider: {settings.LLM_PROVIDER}")
    print(f"🎯 Custom Model Path: {settings.CUSTOM_MODEL_PATH}")
    print(f"💻 Device: {get_device()}")

    try:
        # Initialize vectorstore
        VECTORSTORE = build_or_load_store(chunks=None)
        print("✅ Vectorstore loaded successfully")
        
        # Initialize QA chain
        QA_CHAIN = build_qa_chain(VECTORSTORE)
        print("✅ QA Chain initialized successfully")
        
    except Exception as e:
        VECTORSTORE = None
        QA_CHAIN = None
        print(f"⚠️ Startup warning: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🛑 Shutting down Enhanced RAG Assistant API...")

# ============================================================================
# Document Ingestion Endpoints
# ============================================================================

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest PDF documents into the knowledge base"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Create data directory and save file
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / file.filename
        content = await file.read()
        dest.write_bytes(content)

        print(f"📄 Processing PDF: {file.filename}")

        # Process document
        docs = load_pdfs([dest])
        if not docs:
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")

        chunks = chunk_docs(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from the document")

        # Update global state
        global VECTORSTORE, QA_CHAIN
        VECTORSTORE = build_or_load_store(chunks)
        QA_CHAIN = build_qa_chain(VECTORSTORE)

        print(f"✅ Successfully processed {len(chunks)} chunks from {file.filename}")

        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "chunks_added": len(chunks),
            "filename": file.filename,
            "file_size": len(content),
            "pages_processed": len(docs) if docs else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {str(e)}")

@app.post("/ingest/text")
async def ingest_text(file: UploadFile = File(...)):
    """Ingest text documents into the knowledge base"""
    try:
        # Validate file type
        allowed_extensions = ['.txt', '.md', '.csv']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="File must be .txt, .md, or .csv")

        # Create data directory and save file
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / file.filename
        content = await file.read()
        dest.write_bytes(content)

        print(f"📝 Processing text file: {file.filename}")

        # Process document
        docs = load_texts([dest])
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the file")

        chunks = chunk_docs(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks could be created from the document")

        # Update global state
        global VECTORSTORE, QA_CHAIN
        VECTORSTORE = build_or_load_store(chunks)
        QA_CHAIN = build_qa_chain(VECTORSTORE)

        print(f"✅ Successfully processed {len(chunks)} chunks from {file.filename}")

        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "chunks_added": len(chunks),
            "filename": file.filename,
            "file_size": len(content)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")

@app.delete("/ingest/clear")
async def clear_knowledge_base():
    """Clear the entire knowledge base"""
    import shutil
    global VECTORSTORE, QA_CHAIN
    
    try:
        # Remove persistent data
        if settings.PERSIST_DIR.exists():
            shutil.rmtree(settings.PERSIST_DIR)
            
        # Remove raw files
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            
        # Reset global state
        VECTORSTORE = None
        QA_CHAIN = None
        
        print("🗑️ Knowledge base cleared successfully")
        
        return {
            "status": "success", 
            "message": "Knowledge base cleared successfully"
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear knowledge base: {str(e)}")

# ============================================================================
# Query Processing Endpoints
# ============================================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question to the RAG system"""
    if QA_CHAIN is None:
        raise HTTPException(
            status_code=400, 
            detail="No knowledge base found. Please ingest documents first using /ingest/pdf or /ingest/text endpoints."
        )
    
    try:
        print(f"❓ Processing question: {request.question}")
        
        # Process question through RAG pipeline
        result = QA_CHAIN.invoke({"query": request.question})
        
        # Extract and format sources
        sources = []
        for doc in result.get("source_documents", []):
            meta = doc.metadata or {}
            sources.append(Source(
                source=meta.get("source"),
                page=meta.get("page"),
                snippet=doc.page_content[:300]
            ))
        
        # Clean and optimize answer
        raw_answer = result.get("result", "")
        clean_answer = clean_generated_answer(raw_answer)
        
        print(f"✅ Generated answer: {clean_answer[:100]}...")
        
        return AskResponse(
            answer=clean_answer,
            sources=sources
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.get("/ask/suggestions")
async def get_question_suggestions():
    """Get sample questions based on ingested content"""
    if VECTORSTORE is None:
        return {
            "suggestions": [
                "What is machine learning?",
                "Explain artificial intelligence in simple terms.",
                "How does deep learning work?",
                "What are neural networks?",
                "What are transformers in AI?"
            ]
        }
    
    # Dynamic suggestions based on content
    return {
        "suggestions": [
            "What are the main topics covered in the documents?",
            "Summarize the key concepts from the uploaded content.",
            "What are the most important points mentioned?",
            "Explain the main ideas in simple terms.",
            "What specific information is available in the knowledge base?"
        ]
    }

# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models/available")
async def get_available_models():
    """Get list of available models"""
    custom_model_exists = Path(settings.CUSTOM_MODEL_PATH).exists()
    
    models = {
        "custom": {
            "name": "Custom Trained LoRA Model",
            "description": "Your fine-tuned model on PDF content",
            "type": "local",
            "status": "available" if custom_model_exists else "not_found",
            "path": settings.CUSTOM_MODEL_PATH
        },
        "microsoft/DialoGPT-medium": {
            "name": "DialoGPT Medium",
            "description": "Microsoft conversational AI model",
            "type": "huggingface",
            "status": "available"
        },
        "google/flan-t5-small": {
            "name": "Flan-T5 Small",
            "description": "Google instruction-following model",
            "type": "huggingface",
            "status": "available"
        }
    }
    
    current_model = "custom" if settings.LLM_PROVIDER == "local" else settings.BASE_MODEL_NAME
    
    return {
        "available_models": models,
        "current_model": current_model,
        "current_provider": settings.LLM_PROVIDER,
        "custom_model_exists": custom_model_exists
    }

@app.post("/models/switch")
async def switch_model(model_name: str):
    """Switch between different models"""
    global QA_CHAIN
    
    try:
        print(f"🔄 Switching to model: {model_name}")
        
        # Update settings based on model choice
        if model_name == "custom":
            if not Path(settings.CUSTOM_MODEL_PATH).exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"Custom model not found at {settings.CUSTOM_MODEL_PATH}"
                )
            settings.LLM_PROVIDER = "local"
        else:
            settings.LLM_PROVIDER = "hf"
            settings.BASE_MODEL_NAME = model_name
        
        # Rebuild QA chain with new model
        if VECTORSTORE:
            QA_CHAIN = build_qa_chain(VECTORSTORE)
            print(f"✅ Successfully switched to {model_name}")
            
            return {
                "status": "success", 
                "message": f"Successfully switched to {model_name}",
                "current_model": model_name,
                "provider": settings.LLM_PROVIDER
            }
        else:
            return {
                "status": "warning",
                "message": f"Model switched to {model_name}, but no documents ingested yet",
                "current_model": model_name,
                "provider": settings.LLM_PROVIDER
            }
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model switching failed: {str(e)}")

# ============================================================================
# System Status and Health Endpoints
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "message": "Enhanced RAG Assistant API",
        "description": "Custom-trained LLM with RAG capabilities",
        "version": "2.0.0",
        "status": "running",
        "custom_model_available": Path(settings.CUSTOM_MODEL_PATH).exists(),
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "stats": "/stats",
            "ingest_pdf": "/ingest/pdf",
            "ingest_text": "/ingest/text",
            "ask_question": "/ask",
            "models": "/models/available",
            "model_switch": "/models/switch"
        },
        "features": [
            "Custom LoRA Model Integration",
            "PDF & Text Document Processing", 
            "Semantic Vector Search",
            "Real-time Model Switching",
            "Production-Ready API"
        ]
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    vectorstore_stats = get_vectorstore_stats(VECTORSTORE) if VECTORSTORE else {"status": "not_initialized"}
    
    return {
        "status": "healthy",
        "vectorstore": vectorstore_stats,
        "qa_chain": "ready" if QA_CHAIN else "not_ready",
        "custom_model": "available" if Path(settings.CUSTOM_MODEL_PATH).exists() else "not_found",
        "device": get_device(),
        "provider": settings.LLM_PROVIDER
    }

@app.get("/stats")
async def get_system_stats():
    """Get detailed system statistics"""
    stats = {
        "vectorstore_initialized": VECTORSTORE is not None,
        "qa_chain_ready": QA_CHAIN is not None,
        "current_provider": settings.LLM_PROVIDER,
        "device": get_device()
    }
    
    # Add vectorstore stats if available
    if VECTORSTORE:
        stats["vectorstore"] = get_vectorstore_stats(VECTORSTORE)
    
    # Check raw data directory
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        files = list(raw_dir.glob("*"))
        stats["raw_files"] = len(files)
        stats["raw_files_list"] = [f.name for f in files]
    else:
        stats["raw_files"] = 0
        stats["raw_files_list"] = []
    
    return stats

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with helpful information"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": {
                "documentation": "/docs",
                "health": "/health", 
                "stats": "/stats",
                "ingestion": ["/ingest/pdf", "/ingest/text", "/ingest/clear"],
                "querying": ["/ask", "/ask/suggestions"],
                "models": ["/models/available", "/models/switch"]
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An internal error occurred. Check the server logs for details."
        }
    )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8003,
        reload=True,
        log_level="info"
    )

@app.get("/debug/ollama")
async def debug_ollama():
    """Debug Ollama and Phi-3 status"""
    from .rag_pipeline import get_ollama_status, test_ollama_phi3
    
    status = get_ollama_status()
    test_success, test_result = test_ollama_phi3()
    
    return {
        "ollama_status": status,
        "phi3_test": {
            "success": test_success,
            "result": test_result[:200] if isinstance(test_result, str) else test_result
        },
        "instructions": {
            "install_ollama": "Download from https://ollama.com/download/mac",
            "install_phi3": "ollama pull phi3:mini",
            "start_service": "ollama serve"
        }
    }
