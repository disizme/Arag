from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import time
from datetime import datetime

from shared.models.schemas import (
    DocumentUploadRequest, DocumentUploadResponse, 
    QueryRequest, QueryResponse, HealthResponse, CollectionInfoResponse
)
from backend.app.services.document_processor import document_processor
from backend.app.services.chunking_service import chunking_service
from backend.app.services.ollama_service import ollama_service
from backend.app.services.qdrant_service import qdrant_service
from backend.app.core.config import settings

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = "spacy",
    embedding_model: str = None
):
    """Upload and process a document"""
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        document_type = document_processor.get_document_type(file.filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        chunks = await document_processor.process_document(
            file_path, 
            file.filename, 
            chunking_method
        )
        
        # Apply semantic chunking
        refined_chunks = chunking_service.apply_chunking(chunks, chunking_method)
        
        # Generate embeddings for chunks
        for chunk in refined_chunks:
            embedding = await ollama_service.get_embedding(chunk.content, embedding_model)
            chunk.embedding = embedding
        
        # Store chunks in Qdrant
        await qdrant_service.add_chunks(refined_chunks)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            document_id=refined_chunks[0].metadata["document_id"],
            filename=file.filename,
            document_type=document_type,
            chunks_created=len(refined_chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database"""
    start_time = time.time()
    
    try:
        # Get query embedding
        query_embedding = await ollama_service.get_embedding(request.query, request.embedding_model)
        
        # Search for similar chunks
        similar_chunks = await qdrant_service.search_similar(
            query_embedding,
            limit=request.max_chunks,
            score_threshold=request.similarity_threshold
        )
        
        """         
        if not similar_chunks:
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information to answer your question.",
                relevant_chunks=[],
                model_used=request.model_name,
                processing_time=time.time() - start_time
            )
         """
        # Combine context from similar chunks
        context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
        
        # Generate response
        answer = await ollama_service.generate_response(
            request.query, 
            context, 
            request.model_name
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            relevant_chunks=similar_chunks,
            model_used=request.model_name,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_status = await ollama_service.check_health()
    qdrant_status = await qdrant_service.check_health()
    
    return HealthResponse(
        status="healthy" if ollama_status and qdrant_status else "unhealthy",
        ollama_available=ollama_status,
        qdrant_available=qdrant_status
    )

@router.get("/models")
async def get_available_models():
    """Get available Ollama models"""
    try:
        models = await ollama_service.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection/info")
async def get_collection_info():
    """Get Qdrant collection information"""
    try:
        info = await qdrant_service.get_collection_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))