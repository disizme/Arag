from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import time
import threading
from datetime import datetime
import asyncio

from shared.models.schemas import (
    DocumentUploadRequest, DocumentUploadResponse, 
    QueryRequest, QueryResponse, HealthResponse, CollectionInfoResponse,
    ChunkingMethod, ReasoningStep, ContextRetrievalRequest, ContextRetrievalResponse
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
    chunking_method: str = Form(...),
    embedding_model: str = Form(...)
):
    """Upload and process a document"""
    start_time = time.time()
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        document_type = document_processor.get_document_type(file.filename)
        print(f"[UPLOAD] Document type: {document_type}")
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[UPLOAD] File saved to: {file_path}")
        # Respond immediately that processing has started
        response = {
            "document_id": "processing",
            "filename": file.filename,
            "status": "processing"
        }
        # Start background processing (simulate async, but you can use a task queue in production)
        def process_file():
            step_start = time.time()
            print(f"[UPLOAD] Extracting text and creating initial chunk...")
            initial_chunk = asyncio.run(document_processor.process_document(file_path, file.filename))
            print(f"[UPLOAD] Text extraction done in {time.time() - step_start:.2f}s")
            step_start = time.time()
            # Convert chunking method string to enum
            if isinstance(chunking_method, str):
                chunking_method_enum = getattr(ChunkingMethod, chunking_method.upper(), ChunkingMethod.RECURSIVE)
            else:
                chunking_method_enum = chunking_method
            print(f"[UPLOAD] Applying chunking method: {chunking_method_enum}")
            refined_chunks = chunking_service.apply_chunking(initial_chunk, chunking_method_enum)
            print(f"[UPLOAD] Chunking {len(refined_chunks)} done in {time.time() - step_start:.2f}s")
            step_start = time.time()
            print("[Upload] Embedding generation started")   
            for chunk in refined_chunks:
                embedding = asyncio.run(ollama_service.get_embedding(chunk.content, embedding_model))
                chunk.embedding = embedding
            print(f"[UPLOAD] Embedding generation done in {time.time() - step_start:.2f}s")
            step_start = time.time()
            asyncio.run(qdrant_service.add_chunks(refined_chunks))
            print(f"[UPLOAD] Qdrant upsert done in {time.time() - step_start:.2f}s")
            os.remove(file_path)
            print(f"[UPLOAD] File {file.filename} processed successfully in {time.time() - start_time:.2f}s")
        threading.Thread(target=process_file, daemon=True).start()
        return response
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        print(f"[UPLOAD] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database"""
    start_time = time.time()
    
    try:
        if request.use_multi_step_reasoning:
            # Multi-step reasoning approach
            async def context_retrieval_func(sub_query: str) -> str:
                """Context retrieval function for multi-step reasoning"""
                # Get embedding for sub-query
                sub_query_embedding = await ollama_service.get_embedding(sub_query, request.embedding_model)
                
                # Search for similar chunks
                chunks = await qdrant_service.search_similar(
                    sub_query_embedding,
                    limit=request.max_chunks,
                    score_threshold=request.similarity_threshold
                )
                
                # Combine context from chunks
                return "\n\n".join([chunk["content"] for chunk in chunks])
            
            # Use multi-step reasoning
            reasoning_result = await ollama_service.multi_step_reasoning(
                query=request.query,
                context_retrieval_func=context_retrieval_func,
                model_name=request.model_name
            )
            
            # Get all chunks used across all steps for the response
            all_chunks = []
            reasoning_steps = []
            
            for step in reasoning_result["reasoning_steps"]:
                # Convert step to ReasoningStep model
                reasoning_step = ReasoningStep(
                    step_number=step["step_number"],
                    sub_question=step["sub_question"],
                    context_used=step["context_used"],
                    step_answer=step["step_answer"]
                )
                reasoning_steps.append(reasoning_step)
                
                # Get chunks for this step to include in response
                step_embedding = await ollama_service.get_embedding(step["sub_question"], request.embedding_model)
                step_chunks = await qdrant_service.search_similar(
                    step_embedding,
                    limit=request.max_chunks,
                    score_threshold=request.similarity_threshold
                )
                all_chunks.extend(step_chunks)
            
            # Remove duplicates from chunks
            unique_chunks = []
            seen_ids = set()
            for chunk in all_chunks:
                if chunk["id"] not in seen_ids:
                    unique_chunks.append(chunk)
                    seen_ids.add(chunk["id"])
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                query=request.query,
                answer=reasoning_result["final_answer"],
                relevant_chunks=unique_chunks,
                model_used=request.model_name,
                processing_time=processing_time,
                reasoning_steps=reasoning_steps,
                num_steps=reasoning_result["num_steps"]
            )
        
        else:
            # Standard single-step reasoning approach
            # Get query embedding
            query_embedding = await ollama_service.get_embedding(request.query, request.embedding_model)
            
            # Search for similar chunks
            similar_chunks = await qdrant_service.search_similar(
                query_embedding,
                limit=request.max_chunks,
                score_threshold=request.similarity_threshold
            )
            
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

@router.post("/retrieve-contexts", response_model=ContextRetrievalResponse)
async def retrieve_contexts(request: ContextRetrievalRequest):
    """Retrieve relevant contexts for a query without generating response"""
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
        
        processing_time = time.time() - start_time
        
        return ContextRetrievalResponse(
            query=request.query,
            contexts=similar_chunks,
            total_chunks=len(similar_chunks),
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