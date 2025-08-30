from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import time
import threading
from datetime import datetime
import asyncio

from adaptive_agents.agents.adaptive_wrapper import adaptive_wrapper
from shared.models.schemas import (
    DocumentUploadRequest, DocumentUploadResponse, 
    QueryRequest, QueryResponse, HealthResponse, CollectionInfoResponse,
    ChunkingMethod, ReasoningStep, ContextRetrievalRequest, ContextRetrievalResponse
)
from backend.app.services.document_processor import document_processor
from backend.app.services.chunking_service import chunking_service
from backend.app.services.ollama_service import ollama_service
from backend.app.services.bge_service import bge_service
from backend.app.services.qdrant_service import qdrant_service
from backend.app.core.config import settings


router = APIRouter()

# Helper functions for query processing
async def process_no_fetch_query(request: QueryRequest) -> QueryResponse:
    """Process query with direct LLM (no retrieval)"""
    start_time = time.time()
    
    # Direct LLM without context
    answer = await ollama_service.generate_response(
        request.query, 
        None, 
        request.model_name
    )
    
    processing_time = time.time() - start_time
    
    return QueryResponse(
        query=request.query,
        answer=answer,
        relevant_chunks=[],
        processing_time=processing_time,
    )

async def process_shallow_fetch_query(request: QueryRequest) -> QueryResponse:
    """Process query using sparse retrieval (BGE-M3)"""
    start_time = time.time()
    
    # Search using sparse vectors (BGE-M3)
    similar_chunks = await qdrant_service.search_shallow(
        query_text=request.query,
        limit=request.max_chunks,
        score_threshold=request.similarity_threshold
    )
    
    # Combine context from chunks
    context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
    
    # Generate response with context
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
        processing_time=processing_time,
    )

async def process_dense_fetch_query(request: QueryRequest) -> QueryResponse:
    """Process query using dense retrieval (BGE-M3 embeddings)"""
    start_time = time.time()
    
    # Search for similar chunks using dense vectors
    similar_chunks = await qdrant_service.search_dense(
        query_text=request.query,
        limit=request.max_chunks,
        score_threshold=request.similarity_threshold
    )
    
    # Combine context from chunks
    context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
    
    # Generate response with context
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
        processing_time=processing_time,
    )

async def process_hybrid_fetch_query(request: QueryRequest) -> QueryResponse:
    """Process query using hybrid retrieval (BGE-M3 sparse + dense)"""
    start_time = time.time()
    
    # Search using hybrid approach with BGE-M3
    similar_chunks = await qdrant_service.search_hybrid(
        query_text=request.query,
        limit=request.max_chunks,
        score_threshold=request.similarity_threshold
    )
    
    # Combine context from chunks
    context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
    
    # Generate response with context
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
        processing_time=processing_time,
    )

async def process_multi_fetch_query(request: QueryRequest) -> QueryResponse:
    """Process a multi-step reasoning query with hybrid context retrieval"""
    start_time = time.time()
    
    # Context retrieval function using hybrid search for multi-step reasoning
    async def context_retrieval_func(sub_query: str) -> str:        
        # Use hybrid search for context retrieval in multi-step (BGE-M3)
        chunks = await qdrant_service.search_hybrid(
            query_text=sub_query,
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
        
        # Get chunks for this step to include in response using hybrid search (BGE-M3)
        step_chunks = await qdrant_service.search_hybrid(
            query_text=step["sub_question"],
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
        processing_time=processing_time,
        reasoning_steps=reasoning_steps,
        num_steps=reasoning_result["num_steps"],
    )

@router.post("/query-vanilla", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database with optional adaptive agent"""
    try:
        response = await process_dense_fetch_query(request)
        response.agent_decision = None
        return response
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query-classifier-rag", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database with optional adaptive agent"""
    try:
        # Get decision from adaptive classifier: no fetch, single fetch, multi fetch
        decision = await adaptive_wrapper.predict_query_complexity(request.query)
        
        # Route to appropriate processing function based on strategy
        if decision.complexity.label == "no_fetch":
            response = await process_no_fetch_query(request)
        elif decision.complexity.label == "dense_fetch":
            response = await process_dense_fetch_query(request)
        elif decision.complexity.label == "multi_fetch":
            response = await process_multi_fetch_query(request)

        # Add agent decision information to response (convert to dict format expected by schema)
        response.agent_decision = {
            "strategy": decision.complexity.label,
            "processing_time_ms": decision.processing_time_ms
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query-adaptive-rag", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document database with optional adaptive agent"""
    try:
        # Get decision from adaptive classifier: no fetch, single fetch, multi fetch
        decision = await adaptive_wrapper.analyze_query(request.query)

        # Route to appropriate processing function based on strategy
        if decision.strategy == "no_fetch":
            response = await process_no_fetch_query(request)
        elif decision.strategy == "shallow_fetch":
            response = await process_shallow_fetch_query(request)
        elif decision.strategy == "dense_fetch":
            response = await process_dense_fetch_query(request)
        elif decision.strategy == "hybrid_fetch":
            response = await process_hybrid_fetch_query(request)
        elif decision.strategy == "multi_fetch":
            response = await process_multi_fetch_query(request)
        # Add agent decision information to response (convert to dict format expected by schema)
        response.agent_decision = {
            "strategy": decision.strategy,
            "hallucination_risk": decision.hallucination_risk.score,
            "specialization_need": decision.specialization_need.score,
            "processing_time_ms": decision.processing_time_ms
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_status = await ollama_service.check_health()
    qdrant_status = await qdrant_service.check_health()
    bge_status = await bge_service.check_health()
    
    return HealthResponse(
        status="healthy" if ollama_status and qdrant_status and bge_status else "unhealthy",
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