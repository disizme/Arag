from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"  # Default - sentence-aware recursive chunking
    SPACY = "spacy"          # Semantic chunking using spaCy
    LANGCHAIN = "langchain"  # Semantic chunking using LangChain
    FALLBACK = "fallback"    # Fallback chunking

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int
    score: Optional[float] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentUploadRequest(BaseModel):
    chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE
    embedding_model: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str

class QueryRequest(BaseModel):
    query: str
    model_name: str = "llama2"
    max_chunks: int = 5
    similarity_threshold: float = 0.7
    embedding_model: Optional[str] = None
    use_multi_step_reasoning: bool = False

class ReasoningStep(BaseModel):
    step_number: int
    sub_question: str
    context_used: str
    step_answer: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    relevant_chunks: List[DocumentChunk]
    model_used: str
    processing_time: float
    reasoning_steps: Optional[List[ReasoningStep]] = None
    num_steps: Optional[int] = None

class ContextRetrievalRequest(BaseModel):
    query: str
    max_chunks: int = 5
    similarity_threshold: float = 0.7
    embedding_model: Optional[str] = None

class ContextRetrievalResponse(BaseModel):
    query: str
    contexts: List[DocumentChunk]
    total_chunks: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    qdrant_available: bool
    timestamp: datetime = Field(default_factory=datetime.now)

class CollectionInfoResponse(BaseModel):
    name: str
    vectors_count: int
    status: str