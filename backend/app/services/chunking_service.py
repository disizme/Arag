import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from typing import List, Tuple, Union
from shared.models.schemas import ChunkingMethod, DocumentChunk
from backend.app.core.config import settings

class ChunkingService:
    def __init__(self):
        self.spacy_model = None
        self.semantic_chunker = None
        self.fallback_splitter = None
        self._load_models()
    
    def _load_models(self):
        """Load chunking models"""
        try:
            # Load spaCy model
            self.spacy_model = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        
        try:
            # Initialize LangChain SemanticChunker with Ollama embeddings
            embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_EMBEDDING_MODEL
            )
            self.semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                buffer_size=1,  # Number of sentences to group together
                add_start_index=True,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=95
            )
        except Exception as e:
            print(f"Warning: Could not initialize SemanticChunker: {e}")
        
        # Initialize fallback splitter
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "."]
        )
    
    def chunk_with_recursive(self, text: str) -> List[str]:
        """
        Sentence-aware recursive chunking using spaCy sentence detection
        
        This method maintains sentence boundaries while creating chunks that fit
        within the specified size limits. It uses spaCy's sentence segmentation
        to ensure coherent text chunks.
        
        Key features:
        - Preserves sentence boundaries (never breaks sentences)
        - Respects chunk size limits
        - Maintains reading flow and coherence
        - Includes overlap between chunks for better context continuity
        - Falls back gracefully if spaCy is unavailable
        """
        if not self.spacy_model:
            print("spaCy model not available for sentence detection, using fallback")
            return self._fallback_chunk(text)
        
        try:
            doc = self.spacy_model(text)
            
            # Extract sentences
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if not sentences:
                return [text] if text.strip() else []
            
            chunks = []
            current_chunk_sentences = []
            current_chunk_length = 0
            overlap_sentences = []  # Store sentences for overlap
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # Check if adding this sentence would exceed chunk size
                # We add +1 for the space between sentences
                potential_length = current_chunk_length + sentence_length + (1 if current_chunk_sentences else 0) 
                
                if potential_length <= settings.CHUNK_SIZE:
                    # Add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_chunk_length = potential_length
                    
                else:
                    # Finalize current chunk if it has content
                    if current_chunk_sentences:
                        chunks.append(" ".join(current_chunk_sentences))
                        
                        # Prepare overlap for next chunk
                        overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                    
                    # Start new chunk with overlap + current sentence
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_chunk_length = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) - 1
            
            # Add the final chunk if it has content
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            print(f"Error in recursive sentence-aware chunking: {e}")
            return self._fallback_chunk(text)
    
    def chunk_with_spacy(self, text: str) -> List[str]:
        """Semantic chunking using spaCy NLP pipeline with overlap"""
        if not self.spacy_model:
            return self._fallback_chunk(text)
        
        try:
            doc = self.spacy_model(text)
            
            # Extract sentences with semantic information
            sentences = []
            for sent in doc.sents:
                sentences.append({
                    'text': sent.text.strip(),
                    'start': sent.start_char,
                    'end': sent.end_char,
                    'entities': [ent.label_ for ent in sent.ents],
                    'noun_chunks': [chunk.text for chunk in sent.noun_chunks]
                })
            
            # Group sentences based on semantic similarity and entities
            chunks = []
            current_chunk = []
            current_entities = set()
            current_length = 0
            overlap_sentences = []  # Store sentences for overlap
            
            for sent_info in sentences:
                sent_text = sent_info['text']
                sent_entities = set(sent_info['entities'])
                
                # Check if sentence should start new chunk based on:
                # 1. Length limit
                # 2. Entity overlap (semantic coherence)
                # 3. Topic shift detection
                
                entity_overlap = len(current_entities & sent_entities) > 0 if current_entities else True
                would_exceed_limit = current_length + len(sent_text) > settings.CHUNK_SIZE
                
                if would_exceed_limit or (current_chunk and not entity_overlap and len(current_chunk) > 2):
                    # Finalize current chunk
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        
                        # Prepare overlap for next chunk
                        overlap_sentences = self._get_overlap_sentences(current_chunk)
                    
                    # Start new chunk with overlap + current sentence
                    current_chunk = overlap_sentences + [sent_text]
                    current_entities = sent_entities
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    # Add to current chunk
                    current_chunk.append(sent_text)
                    current_entities.update(sent_entities)
                    current_length += len(sent_text)
            
            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return [chunk.strip() for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            print(f"Error in spaCy chunking: {e}")
            return self._fallback_chunk(text)
    
    def chunk_with_langchain(self, text: str) -> List[str]:
        """Semantic chunking using LangChain SemanticChunker"""
        if not self.semantic_chunker:
            print("SemanticChunker not available, using fallback")
            return self._fallback_chunk(text)
        
        try:
            # Use SemanticChunker for true semantic chunking
            chunks = self.semantic_chunker.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            print(f"Error in LangChain semantic chunking: {e}")
            return self._fallback_chunk(text)
    
    def _fallback_chunk(self, text: str) -> List[str]:
        """Fallback chunking using RecursiveCharacterTextSplitter"""
        if not self.fallback_splitter:
            return self._simple_chunk(text)
        
        try:
            chunks = self.fallback_splitter.split_text(text)
            return [chunk.strip() for chunk in chunks if chunk.strip()]
        except Exception as e:
            print(f"Error in fallback chunking: {e}")
            return self._simple_chunk(text)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on CHUNK_OVERLAP setting"""
        if not sentences or settings.CHUNK_OVERLAP <= 0:
            return []
        
        overlap_sentences = []
        current_overlap_length = 0
        
        # Take sentences from the end working backwards
        for sentence in reversed(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed overlap limit
            if current_overlap_length + sentence_length + (1 if overlap_sentences else 0) <= settings.CHUNK_OVERLAP:
                overlap_sentences.insert(0, sentence)  # Insert at beginning to maintain order
                current_overlap_length += sentence_length + (1 if len(overlap_sentences) > 1 else 0)
            else:
                break
        
        return overlap_sentences
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple word-based chunking method as last resort"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= settings.CHUNK_SIZE:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def apply_chunking(
        self, 
        chunk: DocumentChunk, 
        method: Union[str, ChunkingMethod] = ChunkingMethod.RECURSIVE
    ) -> List[DocumentChunk]:
        """Apply chunking to document chunks using the specified method"""
        refined_chunks = []
        
        # Convert string to enum if needed
        if isinstance(method, str):
            if method.lower() == "recursive":
                text_chunks = self.chunk_with_recursive(chunk.content)
            elif method.lower() == "spacy":
                text_chunks = self.chunk_with_spacy(chunk.content)
            elif method.lower() == "langchain":
                text_chunks = self.chunk_with_langchain(chunk.content)
            else:
                text_chunks = self._fallback_chunk(chunk.content)
        else:
            text_chunks = self._fallback_chunk(chunk.content)
            
        # Create new chunks from the refined text chunks
        for i, text_chunk in enumerate(text_chunks):
            if text_chunk.strip():
                refined_chunk = DocumentChunk(
                    id=f"{chunk.id}_{i}",
                    content=text_chunk.strip(),
                    metadata={
                        **chunk.metadata,
                        "parent_chunk_id": chunk.id,
                        "semantic_chunk_index": i
                    },
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index * 1000 + i,  # Maintain order
                    created_at=chunk.created_at
                )
                refined_chunks.append(refined_chunk)
        
        return refined_chunks

chunking_service = ChunkingService()