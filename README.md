# Adaptive RAG System

An intelligent document query system with adaptive retrieval capabilities, built with FastAPI and Streamlit.

## Features

- **Document Upload**: Support for PDF, DOCX, PPTX, and 15+ other formats via Kreuzberg
- **Advanced Text Chunking**: Three chunking methods available with overlap support:
  - **Recursive** (Default): Sentence-aware chunking that preserves sentence boundaries
  - **Semantic spaCy**: NLP-based chunking with entity recognition
  - **Semantic LangChain**: Advanced semantic chunking with embeddings
- **Multi-Step Reasoning**: Advanced query processing for complex questions:
  - **Query Decomposition**: Breaks complex queries into sub-questions
  - **Step-by-Step Analysis**: Retrieves context for each reasoning step
  - **Knowledge Accumulation**: Builds reasoning chain across steps
  - **Comprehensive Synthesis**: Integrates all steps into final answer
- **Unified Document Processing**: Kreuzberg library for robust text extraction
- **Entity Extraction** (Optional): Advanced entity recognition and keyword extraction
- **Vector Search**: Fast similarity search using Qdrant vector database
- **Multiple Models**: Choose from various Ollama models
- **Interactive Interface**: User-friendly Streamlit frontend

## Architecture

### Core Components
- **FastAPI Backend**: RESTful API for document processing
- **Streamlit Frontend**: Interactive web interface
- **Kreuzberg**: Modern document processing library with OCR fallback
- **Ollama**: Local LLM and embedding service
- **Qdrant**: Vector database for semantic search
- **spaCy**: NLP processing for chunking and entity extraction

### Processing Pipeline
1. Document Upload → Multi-format file processing
2. Text Extraction → Kreuzberg unified extraction with OCR fallback
3. Intelligent Chunking → Choice of recursive, semantic, or LangChain methods with overlap
4. Embedding Generation → Ollama vector creation
5. Vector Storage → Qdrant indexing with metadata
6. Query Processing → Two modes available:
   - **Standard Mode**: Direct similarity search + LLM response generation
   - **Multi-Step Mode**: Query decomposition → step-by-step reasoning → synthesis

### Optional Features
- **Entity Extraction Service**: Advanced NLP processing for:
  - Named entity recognition (PERSON, ORG, GPE, etc.)
  - Keyword extraction (frequency-based, spaCy-based, noun phrases)
  - Metadata extraction (title, author, creation date, etc.)
  - Structured data extraction (tables, embedded resources)

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Qdrant vector database
- spaCy English model
// - Tesseract OCR (for Kreuzberg OCR features)
- Pandoc (for advanced metadata extraction)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd arag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Install external dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr pandoc

# macOS
brew install tesseract pandoc

# Windows
choco install tesseract pandoc
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Setup Services

### Ollama Setup
1. Install Ollama from https://ollama.ai
2. Pull required models:
```bash
ollama pull snowflake-arctic-embed2:latest
ollama pull llama2
```

### Qdrant Setup
1. Install Qdrant:
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
pip install qdrant-client
```

## Running the Application

1. Start the backend:
```bash
python run_backend.py
```

2. Start the frontend (in another terminal):
```bash
python run_frontend.py
```

3. Access the application:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Usage

### Document Upload
1. Go to the Document Upload page
2. Select your chunking method:
   - **Recursive** (recommended): Sentence-aware chunking
   - **spaCy**: Semantic chunking with entity recognition
   - **LangChain**: Advanced semantic chunking
3. Upload your documents (PDF, DOCX, PPTX, images, and 15+ other formats)
4. Wait for processing to complete

### Querying Documents
1. Go to the Chat page
2. Select your preferred model
3. Adjust retrieval settings if needed
4. Choose reasoning mode:
   - **Standard Mode**: Quick responses for simple queries
   - **Multi-Step Mode**: Complex reasoning for analytical questions
5. Ask questions about your uploaded documents

### Optional Entity Extraction Service

The system includes an optional advanced entity extraction service for deeper document analysis:

```python
from backend.app.services.entity_extractor import KreuzbergEntityExtractor

# Initialize the extractor
extractor = KreuzbergEntityExtractor()

# Comprehensive extraction
result = extractor.extract_comprehensive_data("/path/to/document.pdf")

# Access extracted data
print(f"Title: {result['metadata'].get('title')}")
print(f"Keywords: {result['keywords']['spacy_keywords'][:5]}")
print(f"Entities: {result['entities']['spacy_entities'].get('PERSON', [])}")

# Extract only specific data types
keywords = extractor.extract_keywords_only("/path/to/document.pdf")
entities = extractor.extract_entities_only("/path/to/document.pdf")
metadata = extractor.extract_metadata_only("/path/to/document.pdf")
```

## API Endpoints

- `POST /api/v1/upload` - Upload and process documents
- `POST /api/v1/query` - Query the document database (supports multi-step reasoning)
- `GET /api/v1/health` - Check system health
- `GET /api/v1/models` - List available models
- `GET /api/v1/collection/info` - Get collection information

### Query API Usage

#### Standard Query
```json
{
  "query": "What is machine learning?",
  "model_name": "llama2",
  "max_chunks": 5,
  "similarity_threshold": 0.7,
  "use_multi_step_reasoning": false
}
```

#### Multi-Step Reasoning Query
```json
{
  "query": "How do neural networks learn and what are the key optimization techniques used in deep learning?",
  "model_name": "llama2",
  "max_chunks": 5,
  "similarity_threshold": 0.7,
  "use_multi_step_reasoning": true
}
```

#### Multi-Step Response Format
```json
{
  "query": "...",
  "answer": "Comprehensive synthesized answer",
  "relevant_chunks": [...],
  "model_used": "llama2",
  "processing_time": 15.2,
  "reasoning_steps": [
    {
      "step_number": 1,
      "sub_question": "How do neural networks learn?",
      "context_used": "Retrieved context for this step",
      "step_answer": "Step-specific answer"
    },
    {
      "step_number": 2,
      "sub_question": "What are key optimization techniques?",
      "context_used": "Retrieved context for this step",
      "step_answer": "Step-specific answer"
    }
  ],
  "num_steps": 2
}
```

## Development

### Project Structure
```
arag/
├── backend/
│   └── app/
│       ├── api/          # API routes
│       ├── core/         # Configuration
│       ├── services/     # Business logic
│       │   ├── document_processor.py    # Kreuzberg-based document processing
│       │   ├── chunking_service.py      # Text chunking methods
│       │   ├── entity_extractor.py      # Optional entity extraction
│       │   ├── ollama_service.py        # LLM integration with multi-step reasoning
│       │   └── qdrant_service.py        # Vector database
│       └── main.py       # FastAPI app
├── frontend/
│   ├── pages/           # Streamlit pages
│   ├── utils/           # Utilities
│   └── Home.py          # Main page
├── shared/
│   └── models/          # Shared data models
└── requirements.txt
```

### Running Tests
```bash
# Add test commands here when tests are implemented
```

## Configuration

Key configuration options in `.env`:

- `OLLAMA_BASE_URL`: Ollama service URL
- `OLLAMA_EMBEDDING_MODEL`: Model for embeddings
- `QDRANT_HOST`: Qdrant database host
- `QDRANT_PORT`: Qdrant database port
- `CHUNK_SIZE`: Text chunk size for processing
- `CHUNK_OVERLAP`: Overlap between chunks
- `MAX_FILE_SIZE_MB`: Maximum upload file size

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Ollama not available**:
   - Check if Ollama is running
   - Verify models are downloaded
   - Check OLLAMA_BASE_URL in .env

3. **Qdrant connection issues**:
   - Ensure Qdrant is running
   - Check QDRANT_HOST and QDRANT_PORT in .env

4. **Kreuzberg extraction issues**:
   - Ensure Tesseract OCR is installed for image-based documents
   - Install Pandoc for advanced metadata extraction
   - Check file format support (18+ formats supported)

5. **Entity extraction not working**:
   - Verify spaCy model is downloaded: `python -m spacy download en_core_web_sm`
   - Check if Kreuzberg entity extraction features are enabled

### New Features (Latest Update)

#### Multi-Step Reasoning System
- **Query Decomposition**: Automatically breaks complex queries into manageable sub-questions
- **Contextual Retrieval**: Retrieves relevant context for each reasoning step
- **Knowledge Accumulation**: Previous steps inform subsequent reasoning
- **Comprehensive Synthesis**: Integrates all steps into a coherent final answer
- **Detailed Tracing**: Full visibility into the reasoning process

#### Enhanced Chunking with Overlap
- **Chunk Overlap**: Configurable overlap between chunks for better context continuity
- **Sentence Boundary Preservation**: Maintains semantic coherence in recursive chunking
- **Adaptive Overlap**: Smart overlap calculation based on sentence boundaries

#### Document Processing Improvements
- **Kreuzberg Integration**: Replaced multiple document processing libraries with unified Kreuzberg solution
- **Enhanced Format Support**: Now supports 18+ document formats including images
- **OCR Fallback**: Automatic OCR processing for image-based documents

#### Advanced Chunking Options
- **Recursive Chunking**: New default method that preserves sentence boundaries
- **Improved Semantic Chunking**: Enhanced spaCy and LangChain-based methods
- **Smart Fallback**: Graceful degradation when NLP models are unavailable

#### Optional Entity Extraction Service
- **Named Entity Recognition**: Extract PERSON, ORG, GPE, and other entities
- **Keyword Extraction**: Multiple methods (frequency-based, spaCy-based, noun phrases)
- **Metadata Extraction**: Document metadata (title, author, dates, etc.)
- **Structured Data**: Extract tables and embedded resources

## License

This project is part of a thesis on Adaptive Retrieval Augmented Generation.