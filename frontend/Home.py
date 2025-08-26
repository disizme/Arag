import streamlit as st
from utils.api_client import get_api_client

st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for reduced padding
st.markdown("""
<style>
    .block-container {
        padding: 2rem 2rem 2rem 2rem !important;
    }
    
    .main .block-container {
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = get_api_client()

# Header
st.title("🤖 Adaptive RAG System")
st.markdown("An intelligent document query system with adaptive retrieval")

# Hero section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to the Adaptive RAG System
    
    This system combines the power of Large Language Models with intelligent document retrieval 
    to provide accurate, context-aware answers to your questions.
    
    ### Key Features:
    - **Document Upload**: Support for PDF, DOCX, and PPTX files
    - **Semantic Chunking**: Intelligent text segmentation using spaCy or LangChain
    - **Vector Search**: Fast similarity search using Qdrant vector database
    - **Multiple Models**: Choose from various Ollama models
    - **Adaptive Retrieval**: Smart context selection based on query complexity
    """)

with col2:
    st.markdown("### 🚀 Quick Start")
    
    st.markdown("""
    1. **Upload Documents** 📄
       - Go to Document Upload page
       - Upload your PDF/DOCX/PPTX files
       - Choose chunking method
    
    2. **Start Chatting** 💬
       - Go to Chat page
       - Ask questions about your documents
       - Get intelligent responses
    """)

# System Status
st.markdown("---")
st.header("🔧 System Status")

col1, col2 = st.columns(2)

with col1:
    st.subheader("API Health")
    try:
        health = api_client.check_health()
        if health["status"] == "healthy":
            st.success("✅ API is running")
        else:
            st.error("❌ API issues detected")
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")

with col2:
    st.subheader("Services")
    try:
        health = api_client.check_health()
        st.write(f"**Ollama:** {'✅' if health.get('ollama_available') else '❌'}")
        st.write(f"**Qdrant:** {'✅' if health.get('qdrant_available') else '❌'}")
    except:
        st.write("**Ollama:** ❓")
        st.write("**Qdrant:** ❓")


# Knowledge Base Status
st.markdown("---")
st.header("📚 Knowledge Base")

try:
    collection_info = api_client.get_collection_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents", collection_info.get("vectors_count", 0))
    
    with col2:
        st.metric("Collection", collection_info.get("name", "N/A"))
    
    with col3:
        st.metric("Status", collection_info.get("status", "Unknown"))
        
except Exception as e:
    st.error(f"Cannot fetch collection info: {str(e)}")

# Architecture Overview
st.markdown("---")
st.header("🏗️ System Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Core Components:
    - **FastAPI Backend**: RESTful API for document processing
    - **Streamlit Frontend**: Interactive web interface
    - **Ollama**: Local LLM and embedding service
    - **Qdrant**: Vector database for semantic search
    - **Document Processors**: PDF, DOCX, PPTX support
    """)

with col2:
    st.markdown("""
    ### Processing Pipeline:
    1. **Document Upload** → File processing
    2. **Text Extraction** → Content extraction
    3. **Semantic Chunking** → Intelligent segmentation
    4. **Embedding Generation** → Vector creation
    5. **Vector Storage** → Qdrant indexing
    6. **Query Processing** → Similarity search + LLM
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Adaptive RAG System v1.0 | Built with FastAPI & Streamlit</p>
</div>
""", unsafe_allow_html=True)