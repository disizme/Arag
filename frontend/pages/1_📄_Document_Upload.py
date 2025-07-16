import streamlit as st
import os
from frontend.utils.api_client import get_api_client

st.set_page_config(
    page_title="Document Upload - Adaptive RAG",
    page_icon="📄",
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

st.title("📄 Document Upload")
st.markdown("Upload documents to build your knowledge base")

# Initialize API client
api_client = get_api_client()

# Sidebar for settings
with st.sidebar:
    st.header("Upload Settings")
    
    chunking_method = st.selectbox(
        "Chunking Method",
        ["recursive", "spacy", "langchain"],
        help="Choose the method for text chunking"
    )
    
    # Embedding model selection
    try:
        models = api_client.get_available_models()
        embedding_models = [model for model in models if 'embed' in model.lower()]
        if not embedding_models:
            embedding_models = models  # Fallback to all models
        
        selected_embedding_model = st.selectbox(
            "Embedding Model",
            embedding_models,
            help="Choose the model for generating embeddings"
        )
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        selected_embedding_model = "snowflake-arctic-embed2:latest"
    
    st.divider()
    
    # Check system health
    if st.button("Check System Health"):
        try:
            health = api_client.check_health()
            if health["status"] == "healthy":
                st.success("✅ System is healthy")
                st.write(f"Ollama: {'✅' if health['ollama_available'] else '❌'}")
                st.write(f"Qdrant: {'✅' if health['qdrant_available'] else '❌'}")
            else:
                st.error("❌ System is unhealthy")
                st.write(f"Ollama: {'✅' if health['ollama_available'] else '❌'}")
                st.write(f"Qdrant: {'✅' if health['qdrant_available'] else '❌'}")
        except Exception as e:
            st.error(f"Failed to check health: {str(e)}")
    
    # Collection info
    if st.button("Collection Info"):
        try:
            info = api_client.get_collection_info()
            st.json(info)
        except Exception as e:
            st.error(f"Failed to get collection info: {str(e)}")

# Main upload interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'pptx'],
        accept_multiple_files=False,
        help="Supported formats: PDF, DOCX, PPTX"
    )
    
    if uploaded_file:
        with st.expander(f"📄 {uploaded_file.name}"):
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {uploaded_file.type}")
        
        if st.button("Upload Document", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text(f"Processing {uploaded_file.name}...")
                file_content = uploaded_file.read()
                result = api_client.upload_document(
                    file_content, 
                    uploaded_file.name, 
                    chunking_method,
                    selected_embedding_model
                )
                st.success(f"✅ {uploaded_file.name} upload started")
                st.json(result)
            except Exception as e:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {str(e)}")
            uploaded_file.seek(0)
            progress_bar.progress(1.0)
            status_text.text("Done!")

with col2:
    st.header("Instructions")
    
    st.info("""
    **How to use:**
    1. Select your chunking method (recursive recommended)
    2. Upload one or more documents
    3. Wait for processing to complete
    4. Use the Chat page to query your documents
    """)
    
    st.markdown("---")
    
    st.subheader("Supported Formats")
    st.markdown("""
    - **PDF**: Text extraction with page numbers
    - **DOCX**: Microsoft Word documents
    - **PPTX**: PowerPoint presentations
    """)
    
    st.markdown("---")
    
    st.subheader("Chunking Methods")
    st.markdown("""
    - **Recursive**: Sentence-aware recursive chunking (default)
    - **spaCy**: Semantic chunking using spaCy NLP
    - **LangChain**: Semantic chunking using LangChain
    """)

# Footer
st.markdown("---")
st.markdown("🤖 Adaptive RAG System - Document Upload Interface")