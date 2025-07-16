import streamlit as st
import json
import time
from typing import Dict, List, Any
import sys
import os

# Add the parent directory to the path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Context Retrieval",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Context Retrieval")
st.markdown("Retrieve and examine relevant contexts for your queries without generating responses.")

# Initialize API client
api_client = get_api_client()

def display_context_chunk(chunk: Dict[str, Any], index: int):
    """Display a single context chunk"""
    with st.expander(f"📄 Context {index + 1} - Score: {chunk.get('score', 0):.3f}", expanded=index < 3):
        
        # Content
        st.markdown("**Content:**")
        st.markdown(f"```\n{chunk['content']}\n```")
        
        # Metadata in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Source:**")
            st.text(chunk.get('source_file', 'Unknown'))
            
        with col2:
            st.markdown("**Page:**")
            st.text(chunk.get('page_number', 'N/A'))
            
        with col3:
            st.markdown("**Chunk Index:**")
            st.text(chunk.get('chunk_index', 'N/A'))
        
        # Additional metadata
        if chunk.get('metadata'):
            st.markdown("**Metadata:**")
            st.json(chunk['metadata'])

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Query Input")
    query = st.text_area(
        "Enter your query:",
        placeholder="What would you like to search for?",
        height=100
    )

with col2:
    st.subheader("Retrieval Settings")
    
    max_chunks = st.slider(
        "Maximum chunks to retrieve:",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of relevant chunks to retrieve"
    )
    
    similarity_threshold = st.slider(
        "Similarity threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum similarity score for retrieved chunks"
    )
    
    embedding_model = st.selectbox(
        "Embedding model:",
        options=["snowflake-arctic-embed2:latest", "mxbai-embed-large:latest"],
        index=0,
        help="Embedding model to use (None = default)"
    )

# Retrieve button
if st.button("🔍 Retrieve Contexts", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving contexts..."):
            try:
                result = api_client.retrieve_contexts(query, max_chunks, similarity_threshold, embedding_model)
            except Exception as e:
                st.error(f"Error retrieving contexts: {str(e)}")
                result = None
            
            if result:
                st.success(f"Retrieved {result['total_chunks']} contexts in {result['processing_time']:.2f}s")
                
                # Display query
                st.subheader("Query")
                st.markdown(f"**\"{result['query']}\"**")
                
                # Display contexts
                if result['contexts']:
                    st.subheader(f"Retrieved Contexts ({result['total_chunks']})")
                    
                    # Sort by score (highest first)
                    sorted_contexts = sorted(result['contexts'], key=lambda x: x.get('score', 0), reverse=True)
                    
                    for i, context in enumerate(sorted_contexts):
                        display_context_chunk(context, i)
                    
                    # Context analysis
                    st.subheader("📊 Context Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_score = sum(c.get('score', 0) for c in result['contexts']) / len(result['contexts'])
                        st.metric("Average Score", f"{avg_score:.3f}")
                    
                    with col2:
                        sources = set(c.get('source_file', 'Unknown') for c in result['contexts'])
                        st.metric("Unique Sources", len(sources))
                    
                    with col3:
                        total_chars = sum(len(c.get('content', '')) for c in result['contexts'])
                        st.metric("Total Characters", total_chars)
                    
                    # Source breakdown
                    st.subheader("📚 Source Breakdown")
                    source_counts = {}
                    for context in result['contexts']:
                        source = context.get('source_file', 'Unknown')
                        source_counts[source] = source_counts.get(source, 0) + 1
                    
                    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{source}**: {count} chunk{'s' if count > 1 else ''}")
                    
                    # Export option
                    st.subheader("📥 Export")
                    export_data = {
                        "query": result['query'],
                        "contexts": result['contexts'],
                        "total_chunks": result['total_chunks'],
                        "processing_time": result['processing_time'],
                        "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.download_button(
                        label="Download as JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"contexts_{int(time.time())}.json",
                        mime="application/json"
                    )
                    
                else:
                    st.warning("No contexts found matching your query and similarity threshold.")
                    st.info("Try lowering the similarity threshold or using different search terms.")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    **How to use Context Retrieval:**
    
    1. **Enter your query**: Type the question or topic you want to search for
    2. **Adjust settings**: 
       - **Max chunks**: More chunks = more comprehensive results but slower processing
       - **Similarity threshold**: Lower values = more results but potentially less relevant
       - **Embedding model**: Choose the model for generating embeddings
    3. **Retrieve contexts**: Click the button to search your document database
    4. **Analyze results**: Review the retrieved contexts, scores, and source information
    5. **Export data**: Download the results as JSON for further analysis
    
    **Tips for better results:**
    - Use specific keywords related to your topic
    - Try different similarity thresholds (0.5-0.8 typically work well)
    - Look at the scores to understand relevance
    - Check source files to understand where information comes from
    
    **Understanding scores:**
    - Scores range from 0 to 1 (higher = more relevant)
    - Typical good matches score above 0.7
    - Scores below 0.5 may indicate weak relevance
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use this tool to understand what contexts your queries retrieve before generating responses in the Chat page.")