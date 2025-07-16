import streamlit as st
from frontend.utils.api_client import get_api_client
import time
import re

st.set_page_config(
    page_title="Chat - Adaptive RAG",
    page_icon="💬",
    layout="wide"
)

def parse_ollama_response(response_text):
    """Parse Ollama response to extract thinking process and main answer"""
    # Check if response contains <think> tags
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        # Remove the <think> section from the main answer
        main_answer = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
        return thinking, main_answer
    else:
        # No thinking process found, return original response
        return None, response_text

def display_assistant_message(content, sources=None):
    """Display assistant message with thinking process if available"""
    thinking, main_answer = parse_ollama_response(content)
    
    # Display main answer
    st.markdown(main_answer)
    
    # Display thinking process if available
    if thinking:
        with st.expander("🧠 Thinking Process", expanded=False):
            st.markdown(f"*{thinking}*")
    
    # Show sources if available
    if sources:
        with st.expander("📚 Sources"):
            for j, source in enumerate(sources):
                st.markdown(f"**Source {j+1}:** {source['source_file']} (Page {source.get('page_number', 'N/A')})")
                st.markdown(f"*Score: {source.get('score', 'N/A'):.3f}*")
                st.markdown(f"```\n{source['content'][:200]}...\n```")

# Custom CSS for reduced padding and collapsible sidebar
st.markdown("""
<style>
    .block-container {
        padding: 2rem 2rem 2rem 2rem !important;
    }
    
    .sidebar .sidebar-content {
        width: 300px;
    }
    
    .sidebar-toggle {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 999;
        background: #ff4b4b;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .sidebar-minimized .sidebar .sidebar-content {
        width: 0px;
        overflow: hidden;
    }
    
    .main .block-container {
        max-width: none;
    }
    
</style>
""", unsafe_allow_html=True)

st.title("💬 Chat with Documents")
st.markdown("Ask questions about your uploaded documents")

# Initialize API client
api_client = get_api_client()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True


# Sidebar for settings (conditionally shown)
if st.session_state.sidebar_visible:
    with st.sidebar:
        st.header("Chat Settings")
        
        # Get available models
        try:
            models = api_client.get_available_models()
            if models:
                # Separate LLM and embedding models
                llm_models = [model for model in models if 'embed' not in model.lower()]
                
                if not llm_models:
                    llm_models = models
                
                # Add empty option at the beginning
                model_options = ["default"] + llm_models
                selected_model = st.selectbox(
                    "LLM Model", 
                    model_options, 
                    help="Select a model or leave empty for default(qwen3:latest)")
                
            else:
                st.error("No models available")
                selected_model = "default"
        except Exception as e:
            st.error(f"Failed to get models: {str(e)}")
            selected_model = "default"
        
        max_chunks = st.slider(
            "Max Chunks",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of document chunks to retrieve"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="Minimum similarity score for chunk retrieval"
        )
        
        st.divider()
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # System status
        st.subheader("System Status")
        try:
            health = api_client.check_health()
            if health["status"] == "healthy":
                st.success("✅ System Ready")
            else:
                st.error("❌ System Issues")
        except:
            st.error("❌ Cannot connect to API")
else:
    # Default values when sidebar is hidden
    selected_model = "default"
    max_chunks = 5
    similarity_threshold = 0.7

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant"):
                    display_assistant_message(message['content'], message.get("sources"))
                    

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })
    
    # Show loading spinner
    with st.spinner("Thinking..."):
        try:
            # Query the API
            model_to_use = selected_model if selected_model != "default" else "qwen3:latest"
            response = api_client.query_documents(
                query=query,
                model_name=model_to_use,
                max_chunks=max_chunks,
                similarity_threshold=similarity_threshold,
            )
            
            # Add assistant response to chat history
            with st.chat_message("assistant"):
                display_assistant_message(response["answer"], response.get("relevant_chunks"))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("relevant_chunks", [])
            })
            
        except Exception as e:
            st.error(f"Failed to get response: {str(e)}")
            with st.chat_message("assistant"):
                st.markdown(f"Sorry, I encountered an error: {str(e)}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
    
    # Rerun to show the new messages
    st.rerun()

__ = """ 
with col2:
    st.header("Chat Guide")
    
    st.info(
    **Tips for better results:**
    - Ask specific questions
    - Include context when possible
    - Use keywords from your documents
    - Be clear and concise
    )
     """

_ = """    
    st.markdown("---")

    st.subheader("Sample Questions")
    
    sample_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the conclusions?",
        "Explain the methodology used",
        "What are the implications?"
    ]
    
   
for question in sample_questions:
        if st.button(question, key=f"sample_{question}"):
            # Add the sample question to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            st.rerun()
 """