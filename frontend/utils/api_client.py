import requests
import streamlit as st
from typing import Dict, Any, List, Optional
import io

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def query_documents(self, query: str, model_name: str = "llama3.2:3b", embedding_model: str = "snowflake-arctic-embed2:latest", max_chunks: int = 5, similarity_threshold: float = 0.3, agent_type: str = "adaptive-rag") -> Dict[str, Any]:
        """Query the document database"""
        try:
            data = {
                "query": query,
                "model_name": model_name,
                "max_chunks": max_chunks,
                "similarity_threshold": similarity_threshold,
                "embedding_model": embedding_model
            }
            api_route = f"{self.base_url}/query-{agent_type}"
            response = requests.post(
                api_route,
                json=data,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to query documents: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to check health: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get available models"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            return response.json()["models"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get models: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            response = requests.get(f"{self.base_url}/collection/info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get collection info: {str(e)}")
    
# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()