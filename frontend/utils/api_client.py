import requests
import streamlit as st
from typing import Dict, Any, List, Optional
import io

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def upload_document(self, file_content: bytes, filename: str, chunking_method: str = "spacy", embedding_model: str = None) -> Dict[str, Any]:
        """Upload a document to the backend"""
        try:
            files = {"file": (filename, io.BytesIO(file_content), "application/octet-stream")}
            data = {"chunking_method": chunking_method}
            if embedding_model:
                data["embedding_model"] = embedding_model
            
            response = requests.post(
                f"{self.base_url}/upload",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout for large files
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to upload document: {str(e)}")
    
    def query_documents(self, query: str, model_name: str = "qwen3:latest", max_chunks: int = 5, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Query the document database"""
        try:
            data = {
                "query": query,
                "model_name": model_name,
                "max_chunks": max_chunks,
                "similarity_threshold": similarity_threshold
            }
      
            response = requests.post(
                f"{self.base_url}/query",
                json=data,
                timeout=180  # 3 minute timeout
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