import ollama
import asyncio
from typing import List, Dict, Any
from backend.app.core.config import settings

class OllamaService:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get embedding for a text using Ollama"""
        try:
            embedding_model = model or settings.OLLAMA_EMBEDDING_MODEL
            response = await asyncio.to_thread(
                self.client.embeddings,
                model=embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")
    
    async def generate_response(self, query: str, context: str, model_name: str) -> str:
        """Generate response using Ollama with context"""
        try:
            prompt = f"""Context: {context}
            
Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=model_name,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = await asyncio.to_thread(self.client.list)
            return [model['name'] for model in response['models']]
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")
    
    async def check_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            await asyncio.to_thread(self.client.list)
            return True
        except:
            return False

ollama_service = OllamaService()