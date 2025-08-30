import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
from FlagEmbedding import BGEM3FlagModel

@dataclass
class BGEEmbeddings:
    """Container for BGE-M3 embeddings"""
    dense: List[float]
    sparse: Dict[int, float]

class BGEService:
    """BGE-M3 embedding service for both dense and sparse vectors"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.model = None
        self._device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the BGE-M3 model"""
        try:
            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            
            print(f"[BGE] Initializing BGE-M3 model on {self._device}")
            
            # Initialize model with device and fp16 settings
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self._device
            )
            
            print(f"[BGE] BGE-M3 model loaded successfully")
            
        except Exception as e:
            print(f"[BGE] Error initializing BGE-M3 model: {str(e)}")
            raise Exception(f"Failed to initialize BGE-M3 model: {str(e)}")
    
    async def get_embeddings(self, text: str) -> BGEEmbeddings:
        """Get both dense and sparse embeddings for a text"""
        try:
            # Run model inference in thread to avoid blocking
            embeddings_data = await asyncio.to_thread(
                self._encode_text, text
            )
            
            return BGEEmbeddings(
                dense=embeddings_data['dense_vecs'][0].tolist(),
                sparse=embeddings_data['lexical_weights'][0]
            )
            
        except Exception as e:
            raise Exception(f"Failed to get BGE embeddings: {str(e)}")
    
    async def get_dense_embedding(self, text: str) -> List[float]:
        """Get only dense embedding for a text"""
        embeddings = await self.get_embeddings(text)
        return embeddings.dense
    
    async def get_sparse_vector(self, text: str) -> Dict[int, float]:
        """Get only sparse vector for a text"""
        embeddings = await self.get_embeddings(text)
        return embeddings.sparse
    
    def _encode_text(self, text: str) -> Dict[str, Any]:
        """Encode text using BGE-M3 model (blocking call)"""
        if not self.model:
            raise Exception("BGE-M3 model not initialized")
        
        # Use BGE-M3 encode method to get both dense and sparse
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # We don't need colbert for now
        )
        
        return result
        
    async def check_health(self) -> bool:
        """Check if the BGE service is working"""
        try:
            test_embedding = await self.get_dense_embedding("test")
            return len(test_embedding) == 1024
        except:
            return False

# Global service instance
bge_service = BGEService()