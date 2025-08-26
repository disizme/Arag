"""
Base Predictor Class

Abstract base class for all predictor agents in the adaptive RAG system.
Provides common functionality for model loading, prediction, and device management.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Base class for prediction results"""
    score: float

class BasePredictor(ABC):
    """
    Abstract base class for all predictor agents.
    
    Provides common functionality:
    - Device management (CPU/CUDA/MPS)
    - Model loading and caching
    - Tokenization utilities
    - Batch processing support
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 1
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Auto-detect best available device
        self.device = self._get_optimal_device(device)
        logger.info(f"[{self.__class__.__name__}] Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _get_optimal_device(self, preferred_device: Optional[str] = None) -> str:
        """Auto-detect the best available device for model inference"""
        if preferred_device:
            return preferred_device
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            model_source = self.model_path or self.model_name
            
            logger.info(f"[{self.__class__.__name__}] Loading model: {model_source}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"[{self.__class__.__name__}] Model loaded successfully")
            
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Error loading model: {e}")
            raise
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text for model processing (matches training format)"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",           # Match training: pad to exact max_length
            max_length=self.max_length,
            add_special_tokens=True,        # Match training: explicit special tokens
            return_tensors="pt",
            return_attention_mask=True      # Match training: explicit attention mask
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def _predict_batch(self, texts: list[str]) -> torch.Tensor:
        """Process a batch of texts through the model (matches training format)"""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",           # Match training: pad to exact max_length
            max_length=self.max_length,
            add_special_tokens=True,        # Match training: explicit special tokens
            return_tensors="pt",
            return_attention_mask=True      # Match training: explicit attention mask
        )
        
        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits
    
    @abstractmethod
    async def predict(self, query: str) -> PredictionResult:
        """
        Make a prediction for a single query.
        
        Args:
            query: The input query to analyze
            
        Returns:
            PredictionResult containing score, confidence, and reasoning
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"[{self.__class__.__name__}] Resources cleaned up")
