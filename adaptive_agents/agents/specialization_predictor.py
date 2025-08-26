"""
Specialization Predictor Agent

Predicts if a query requires domain-specific or specialized knowledge.
Uses only the trained DeBERTa regression model.

Key Features:
- DeBERTa-based fine-tuned regression model
- Returns score from 0.0 (no specialized knowledge needed) to 1.0 (highly specialized)
- No fallbacks or pattern matching - pure model prediction
"""

import torch
from typing import Dict, Any, Optional
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from models.base_predictor import BasePredictor, PredictionResult

logger = logging.getLogger(__name__)

class SpecializationPredictor(BasePredictor):
    """
    Predicts if queries require specialized domain knowledge.
    
    Uses DeBERTa-based regression model trained on specialization datasets.
    Returns only the model prediction score.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        # Set default model path to trained model if not provided
        if model_path is None:
            import os
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_model_path = os.path.join(current_dir, "models", "saved_models", "dr-trained", "domain_relevance_predictor_model")
            if os.path.exists(default_model_path):
                model_path = default_model_path
                logger.info(f"[SPECIALIZATION-PREDICTOR] Using saved model: {model_path}")
            else:
                raise FileNotFoundError(f"No trained model found at {default_model_path}")
        
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            max_length=512
        )
        
        logger.info(f"[SPECIALIZATION-PREDICTOR] Initialized with trained model")
    
    async def predict(self, query: str) -> PredictionResult:
        """
        Predict specialization need for a query using only the trained model.
        
        Args:
            query: The input query to analyze
            
        Returns:
            PredictionResult with score from trained model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
        
        # Tokenize input
        inputs = self._tokenize_text(query)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # For regression model, get the direct 0-1 score (no sigmoid needed)
            raw_score = logits[0][0].item()
            score = max(0.0, min(1.0, raw_score))  # Clamp to [0,1] range
            
        return PredictionResult(
            score=score
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "model_loaded": self.model is not None,
            "purpose": "Specialization need prediction (regression model)"
        }

# Global instance for easy usage
specialization_predictor = SpecializationPredictor()

# Convenience function for direct usage
async def predict_specialization_need(query: str) -> PredictionResult:
    """
    Convenience function to predict specialization need.
    
    Args:
        query: The input query to analyze
        
    Returns:
        PredictionResult with score from trained model
    """
    return await specialization_predictor.predict(query)