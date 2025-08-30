"""
Hallucination Predictor Agent

Predicts if the core LLM system might hallucinate when given a specific query.
Uses only the trained DeBERTa regression model.

Key Features:
- DeBERTa-based fine-tuned regression model
- Returns score from 0.0 (no risk) to 1.0 (high risk)
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

class HallucinationPredictor(BasePredictor):
    """
    Predicts likelihood of LLM hallucination for given queries.
    
    Uses DeBERTa-based regression model trained on hallucination datasets.
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
            #default_model_path = os.path.join(current_dir, "models", "saved_models", "hr-trained", "hallucination_predictor_model")
            default_model_path = os.path.join(current_dir, "models", "v2", "hallucination_predictor_v2")
            if os.path.exists(default_model_path):
                model_path = default_model_path
                logger.info(f"[HALLUCINATION-PREDICTOR] Using saved model: {model_path}")
            else:
                raise FileNotFoundError(f"No trained model found at {default_model_path}")
        
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            device=device,
            max_length=512
        )
        
        logger.info(f"[HALLUCINATION-PREDICTOR] Initialized with trained model")
    
    async def predict(self, query: str) -> PredictionResult:
        """
        Predict hallucination risk for a query using only the trained model.
        
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
            "purpose": "Hallucination risk prediction (regression model)"
        }

# Global instance for easy usage
hallucination_predictor = HallucinationPredictor()

# Convenience function for direct usage
async def predict_hallucination_risk(query: str) -> PredictionResult:
    """
    Convenience function to predict hallucination risk.
    
    Args:
        query: The input query to analyze
        
    Returns:
        PredictionResult with score from trained model
    """
    return await hallucination_predictor.predict(query)