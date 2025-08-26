#!/usr/bin/env python3
"""
Simple query complexity predictor for Adaptive-RAG
Loads trained T5 model and predicts A/B/C for input queries
"""

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
from pathlib import Path
import os
from typing import Dict, Any
# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

# Label mapping
label_to_strategy = {
    'A': 'no_fetch',
    'B': 'dense_fetch', 
    'C': 'multi_fetch'
}
class ComplexityResult():
    label: str
    confidence: float

class QueryComplexityPredictor():
    def __init__(self, 
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        max_length: int = 384,
        padding: bool = False,
        return_tensors: str = "pt"
    ):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, "models", "saved_models", "adaptive_rag_classifier")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
        
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, query: str) -> ComplexityResult:
        """Predict complexity class for a query"""
        inputs = self.tokenizer(
            query.strip(),
            truncation=True,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors=self.return_tensors
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                max_length=30,
                num_beams=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Get probabilities for A, B, C tokens (same as training)
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(
                torch.stack([
                    scores[:, self.tokenizer('A').input_ids[0]],
                    scores[:, self.tokenizer('B').input_ids[0]], 
                    scores[:, self.tokenizer('C').input_ids[0]],
                ]), dim=0,
            ).detach().cpu().numpy()
            
            # Get prediction
            pred_label = np.argmax(probs, 0)[0]
            pred_class = ['A', 'B', 'C'][pred_label]
            confidence = float(np.max(probs, 0)[0])
            label = label_to_strategy[pred_class]
            return ComplexityResult(label, confidence)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_length": self.max_length,
            "model_loaded": self.model is not None,
            "purpose": "Query complexity prediction (T5)"
        }

query_complexity_predictor = QueryComplexityPredictor()

async def predict_query_complexity(query: str) -> ComplexityResult:
    """Convenience function to predict query complexity"""
    return await query_complexity_predictor.predict(query)
