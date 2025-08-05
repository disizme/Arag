#!/usr/bin/env python3
"""
Hallucination Predictor Inference Script

Simple script to load trained hallucination predictor model and make predictions.

Usage:
    python predict_hallucination.py "What is machine learning?"
    python predict_hallucination.py --interactive
"""

import os
import sys
import argparse
import asyncio
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# Disable warnings and configure environment
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HallucinationPredictor:
    """
    Hallucination predictor for inference using trained models.
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_path = Path(model_path)
        self.max_length = max_length
        
        # Auto-detect device
        self.device = self._get_optimal_device(device)
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        
        # Load model and tokenizer
        self._load_model()
        
        print(f"✅ Hallucination predictor loaded from: {model_path}")
    
    def _get_optimal_device(self, preferred_device: Optional[str] = None) -> str:
        """Auto-detect the best available device"""
        if preferred_device:
            return preferred_device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            print(f"🔄 Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                use_fast=True
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            print(f"✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text for model processing"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    async def predict(self, query: str) -> Dict[str, Any]:
        """
        Predict hallucination risk for a query.
        
        Args:
            query: The input query to analyze
            
        Returns:
            Dictionary with prediction results
        """
        # Tokenize input
        inputs = self._tokenize_text(query)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # For regression model, logits is the direct score prediction
            risk_score = float(logits.item())
            
            # Clip to valid range [0, 1]
            risk_score = max(0.0, min(1.0, risk_score))
        
        return {
            "query": query,
            "risk_score": risk_score,
            "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
        }
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_prediction_result(result: Dict[str, Any]):
    """Print prediction results"""
    print("\n" + "="*60)
    print(f"Query: {result['query']}")
    print("="*60)
    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Risk Level: {result['risk_level']}")
    
    if result['risk_level'] == "HIGH":
        print("🔴 Recommendation: Use RAG retrieval for factual accuracy")
    elif result['risk_level'] == "MEDIUM":
        print("🟡 Recommendation: RAG retrieval may be beneficial")
    else:
        print("🟢 Recommendation: Direct LLM response likely acceptable")


async def predict_single_query(predictor: HallucinationPredictor, query: str):
    """Make a prediction for a single query"""
    try:
        result = await predictor.predict(query)
        print_prediction_result(result)
    except Exception as e:
        print(f"❌ Error making prediction: {e}")


async def interactive_mode(predictor: HallucinationPredictor):
    """Run in interactive mode for multiple queries"""
    print("🤖 Hallucination Predictor - Interactive Mode")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("="*60)
    
    try:
        while True:
            try:
                query = input("\nEnter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    print("Please enter a query.")
                    continue
                
                result = await predictor.predict(query)
                print_prediction_result(result)
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Goodbye!")


def find_saved_model() -> Optional[Path]:
    """Find the saved hallucination predictor model"""
    models_dir = Path(__file__).parent / "models" / "saved_models"
    
    for model_dir in models_dir.glob("hallucination_predictor_*"):
        if model_dir.is_dir():
            if (model_dir / "config.json").exists() and (model_dir / "model.safetensors").exists():
                return model_dir
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Predict hallucination risk using trained model"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to analyze (use quotes for multi-word queries)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for multiple queries"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Device to use for inference (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Please provide a query or use --interactive mode")
    
    # Find model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = find_saved_model()
        if model_path is None:
            print("❌ No saved model found. Please train a model first or specify --model-path")
            return 1
        print(f"📁 Found saved model: {model_path}")
    
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return 1
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    # Initialize predictor
    print("🔄 Loading hallucination predictor...")
    try:
        predictor = HallucinationPredictor(
            model_path=str(model_path),
            device=device
        )
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Run prediction
    try:
        if args.interactive:
            asyncio.run(interactive_mode(predictor))
        else:
            asyncio.run(predict_single_query(predictor, args.query))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        # Clean up resources
        try:
            predictor.cleanup()
        except:
            pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())