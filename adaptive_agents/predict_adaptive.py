#!/usr/bin/env python3
"""
Adaptive Wrapper Predictor Inference Script

Interactive script to use the adaptive wrapper that loads both trained models
(hallucination and specialization predictors) and provides individual scores
plus final routing decision.

Usage:
    python predict_adaptive.py "What is machine learning?"
    python predict_adaptive.py --interactive
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
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from agents.adaptive_wrapper import AdaptiveWrapper, RoutingDecision

class AdaptivePredictorInterface:
    """
    Interactive interface for the adaptive wrapper with both trained models.
    """
    
    def __init__(
        self,
        hallucination_high_threshold: float = 0.6,
        hallucination_low_threshold: float = 0.3,
        specialization_high_threshold: float = 0.7,
        specialization_low_threshold: float = 0.3,
    ):
        self.adaptive_wrapper = None
        self.thresholds = {
            'hallucination_high': hallucination_high_threshold,
            'hallucination_low': hallucination_low_threshold,
            'specialization_high': specialization_high_threshold,
            'specialization_low': specialization_low_threshold,
        }
        
        # Initialize the adaptive wrapper
        self._initialize_wrapper()
    
    def _initialize_wrapper(self):
        """Initialize the adaptive wrapper with trained models"""
        try:
            print("🔄 Loading adaptive wrapper with trained models...")
            
            self.adaptive_wrapper = AdaptiveWrapper(
                hallucination_high_threshold=self.thresholds['hallucination_high'],
                hallucination_low_threshold=self.thresholds['hallucination_low'],
                specialization_high_threshold=self.thresholds['specialization_high'],
                specialization_low_threshold=self.thresholds['specialization_low'],
            )
            
            print("✅ Adaptive wrapper loaded successfully!")
            
            # Print model information
            config = self.adaptive_wrapper.get_configuration()
            hall_info = config['hallucination_predictor']
            spec_info = config['specialization_predictor']
            classifier_info = config['query_complexity_predictor']
            print(f"📊 Hallucination Model: {hall_info['model_path']}")
            print(f"🎯 Specialization Model: {spec_info['model_path']}")
            print(f"🎯 Classifier Model: {classifier_info['model_path']}")
            print(f"⚙️  Thresholds: Hall={self.thresholds['hallucination_high']}-{self.thresholds['hallucination_low']}, Spec={self.thresholds['specialization_high']}-{self.thresholds['specialization_low']}")
            
        except Exception as e:
            print(f"❌ Error loading adaptive wrapper: {e}")
            raise
    
    async def predict(self, query: str) -> Dict[str, Any]:
        """
        Analyze query and get routing decision with individual scores.
        
        Args:
            query: The input query to analyze
            
        Returns:
            Dictionary with detailed analysis results
        """
        try:
            decision = await self.adaptive_wrapper.analyze_query(query)
            classifier_decision = await self.adaptive_wrapper.predict_query_complexity(query)
            return {
                "query": query,
                "hallucination_score": decision.hallucination_risk.score,
                "specialization_score": decision.specialization_need.score,
                "strategy": decision.strategy.value,
                "strategy_name": decision.strategy.value.replace("_", " ").title(),
                "processing_time_ms": decision.processing_time_ms,
                "classifier_proccessing_time_ms": classifier_decision.processing_time_ms,
                "classifier_label": classifier_decision.complexity.label,
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise e

def print_prediction_result(result: Dict[str, Any]):
    """Print detailed prediction results"""
    print("\n" + "="*80)
    print(f"📝 Query: {result['query']}")
    print("="*80)

    print(f"\n🎯 Classifier:")
    print(f"   Label: {result['classifier_label']}")
    print(f"Classifier Processing Time: {result['classifier_proccessing_time_ms']:.1f}ms")

    # Individual Agent Scores
    print("🤖 INDIVIDUAL AGENT SCORES")
    print("-" * 40)
    print(f"🔍 Hallucination Risk:")
    print(f"   Score: {result['hallucination_score']:.3f}")
    
    print(f"\n🎯 Specialization Need:")
    print(f"   Score: {result['specialization_score']:.3f}")

    # Final Decision
    print(f"\n🎯 ROUTING DECISION")
    print("-" * 40)
    print(f"Strategy: {result['strategy_name']}")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")

    print("="*80)


async def predict_single_query(predictor: AdaptivePredictorInterface, query: str):
    """Make a prediction for a single query"""
    try:
        result = await predictor.predict(query)
        print_prediction_result(result)
    except Exception as e:
        print(f"❌ Error making prediction: {e}")


async def interactive_mode(predictor: AdaptivePredictorInterface):
    """Run in interactive mode for multiple queries"""
    print("\n🤖 Adaptive RAG Predictor - Interactive Mode")
    print("Available commands:")
    print("  - Enter a query to analyze")
    print("  - 'config' to view current configuration")
    print("  - 'help' to show this help")
    print("  - 'quit', 'exit', or Ctrl+C to stop")
    print("="*80)
    
    try:
        while True:
            try:
                user_input = input("\n🔍 Enter query or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  - Enter a query to analyze")
                    print("  - 'config' to view current configuration")
                    print("  - 'quit', 'exit' to stop")
                    continue
                elif user_input.lower() == 'config':
                    config = predictor.adaptive_wrapper.get_configuration()
                    print("\n⚙️  Current Configuration:")
                    print(f"Hallucination threshold: {config['thresholds']['hallucination']}")
                    print(f"Specialization threshold: {config['thresholds']['specialization']}")
                    continue

                if not user_input:
                    print("Please enter a query or command.")
                    continue
                
                result = await predictor.predict(user_input)
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



def main():
    parser = argparse.ArgumentParser(
        description="Analyze queries using adaptive wrapper with both trained models"
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Please provide a query or use --interactive mode")
    
    # Initialize predictor
    print("🔄 Initializing adaptive predictor...")
    try:
        predictor = AdaptivePredictorInterface()
        
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return 1
    
    # Run prediction
    try:
        if args.interactive:
            asyncio.run(interactive_mode(predictor))
        else:
            asyncio.run(predict_single_query(predictor, args.query))
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())