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
        hallucination_threshold: float = 0.5,
        specialization_threshold: float = 0.5,
    ):
        self.adaptive_wrapper = None
        self.thresholds = {
            'hallucination': hallucination_threshold,
            'specialization': specialization_threshold,
        }
        
        # Initialize the adaptive wrapper
        self._initialize_wrapper()
    
    def _initialize_wrapper(self):
        """Initialize the adaptive wrapper with trained models"""
        try:
            print("🔄 Loading adaptive wrapper with trained models...")
            
            self.adaptive_wrapper = AdaptiveWrapper(
                hallucination_threshold=self.thresholds['hallucination'],
                specialization_threshold=self.thresholds['specialization'],
            )
            
            print("✅ Adaptive wrapper loaded successfully!")
            
            # Print model information
            config = self.adaptive_wrapper.get_configuration()
            hall_info = config['hallucination_predictor']
            spec_info = config['specialization_predictor']
            
            print(f"📊 Hallucination Model: {hall_info.get('model_path', 'Pattern-based fallback')}")
            print(f"🎯 Specialization Model: {spec_info.get('model_path', 'Pattern-based fallback')}")
            print(f"⚙️  Thresholds: Hall={self.thresholds['hallucination']}, Spec={self.thresholds['specialization']}")
            
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
            
            return {
                "query": query,
                "hallucination_score": decision.hallucination_risk.score,
                "hallucination_confidence": decision.hallucination_risk.confidence,
                "specialization_score": decision.specialization_need.score,
                "specialization_confidence": decision.specialization_need.confidence,
                "strategy": decision.strategy.value,
                "strategy_name": decision.strategy.value.replace("_", " ").title(),
                "use_rag": decision.use_rag,
                "use_complex_reasoning": decision.use_complex_reasoning,
                "reasoning": decision.reasoning,
                "processing_time_ms": decision.processing_time_ms
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            raise
    
    def update_thresholds(
        self, 
        hallucination: Optional[float] = None,
        specialization: Optional[float] = None
    ):
        """Update decision thresholds"""
        if hallucination is not None:
            self.thresholds['hallucination'] = hallucination
        if specialization is not None:
            self.thresholds['specialization'] = specialization
            
        self.adaptive_wrapper.set_thresholds(
            hallucination_threshold=self.thresholds['hallucination'],
            specialization_threshold=self.thresholds['specialization']
        )
        
        print(f"⚙️  Updated thresholds: Hall={self.thresholds['hallucination']}, Spec={self.thresholds['specialization']}")


def print_prediction_result(result: Dict[str, Any]):
    """Print detailed prediction results"""
    print("\n" + "="*80)
    print(f"📝 Query: {result['query']}")
    print("="*80)
    
    # Individual Agent Scores
    print("🤖 INDIVIDUAL AGENT SCORES")
    print("-" * 40)
    print(f"🔍 Hallucination Risk:")
    print(f"   Score: {result['hallucination_score']:.3f}")
    print(f"   Confidence: {result['hallucination_confidence']:.3f}")
    
    print(f"\n🎯 Specialization Need:")
    print(f"   Score: {result['specialization_score']:.3f}")
    print(f"   Confidence: {result['specialization_confidence']:.3f}")
    
    # Final Decision
    print(f"\n🎯 ROUTING DECISION")
    print("-" * 40)
    print(f"Strategy: {result['strategy_name']}")
    print(f"Use RAG: {'✅ Yes' if result['use_rag'] else '❌ No'}")
    print(f"Complex Reasoning: {'✅ Yes' if result['use_complex_reasoning'] else '❌ No'}")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATION")
    print("-" * 40)
    if result['strategy'] == 'direct_llm':
        print("🟢 Use direct LLM response - low risk of hallucination and general knowledge")
    elif result['strategy'] == 'single_step_rag':
        print("🟡 Use single-step RAG - retrieve relevant context and generate response")
    elif result['strategy'] == 'multi_step_rag':
        print("🔴 Use multi-step RAG - complex query requiring detailed analysis and multiple retrievals")
    
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
    print("  - 'thresholds' to view/modify decision thresholds")
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
                    print("  - 'thresholds' to view/modify decision thresholds")
                    print("  - 'config' to view current configuration")
                    print("  - 'quit', 'exit' to stop")
                    continue
                elif user_input.lower() == 'config':
                    config = predictor.adaptive_wrapper.get_configuration()
                    print("\n⚙️  Current Configuration:")
                    print(f"Hallucination threshold: {config['thresholds']['hallucination']}")
                    print(f"Specialization threshold: {config['thresholds']['specialization']}")
                    continue
                elif user_input.lower() == 'thresholds':
                    print(f"\nCurrent thresholds:")
                    print(f"Hallucination: {predictor.thresholds['hallucination']}")
                    print(f"Specialization: {predictor.thresholds['specialization']}")
                    
                    modify = input("\nModify thresholds? (y/n): ").strip().lower()
                    if modify == 'y':
                        try:
                            hall = input(f"Hallucination threshold ({predictor.thresholds['hallucination']}): ").strip()
                            spec = input(f"Specialization threshold ({predictor.thresholds['specialization']}): ").strip()
                            
                            hall_val = float(hall) if hall else None
                            spec_val = float(spec) if spec else None
                            
                            predictor.update_thresholds(hall_val, spec_val)
                        except ValueError:
                            print("❌ Invalid threshold values. Please enter numbers between 0 and 1.")
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
    
    parser.add_argument(
        "--hallucination-threshold",
        type=float,
        default=0.5,
        help="Threshold for hallucination risk (default: 0.5)"
    )
    
    parser.add_argument(
        "--specialization-threshold",
        type=float,
        default=0.5,
        help="Threshold for specialization need (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.query:
        parser.error("Please provide a query or use --interactive mode")
    
    # Validate thresholds
    for threshold_name, threshold_value in [
        ("hallucination", args.hallucination_threshold),
        ("specialization", args.specialization_threshold),
    ]:
        if not 0.0 <= threshold_value <= 1.0:
            parser.error(f"{threshold_name} threshold must be between 0.0 and 1.0")
    
    # Initialize predictor
    print("🔄 Initializing adaptive predictor...")
    try:
        predictor = AdaptivePredictorInterface(
            hallucination_threshold=args.hallucination_threshold,
            specialization_threshold=args.specialization_threshold
        )
        
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