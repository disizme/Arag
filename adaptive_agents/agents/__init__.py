"""
Adaptive RAG Agents

This package contains the core agents for the Adaptive RAG system:
- HallucinationPredictor: Predicts if LLM might hallucinate
- SpecializationPredictor: Predicts if query needs specialized knowledge
- AdaptiveWrapper: Main coordinator implementing dual question framework
"""

from .hallucination_predictor import (
    HallucinationPredictor,
    predict_hallucination_risk,
    hallucination_predictor
)

from .specialization_predictor import (
    SpecializationPredictor,
    predict_specialization_need,
    specialization_predictor
)

from .query_complexity_predictor import (
    QueryComplexityPredictor,
    predict_query_complexity,
    query_complexity_predictor
)

from .adaptive_wrapper import (
    AdaptiveWrapper,
    RoutingDecision,
    QueryStrategy,
    adaptive_wrapper
)

__all__ = [
    # Main classes
    "HallucinationPredictor",
    "SpecializationPredictor", 
    "QueryComplexityPredictor",
    "AdaptiveWrapper",
    
    # Result classes
    "RoutingDecision",
    
    # Enums
    "QueryStrategy",
    
    # Convenience functions
    "predict_hallucination_risk",
    "predict_specialization_need",
    "predict_query_complexity",
    
    # Global instances
    "hallucination_predictor",
    "specialization_predictor",
    "query_complexity_predictor",
    "adaptive_wrapper"
]

# Version info
__version__ = "1.0.0"