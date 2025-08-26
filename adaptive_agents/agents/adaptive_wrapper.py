"""
Adaptive Wrapper Agent

Main coordinator that combines hallucination prediction and specialization prediction
to make intelligent routing decisions for RAG systems.

Implements the Dual Question Framework with specialization-first logic:
1. Question 1: Will the core LLM likely hallucinate on this query?
2. Question 2: Does this query need specialized domain knowledge?

High specialization threshold: 0.7
High hallucination threshold: 0.7
Low specialization threshold: 0.3
Low hallucination threshold: 0.4

Routes queries to:
- No Fetch (Direct LLM)
    - specialization score below low specialization threshold
- Shallow Fetch (sparse retrieval)
    - specialization score between two specialization thresholds and hallucination score below low hallucination threshold
- Dense Fetch (dense retrieval)
    - specialization score between two specialization thresholds and hallucination score between two hallucination thresholds
- Hybrid Fetch (hybrid retrieval) 
    - specialization score between two specialization thresholds and hallucination score above high hallucination thresholds
    - OR specialization score above high specialization threshold and hallucination score below high hallucination thresholds
- Multi-fetch (complex reasoning with multiple retrievals)
    - specialization score above high specialization threshold and hallucination score above high hallucination thresholds

"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

from .hallucination_predictor import HallucinationPredictor
from .specialization_predictor import SpecializationPredictor
from .query_complexity_predictor import ComplexityResult, QueryComplexityPredictor
from ..models.base_predictor import PredictionResult

logger = logging.getLogger(__name__)

class QueryStrategy(str, Enum):
    """Strategy for handling the query"""
    NO_FETCH = "no_fetch"               # Direct LLM without context
    SHALLOW_FETCH = "shallow_fetch"     # Sparse retrieval (BM25)
    DENSE_FETCH = "dense_fetch"         # Dense retrieval (embeddings)
    HYBRID_FETCH = "hybrid_fetch"       # Hybrid retrieval (sparse + dense)
    MULTI_FETCH = "multi_fetch"         # Complex reasoning with multiple retrievals

@dataclass
class RoutingDecision:
    """Final routing decision from the adaptive wrapper"""
    strategy: QueryStrategy
    hallucination_risk: PredictionResult
    specialization_need: PredictionResult
    reasoning: str
    processing_time_ms: float
    
    # Convenience properties for integration
    @property
    def use_rag(self) -> bool:
        """Whether to use RAG (any form) vs direct LLM"""
        return self.strategy != QueryStrategy.NO_FETCH
    
    @property
    def use_complex_reasoning(self) -> bool:
        """Whether to use multi-step reasoning"""
        return self.strategy == QueryStrategy.MULTI_FETCH

class AdaptiveWrapper:
    """
    Main adaptive agent that coordinates hallucination and specialization predictors.
    
    Uses specialization-first routing logic where high domain expertise need
    always routes to RAG, regardless of hallucination risk.
    """
    
    def __init__(
        self,
        hallucination_high_threshold: float = 0.7,
        hallucination_low_threshold: float = 0.4,
        specialization_high_threshold: float = 0.7,
        specialization_low_threshold: float = 0.3,
    ):
        """
        Initialize the adaptive wrapper.
        
        Args:
            hallucination_high_threshold: High threshold for hallucination (default 0.7)
            hallucination_low_threshold: Low threshold for hallucination (default 0.4)
            specialization_high_threshold: High threshold for specialization (default 0.7)
            specialization_low_threshold: Low threshold for specialization (default 0.3)
        """
        
        # Initialize predictors with saved models
        self.hallucination_predictor =  HallucinationPredictor()
        self.specialization_predictor =  SpecializationPredictor()
        self.query_complexity_predictor =  QueryComplexityPredictor()
        
        # Decision thresholds
        self.hallucination_high_threshold = hallucination_high_threshold
        self.hallucination_low_threshold = hallucination_low_threshold
        self.specialization_high_threshold = specialization_high_threshold
        self.specialization_low_threshold = specialization_low_threshold
        
        logger.info("[ADAPTIVE-WRAPPER] Initialized with dual threshold routing")
        logger.info(f"[ADAPTIVE-WRAPPER] Hallucination thresholds - High: {hallucination_high_threshold}, Low: {hallucination_low_threshold}")
        logger.info(f"[ADAPTIVE-WRAPPER] Specialization thresholds - High: {specialization_high_threshold}, Low: {specialization_low_threshold}")
    
    async def analyze_query(self, query: str) -> RoutingDecision:
        """
        Analyze a query and determine the optimal routing strategy.
        
        Args:
            query: The input query to analyze
            
        Returns:
            RoutingDecision with strategy and detailed analysis
        """
        start_time = time.time()
        
        try:
            logger.info(f"[ADAPTIVE-WRAPPER] Analyzing query: {query[:50]}...")
            
            # Run both predictors concurrently for efficiency
            hallucination_task = self.hallucination_predictor.predict(query)
            specialization_task = self.specialization_predictor.predict(query)
            
            hallucination_risk, specialization_need = await asyncio.gather(
                hallucination_task,
                specialization_task
            )
            
            # Apply specialization-first routing logic
            strategy = self._determine_strategy(hallucination_risk, specialization_need)
            
            # Generate human-readable reasoning
            reasoning = self._generate_reasoning(strategy, hallucination_risk, specialization_need)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            decision = RoutingDecision(
                strategy=strategy,
                hallucination_risk=hallucination_risk,
                specialization_need=specialization_need,
                reasoning=reasoning,
                processing_time_ms=processing_time
            )
            
            logger.info(f"[ADAPTIVE-WRAPPER] Decision: {strategy.value} (hall: {hallucination_risk.score:.3f}, spec: {specialization_need.score:.3f}, time: {processing_time:.1f}ms)")
            
            return decision
            
        except Exception as e:
            logger.error(f"[ADAPTIVE-WRAPPER] Error in analysis: {e}")
            return self._fallback_decision(query, time.time() - start_time)
    
    def _determine_strategy(
        self, 
        hallucination_risk: PredictionResult, 
        specialization_need: PredictionResult
    ) -> QueryStrategy:
        """
        Routing logic based on dual thresholds for specialization and hallucination.
        
        Thresholds:
        - High specialization: 0.7, Low: 0.3
        - High hallucination: 0.7, Low: 0.4
        """
        
        hall_score = hallucination_risk.score
        spec_score = specialization_need.score
        
        # No Fetch (Direct LLM): specialization score below low threshold
        if spec_score < self.specialization_low_threshold:
            return QueryStrategy.NO_FETCH
        
        # Multi-fetch (complex reasoning): high specialization AND high hallucination
        elif spec_score > self.specialization_high_threshold and hall_score > self.hallucination_high_threshold:
            return QueryStrategy.MULTI_FETCH
        
        # Hybrid Fetch: (medium specialization AND high hallucination) OR (high specialization AND low-medium hallucination)
        elif ((self.specialization_low_threshold <= spec_score <= self.specialization_high_threshold and hall_score > self.hallucination_high_threshold) or 
              (spec_score > self.specialization_high_threshold and hall_score < self.hallucination_high_threshold)):
            return QueryStrategy.HYBRID_FETCH
        
        # Dense Fetch: medium specialization AND medium hallucination
        elif (self.specialization_low_threshold <= spec_score <= self.specialization_high_threshold and 
              self.hallucination_low_threshold <= hall_score <= self.hallucination_high_threshold):
            return QueryStrategy.DENSE_FETCH
        
        # Shallow Fetch: medium specialization AND low hallucination
        elif (self.specialization_low_threshold <= spec_score <= self.specialization_high_threshold and 
              hall_score < self.hallucination_low_threshold):
            return QueryStrategy.SHALLOW_FETCH
        
        # Default fallback
        else:
            return QueryStrategy.NO_FETCH
    
    def _generate_reasoning(
        self,
        strategy: QueryStrategy,
        hallucination_risk: PredictionResult,
        specialization_need: PredictionResult
    ) -> str:
        """Generate clear reasoning showing the routing decision based on dual thresholds"""
        
        hall_score = hallucination_risk.score
        spec_score = specialization_need.score
        
        if strategy == QueryStrategy.NO_FETCH:
            return f"No Fetch (Direct LLM): Low specialization need ({spec_score:.2f} < 0.3)"
        elif strategy == QueryStrategy.SHALLOW_FETCH:
            return f"Shallow Fetch: Medium specialization ({spec_score:.2f}) + low hallucination ({hall_score:.2f} < 0.4)"
        elif strategy == QueryStrategy.DENSE_FETCH:
            return f"Dense Fetch: Medium specialization ({spec_score:.2f}) + medium hallucination ({hall_score:.2f})"
        elif strategy == QueryStrategy.HYBRID_FETCH:
            return f"Hybrid Fetch: Complex scoring - spec={spec_score:.2f}, hall={hall_score:.2f}"
        elif strategy == QueryStrategy.MULTI_FETCH:
            return f"Multi-fetch: High specialization ({spec_score:.2f} > 0.7) + high hallucination ({hall_score:.2f} > 0.7)"
        else:
            return f"Unknown strategy: spec={spec_score:.2f}, hall={hall_score:.2f}"
    
    def _fallback_decision(self, query: str, processing_time: float) -> RoutingDecision:
        """Safe fallback decision when analysis fails"""
        
        # Create fallback predictions
        fallback_hallucination = PredictionResult(
            score=0.6,
            reasoning="Fallback prediction",
            model_type="fallback"
        )
        
        fallback_specialization = PredictionResult(
            score=0.6,
            reasoning="Fallback prediction",
            model_type="fallback"
        )
        
        return RoutingDecision(
            strategy=QueryStrategy.DENSE_FETCH,  # Safe middle ground
            hallucination_risk=fallback_hallucination,
            specialization_need=fallback_specialization,
            reasoning="Fallback to dense fetch due to analysis error",
            processing_time_ms=processing_time * 1000
        )
    
    def set_thresholds(
        self,
        hallucination_high_threshold: Optional[float] = None,
        hallucination_low_threshold: Optional[float] = None,
        specialization_high_threshold: Optional[float] = None,
        specialization_low_threshold: Optional[float] = None,
    ):
        """Update decision thresholds"""
        if hallucination_high_threshold is not None:
            self.hallucination_high_threshold = hallucination_high_threshold
            
        if hallucination_low_threshold is not None:
            self.hallucination_low_threshold = hallucination_low_threshold
            
        if specialization_high_threshold is not None:
            self.specialization_high_threshold = specialization_high_threshold
            
        if specialization_low_threshold is not None:
            self.specialization_low_threshold = specialization_low_threshold
                    
        logger.info(f"[ADAPTIVE-WRAPPER] Updated thresholds:")
        logger.info(f"  Hallucination - High: {self.hallucination_high_threshold}, Low: {self.hallucination_low_threshold}")
        logger.info(f"  Specialization - High: {self.specialization_high_threshold}, Low: {self.specialization_low_threshold}")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "thresholds": {
                "hallucination": {
                    "high": self.hallucination_high_threshold,
                    "low": self.hallucination_low_threshold
                },
                "specialization": {
                    "high": self.specialization_high_threshold,
                    "low": self.specialization_low_threshold
                },
            },
            "hallucination_predictor": self.hallucination_predictor.get_model_info(),
            "specialization_predictor": self.specialization_predictor.get_model_info(),
            "query_complexity_predictor": self.query_complexity_predictor.get_model_info()
        }
    
    async def predict_query_complexity(self, query: str) -> ComplexityResult:
        """
        Predict query complexity using the complexity predictor.
        
        Args:
            query: The input query to analyze
            
        Returns:
            PredictionResult with complexity score (0.0=A, 0.5=B, 1.0=C)
        """
        return await self.query_complexity_predictor.predict(query)

# Global instance for easy usage
adaptive_wrapper = AdaptiveWrapper()