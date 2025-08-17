
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from transformers import pipeline
from sentence_transformers import CrossEncoder, SentenceTransformer
import math
import torch
import transformers
transformers.logging.set_verbosity_error()

@dataclass
class HallucinationSignals:
    factual_precision_risk: float
    obscurity_risk: float
    complexity_risk: float
    #answer_deviation: float

class HallucinationLabelers:
    def __init__(self, cross_encoder: str):
        # Use MPS device if available, fallback to CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self._cross_encoder = CrossEncoder(cross_encoder, device=device)
        #self._text_generator = pipeline(
        #    "text2text-generation", 
        #    model=text_gen_model, 
        #    tokenizer=text_gen_model,
        #    max_new_tokens=256,
        #    device=device
        #)


    def _batch_cross_encode_risks(self, question: str) -> Dict[str, float]:
        """Use cross-encoder to evaluate all risk hypotheses in a single batch call."""
        try:
            hypotheses = {
                "factual_precision": "This question demands exact facts, specific numbers, precise dates, or detailed information that could easily be fabricated if unknown.",
                "complexity": "This question involves multi-step reasoning, complex analysis, or connecting multiple concepts together.",
                "obscurity_high": "This question covers specialized, technical, or niche topics that most people wouldn't know.",
                "obscurity_low": "This question covers common knowledge, basic facts, or widely known information."
            }
            
            # Create pairs for batch processing
            pairs = [(question, hypothesis) for hypothesis in hypotheses.values()]
            
            # Get scores in single batch call
            scores = self._cross_encoder.predict(pairs)
            
            # Map scores back to risk types
            risk_scores = {}
            for i, risk_type in enumerate(hypotheses.keys()):
                score = float(scores[i])
                
                # STSB cross-encoder outputs similarity scores (typically 0-1 range already)
                # Clamp to ensure 0-1 range
                normalized_score = max(0.0, min(1.0, score))
                risk_scores[risk_type] = normalized_score
            
            # Calculate obscurity risk as ratio: obscure_similarity / (obscure + common)
            obscure_sim = risk_scores["obscurity_high"]
            common_sim = risk_scores["obscurity_low"]
            if obscure_sim + common_sim > 0:
                obscurity_risk = obscure_sim / (obscure_sim + common_sim)
            else:
                obscurity_risk = 0.4
            
            return {
                "factual_precision": risk_scores["factual_precision"],
                "complexity": risk_scores["complexity"],
                "obscurity": obscurity_risk
            }
            
        except Exception:
            # Return default scores if batch processing fails
            return {"factual_precision": 0.4, "complexity": 0.4, "obscurity": 0.4}



    #def _generated_vs_actual_deviation(self, question: str, actual_answer: str) -> float:
    #    """Generate answer using text generation model and compare deviation from actual answer."""
    #    try:
    #        # Generate answer using parametric knowledge only
    #        generated_results = self._text_generator(question[:200], max_new_tokens=256, num_return_sequences=1, temperature=0.7)
    #        if not generated_results or not generated_results[0].get("generated_text"):
    #            return 0.5  # Default if generation fails
                
    #        generated_answer = generated_results[0]["generated_text"].strip()
            # Compare generated vs actual using semantic similarity
    #        embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
    #        gen_emb = embedder.encode([generated_answer])
    #        actual_emb = embedder.encode([actual_answer])
            
    #        similarity = float(gen_emb @ actual_emb.T)
    #        deviation = 1.0 - max(0.0, similarity)  # Higher deviation = more risk
            
    #        return min(1.0, max(0.0, deviation))
            
        except Exception as e:
            return 0.4  # Default if anything fails

    def compute_signals(self, question: str, answer: str, contexts: List[str] = None) -> HallucinationSignals:
        # Get all cross-encoder risk scores in single batch call
        risk_scores = self._batch_cross_encode_risks(question)
        
        # Get answer deviation signal
        #answer_deviation = self._generated_vs_actual_deviation(question, answer)
        return HallucinationSignals(
            factual_precision_risk=risk_scores["factual_precision"],  # Combined specificity + precision risk
            obscurity_risk=risk_scores["obscurity"],                   # Is topic obscure/niche
            complexity_risk=risk_scores["complexity"],                 # Does query require complex reasoning
            #answer_deviation=answer_deviation,                         # Generated vs actual answer deviation
        )

    def weak_votes(self, signals: HallucinationSignals, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Convert to continuous scores based on distance from threshold using sigmoid."""
        
        def sigmoid_score(signal: float, threshold: float, scaling: float = 5.0) -> float:
            """Convert signal to continuous score based on distance from threshold."""
            return 1.0 / (1.0 + math.exp(-(signal - threshold) * scaling))
        
        votes: Dict[str, float] = {}
        votes["factual_precision"] = sigmoid_score(signals.factual_precision_risk, thresholds.get("factual_precision_risk_high_risk", 0.4), scaling=5)
        votes["obscurity"] = sigmoid_score(signals.obscurity_risk, thresholds.get("obscurity_risk_high_risk", 0.4), scaling=5)
        votes["complexity"] = sigmoid_score(signals.complexity_risk, thresholds.get("complexity_risk_high_risk", 0.4), scaling=5)
        #votes["answer_dev"] = sigmoid_score(signals.answer_deviation, thresholds.get("answer_deviation_high_risk", 0.5), scaling=5)
        return votes
