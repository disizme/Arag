
from __future__ import annotations
from typing import Dict, List
import numpy as np

ABSTAIN = -1

def build_label_matrix(vote_dicts: List[Dict[str, float]], labeler_names: List[str]) -> np.ndarray:
    # Use float dtype to support continuous values
    L = np.full((len(vote_dicts), len(labeler_names)), float(ABSTAIN), dtype=float)
    for i, votes in enumerate(vote_dicts):
        for j, name in enumerate(labeler_names):
            if name in votes:
                L[i, j] = float(votes[name])
    return L

def aggregate_probabilities(L: np.ndarray, seed: int = 42, labeler_names: List[str] = None) -> np.ndarray:
    
    # Define weights for domain relevance signals (cross_encoder > bm25 > embedding > lexicon)
    domain_weights = {
        #Domain Relevance
        "cross_encoder": 0.4,  # Highest weight
        "bm25_ratio": 0.3,
        "embed": 0.2, 
        "lexicon": 0.1,         # Lowest weight
        #Hallucination
        "factual_precision": 0.3,
        "obscurity": 0.25,
        "complexity": 0.45,
        #"answer_dev": 0.1
    }
    
    # Default equal weights for hallucination or unknown labelers
    default_weight = 1.0 / L.shape[1] if L.shape[1] > 0 else 1.0
    
    probs = []
    for row in L:
        vals = [v for v in row.tolist() if v != ABSTAIN]
        if not vals:
            probs.append(0.5)
        else:
            if labeler_names and len(labeler_names) == len(row):
                # Use weighted average for domain signals
                weighted_sum = 0.0
                confidence_sum = 0.0
                for i, val in enumerate(row):
                    if val != ABSTAIN:
                        weight = domain_weights.get(labeler_names[i], default_weight)
                        #confidence = abs(val-0.5) * 2
                        #confidence_boost = confidence ** 3 * 4
                        #boosted_weight = weight * (1 + confidence_boost)  # boost for hallucination data
                        boosted_weight = weight
                        confidence_sum += boosted_weight
                        weighted_sum += val * boosted_weight
                p = float(weighted_sum / confidence_sum) if confidence_sum > 0 else 0.5
            else:
                # Simple average for hallucination or when no labeler names provided
                p = float(sum(vals) / len(vals))
            probs.append(p)
    probs = np.clip(np.array(probs), 0.0, 1.0)
    return probs
