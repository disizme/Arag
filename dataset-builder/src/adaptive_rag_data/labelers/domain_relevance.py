
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from ..retrieval.bm25 import BM25Index
import transformers
transformers.logging.set_verbosity_error()

@dataclass
class DomainSignals:
    lexicon_ratio: float
    embedding_sim: float
    bm25_ratio: float
    cross_encoder_score: float

class DomainLabelers:
    def __init__(self, embedding_model: str, cross_encoder_model: str, lexicon: List[str]):
        # Use MPS device if available, fallback to CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.cross_encoder = CrossEncoder(cross_encoder_model, device=device)
        self.lexicon_set = set([w.lower() for w in lexicon])
        self.ai_centroid: np.ndarray | None = None

    def build_ai_centroid(self, sample_texts: List[str]) -> None:
        # Reduce sample size for faster initialization
        sample_size = min(1024, len(sample_texts))
        vecs = self.embedder.encode(
            sample_texts[:sample_size], 
            convert_to_numpy=True, 
            show_progress_bar=False, 
            normalize_embeddings=True,
            batch_size=64  # Optimize batch size
        )
        self.ai_centroid = vecs.mean(axis=0)
        self.ai_centroid = self.ai_centroid / (np.linalg.norm(self.ai_centroid) + 1e-12)

    def _lexicon_ratio(self, query: str) -> float:
        query_lower = query.lower()
        hits = 0
        
        # Check each lexicon term (handles both single and multi-word terms)
        for term in self.lexicon_set:
            if term.lower() in query_lower:
                hits += 1
        
        # Normalize by total lexicon terms instead of query words
        return hits / max(1, len(self.lexicon_set))

    def _embedding_sim(self, query: str) -> float:
        if self.ai_centroid is None:
            return 0.0
        q = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)[0]
        return float(np.dot(q, self.ai_centroid))

    def _bm25_ratio(self, query: str, ai_idx: BM25Index, top_k: int) -> float:
        ai_top = ai_idx.query(query, top_k=top_k)
        ai_score = ai_top[0][1] if ai_top else 0.0
        return ai_score

    def _cross_encoder(self, query: str, ai_idx: BM25Index, top_k: int) -> float:
        # Use cross-encoder to compare question against AI/ML domain templates
        ai_templates = [
            "artificial intelligence and machine learning concepts",
            "data science and statistical methods", 
            "neural networks and deep learning",
            "data analysis and predictive modeling",
            "pattern recognition and deep learning algorithms",
            "reinforcement learning and decision making",
            "computer vision and image processing",
            "natural language processing and text analysis",
        ]
        
        pairs = [(query, template) for template in ai_templates]
        if not pairs:
            return 0.0
        scores = self.cross_encoder.predict(pairs)
        if len(scores) == 0:
            return 0.0
        max_score = float(np.max(scores))
        return max_score

    def compute_signals(self, query: str, ai_idx: BM25Index, top_k: int) -> DomainSignals:
        return DomainSignals(
            lexicon_ratio=self._lexicon_ratio(query),
            embedding_sim=self._embedding_sim(query),
            bm25_ratio=self._bm25_ratio(query, ai_idx, top_k),
            cross_encoder_score=self._cross_encoder(query, ai_idx, top_k),
        )

    def weak_votes(self, signals: DomainSignals, thresholds: Dict[str, float]) -> Dict[str, float]:
        """Convert to continuous scores based on distance from threshold using sigmoid."""
        import math
        
        def sigmoid_score(signal: float, threshold: float, scaling: float = 5.0) -> float:
            """Convert signal to continuous score based on distance from threshold."""
            return 1.0 / (1.0 + math.exp(-(signal - threshold) * scaling))
        
        votes: Dict[str, float] = {}
        votes["lexicon"] = sigmoid_score(signals.lexicon_ratio, thresholds.get("lexicon_high", 0.01), scaling=400)
        votes["embed"] = sigmoid_score(signals.embedding_sim, thresholds.get("embedding_sim_high", 0.25), scaling=4)
        votes["bm25_ratio"] = sigmoid_score(signals.bm25_ratio, thresholds.get("bm25_ratio_high", 40), scaling=0.04)  # Lower scaling for wide range
        votes["cross_encoder"] = sigmoid_score(signals.cross_encoder_score, thresholds.get("cross_encoder_high", 0.3), scaling=5)
        return votes