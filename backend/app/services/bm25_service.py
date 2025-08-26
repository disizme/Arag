from typing import List, Dict
from rank_bm25 import BM25Okapi
import re


class BM25Service:
    """Simple BM25 sparse embedding service for Qdrant hybrid search."""
    
    def __init__(self):
        self.bm25_model = None
        self.vocabulary = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 1]
    
    def fit(self, texts: List[str]) -> None:
        """Fit BM25 model on corpus of texts."""
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
        # Build vocabulary
        all_terms = set()
        for tokens in tokenized_corpus:
            all_terms.update(tokens)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
    
    def get_sparse_vector(self, text: str) -> Dict[int, float]:
        """Convert text to sparse vector using BM25 term weights."""
        if not self.bm25_model:
            raise ValueError("Model not fitted. Call fit() first.")
        
        tokens = self._tokenize(text)
        scores = self.bm25_model.get_scores(tokens)
        
        sparse_vector = {}
        for token in set(tokens):
            if token in self.vocabulary:
                term_id = self.vocabulary[token]
                # Use the token's IDF weight as sparse feature
                if hasattr(self.bm25_model, 'idf') and token in self.bm25_model.idf:
                    sparse_vector[term_id] = self.bm25_model.idf[token]
        
        return sparse_vector


bm25_service = BM25Service()