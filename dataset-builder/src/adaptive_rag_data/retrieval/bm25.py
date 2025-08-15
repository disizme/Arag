
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import json

@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    title: str | None = None

class BM25Index:
    def __init__(self, docs: List[CorpusDoc], use_title: bool = True):
        self.docs = docs
        self.use_title = use_title
        tokenized_corpus = [self._tokenize(self._doc_text(d)) for d in docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _doc_text(self, doc: CorpusDoc) -> str:
        if self.use_title and doc.title:
            return f"{doc.title} {doc.text}"
        return doc.text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # More efficient tokenization with minimal processing
        return text.lower().split()

    def query(self, query: str, top_k: int = 5) -> List[Tuple[CorpusDoc, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        idx_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.docs[i], float(s)) for i, s in idx_scores]

def load_corpus(path: str) -> List[CorpusDoc]:
    docs: List[CorpusDoc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(CorpusDoc(doc_id=str(obj.get("id")), text=obj.get("text", ""), title=obj.get("title")))
    return docs
