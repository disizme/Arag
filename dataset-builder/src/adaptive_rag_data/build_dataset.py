
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from .config import Config
from .retrieval.bm25 import load_corpus, BM25Index
from .labelers.hallucination import HallucinationLabelers
from .labelers.domain_relevance import DomainLabelers
from .aggregation.label_model import build_label_matrix, aggregate_probabilities

@dataclass
class Example:
    ex_id: str
    question: str
    answer: str | None
    source: str

def _ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _load_questions(path: str) -> List[Example]:
    items: List[Example] = []
    
    # If path is a directory, load all JSON files from it
    if os.path.isdir(path):
        json_files = [f for f in os.listdir(path) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files in {path}")
        
        for json_file in json_files:
            file_path = os.path.join(path, json_file)
            source_name = json_file.replace('_extracted.json', '').replace('.json', '')
            initial_count = len(items)
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list format and single object format
                if isinstance(data, list):
                    for obj in data:
                        items.append(Example(
                            ex_id=str(obj.get("id")), 
                            question=obj.get("question"), 
                            answer=obj.get("answer"), 
                            source=str(obj.get("source"))
                        ))
                else:
                    items.append(Example(
                        ex_id=str(data.get("id")), 
                        question=data.get("question"), 
                        answer=data.get("answer"), 
                        source=str(data.get("source"))
                    ))
            
            questions_loaded = len(items) - initial_count
            print(f"Loaded {questions_loaded} questions from {json_file}")
    
    # If path is a file, load it as before (supporting both JSONL and JSON formats)
    elif os.path.isfile(path):
        if path.endswith('.jsonl'):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    items.append(Example(ex_id=str(obj.get("id")), question=obj.get("question"), answer=obj.get("answer"), source=str(obj.get("source"))))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        items.append(Example(ex_id=str(obj.get("id")), question=obj.get("question"), answer=obj.get("answer"), source=str(obj.get("source"))))
                else:
                    items.append(Example(ex_id=str(data.get("id")), question=data.get("question"), answer=data.get("answer"), source=str(data.get("source"))))
    
    return items

def build_dataset(cfg: Config) -> None:
    ai_corpus_path = cfg.get("paths", "ai_corpus")
    q_path = cfg.get("paths", "questions")
    out_dir = cfg.get("paths", "output_dir")
    art_dir = cfg.get("paths", "artifacts_dir")
    lex_path = cfg.get("paths", "lexicon_path")

    _ensure_dirs([out_dir, art_dir])

    ai_docs = load_corpus(ai_corpus_path)
    ai_idx = BM25Index(ai_docs, use_title=bool(cfg.get("retrieval", "bm25", "use_title", default=True)))

    top_k = int(cfg.get("retrieval", "top_k", default=5))

    with open(lex_path, "r", encoding="utf-8") as f:
        lexicon = [ln.strip() for ln in f if ln.strip()]

    hall = HallucinationLabelers(
        cross_encoder=str(cfg.get("models", "cross_encoder")),
        text_gen_model=str(cfg.get("models", "text_gen_model")),
    )
    dom = DomainLabelers(
        embedding_model=str(cfg.get("retrieval", "vector", "embedding_model")),
        cross_encoder_model=str(cfg.get("models", "cross_encoder")),
        lexicon=lexicon,
    )

    dom.build_ai_centroid([d.text for d in ai_docs[:5000]])

    examples = _load_questions(q_path)

    rows_domain: List[Dict[str, Any]] = []
    rows_hallu: List[Dict[str, Any]] = []

    for ex in tqdm(examples, desc="Building signals"):
        dom_signals = dom.compute_signals(ex.question, ai_idx, top_k)
        dom_votes = dom.weak_votes(dom_signals, thresholds=cfg.get("weak_labeling", "domain", "thresholds"))
        rows_domain.append({
            "id": ex.ex_id,
            "question": ex.question,
            "answer": ex.answer,
            "source": ex.source,
            "signals": {
                "lexicon_ratio": dom_signals.lexicon_ratio,
                "embedding_sim": dom_signals.embedding_sim,
                "bm25_ratio": dom_signals.bm25_ratio,
                "cross_encoder_score": dom_signals.cross_encoder_score,
            },
            "votes": dom_votes,
        })

        if ex.answer:
            hall_signals = hall.compute_signals(ex.question, ex.answer)
            hall_votes = hall.weak_votes(hall_signals, thresholds=cfg.get("weak_labeling", "hallucination", "thresholds"))
            rows_hallu.append({
                "id": ex.ex_id,
                "question": ex.question,
                "answer": ex.answer,
                "source": ex.source,
                "signals": {
                    "factual_precision_risk": hall_signals.factual_precision_risk,
                    "obscurity_risk": hall_signals.obscurity_risk,
                    "complexity_risk": hall_signals.complexity_risk,
                    "answer_deviation": hall_signals.answer_deviation,
                },
                "votes": hall_votes,
            })

    dom_labelers = ["lexicon", "embed", "bm25_ratio", "cross_encoder"]
    L_dom = build_label_matrix([r["votes"] for r in rows_domain], dom_labelers)
    
    p_dom = aggregate_probabilities(L_dom, seed=int(cfg.get("aggregation", "seed", default=42)), labeler_names=dom_labelers)
    for i, r in enumerate(rows_domain):
        r["score"] = round(float(p_dom[i]), 2)

    if rows_hallu:
        hall_labelers = ["factual_precision", "obscurity", "complexity", "answer_dev"]
        L_h = build_label_matrix([r["votes"] for r in rows_hallu], hall_labelers)
        
        p_h = aggregate_probabilities(L_h, seed=int(cfg.get("aggregation", "seed", default=42)))
        for i, r in enumerate(rows_hallu):
            r["score"] = round(float(p_h[i]), 2)

    df_dom = pd.DataFrame(rows_domain)
    df_hall = pd.DataFrame(rows_hallu) if rows_hallu else pd.DataFrame(columns=["id"]) 

    os.makedirs(out_dir, exist_ok=True)
    dom_out_parquet = os.path.join(out_dir, "domain_relevance_dataset.parquet")
    dom_out_json = os.path.join(out_dir, "domain_relevance_dataset.json")
    hall_out_parquet = os.path.join(out_dir, "hallucination_risk_dataset.parquet")
    hall_out_json = os.path.join(out_dir, "hallucination_risk_dataset.json")
    
    df_dom.to_parquet(dom_out_parquet, index=False)
    df_dom.to_json(dom_out_json, orient="records", indent=2)
    
    if not df_hall.empty:
        df_hall.to_parquet(hall_out_parquet, index=False)
        df_hall.to_json(hall_out_json, orient="records", indent=2)

    with open(os.path.join(art_dir, "domain_labelers.txt"), "w", encoding="utf-8") as f:
        f.write("".join(dom_labelers))
    if rows_hallu:
        with open(os.path.join(art_dir, "hallucination_labelers.txt"), "w", encoding="utf-8") as f:
            f.write("".join(hall_labelers))

def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    build_dataset(cfg)

if __name__ == "__main__":
    cli()
