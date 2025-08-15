
## Adaptive RAG Dataset Builder

This repository contains a reproducible pipeline to build high-quality training datasets for two agents:
- **Domain Relevance Classifier** (0–1): Determines if questions are related to AI/ML domain
- **Hallucination Risk Predictor** (0–1): Predicts likelihood of LLM hallucination for queries

The system uses **model-based weak supervision** with semantic signals, continuous scoring, and probabilistic label aggregation. No paid APIs or hardcoded keyword matching required.

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2) Prepare corpora and inputs
- **AI/ML corpus**: `data/corpus/ai_ml_corpus.jsonl` with fields: `{ "id": str, "text": str, "title": str(optional) }`
- **Questions**: Multiple JSON files in `data/raw/` with: `{ "id": str, "question": str, "answer": str(optional), "source": str }`
- **AI/ML lexicon**: `data/resources/ai_ml_lexicon.txt` (one term per line, optional but recommended)

The system automatically loads all `.json` files from the raw directory and combines them.

### 3) Configure
Edit `configs/config.yaml` to set models, retrieval parameters, and paths.

### 4) Build dataset
```bash
python make_dataset.py --config configs/config.yaml
```
Artifacts are saved to `data/processed/`:
- `domain_relevance_dataset.json` + `.parquet`
- `hallucination_risk_dataset.json` + `.parquet` 
- Label model diagnostics under `data/artifacts/`

### 5) Notes
- The pipeline first builds weak labels, aggregates probabilistic labels, balances splits by source, and outputs both features and labels.




**Model Recommendations:**

Cross-encoder (both domain + hallucination): `cross-encoder/stsb-distilroberta-base` is excellent for semantic similarity scoring and works well for both domain relevance and hallucination risk assessment.

Text generation model: `google/flan-t5-small` for answer deviation detection. Alternatives: `google/flan-t5-base` (better quality), `microsoft/DialoGPT-small` (conversational).

E5-base-v2 embeddings perform well for semantic retrieval and centroiding.

**Lightweight alternatives:**
- Cross-encoder: `cross-encoder/ms-marco-TinyBERT-L-2-v2` 
- Text generation: `google/flan-t5-small` (already lightweight)
- Embeddings: `intfloat/e5-small-v2`



## Signal Architecture

### **Domain Relevance Signals (4 signals)**
*Determines if questions are related to AI/ML domain*

1. **Lexicon Ratio** (`lexicon_ratio`)
   - **Method**: Word matching against AI/ML lexicon file
   - **Range**: 0.0-1.0 (proportion of AI/ML terms)
   - **Weight**: 20% (lowest priority)

2. **Embedding Similarity** (`embedding_sim`)
   - **Method**: Cosine similarity between question and AI domain centroid
   - **Range**: 0.0-1.0 (semantic similarity to AI topics)
   - **Weight**: 25%

3. **BM25 Score** (`bm25_ratio`)
   - **Method**: BM25 retrieval score against AI corpus
   - **Range**: 0.0-250+ (raw BM25 score, sigmoid scaling=0.04)
   - **Weight**: 30%

4. **Cross-Encoder Score** (`cross_encoder_score`)
   - **Method**: Cross-encoder similarity to AI domain templates
   - **Range**: 0.0-1.0 (after sigmoid normalization)
   - **Weight**: 35% (highest priority)

**Aggregation**: Weighted average with continuous sigmoid scoring around thresholds

---

### **Hallucination Risk Signals (4 signals)**
*Predicts likelihood of LLM hallucination based on query characteristics*

1. **Factual Precision Risk** (`factual_precision_risk`)
   - **Method**: Cross-encoder similarity with "This question requires specific factual information, precise facts, numbers, dates, or names"
   - **Risk**: Questions demanding precise facts are harder to answer correctly without fabrication
   - **Range**: 0.0-1.0 (higher = more hallucination risk)

2. **Obscurity Risk** (`obscurity_risk`)
   - **Method**: Cross-encoder similarity ratio between "specialized/niche knowledge" vs "general knowledge"
   - **Risk**: Obscure topics more likely to be hallucinated
   - **Range**: 0.0-1.0 (higher = more hallucination risk)

3. **Complexity Risk** (`complexity_risk`)
   - **Method**: Cross-encoder similarity with "This question requires complex multi-step reasoning"
   - **Risk**: Complex reasoning chains more error-prone
   - **Range**: 0.0-1.0 (higher = more hallucination risk)

4. **Answer Deviation** (`answer_deviation`)
   - **Method**: Generate answer with text generation model, compare semantic similarity to ground truth
   - **Risk**: High deviation indicates likely hallucination
   - **Range**: 0.0-1.0 (higher = more hallucination risk)

**Aggregation**: Equal-weight average of all 4 signals

---

## Key Features

- **Model-Based**: Uses NLI, embeddings, and cross-encoders instead of keyword matching
- **Continuous Scoring**: Sigmoid functions provide smooth transitions around thresholds  
- **Context-Free Hallucination**: Focuses on query characteristics, not answer validation
- **Domain-Agnostic**: Hallucination detection works across all topics
- **Weighted Domain Signals**: Prioritizes sophisticated signals over simple lexical matching