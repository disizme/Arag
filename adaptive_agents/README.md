# Adaptive RAG Agents

A standalone, plug-and-play system implementing the **Dual Question Framework** for adaptive retrieval-augmented generation. The system uses trained DeBERTa regression models to intelligently route queries between direct LLM responses and RAG-enhanced responses.

## Architecture Overview

The system consists of two specialized agents that analyze queries **before** they are sent to the core RAG system:

### 1. Hallucination Predictor Agent
- **Purpose**: Predicts if the LLM in the core system might hallucinate given the input query
- **Timing**: Pre-response analysis (before query goes to core LLM)
- **Model**: DeBERTa-v3-base fine-tuned regression model
- **Output**: Risk score (0.0-1.0) indicating likelihood of hallucination
- **Location**: `models/saved_models/v2/hallucination_predictor_v2/`

### 2. Specialization Predictor Agent  
- **Purpose**: Determines if the query requires domain-specific/course-specific knowledge
- **Timing**: Pre-response analysis (concurrent with hallucination prediction)
- **Model**: DeBERTa-v3-base fine-tuned regression model
- **Output**: Need score (0.0-1.0) indicating requirement for specialized context
- **Location**: `models/saved_models/v2/specialization_predictor_v2/`

### 3. Query Complexity Classifier (Optional)
- **Purpose**: Predicts complexity of the query
- **Timing**: Pre-response analysis (before query goes to core LLM)
- **Model**: T5-large fine-tuned classification model
- **Output**: Classification A, B or C corresponding to No Fetch, Single Fetch and Multi-fetch
- **Location**: `models/saved_models/v2/adaptive_rag_classifier/`
- **Source**: Agent trained based on Paper by Soyeong Jeong, etal.
    #### Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

### 4. Adaptive Wrapper Agent
- **Purpose**: Coordinates both predictors and makes final routing decision; provides API to use the query classifier separately
- **Logic**: Routes to different RAG retrieval strategies (shallow/dense/hybrid/multi-step) vs direct LLM based on score thresholds

## Key Features
### 🤖 Dual Question Framework
1. **Question 1**: Will the core LLM likely hallucinate on this query?
2. **Question 2**: Does this query need specialized domain knowledge?

### 📊 Trained Models
- **Hallucination Predictor**: DeBERTa-v3-base regression model (trained and ready)
- **Specialization Predictor**: DeBERTa-v3-base regression model (trained and ready)
- **Output Format**: `PredictionResult` with `score`

### 🎓 Training Infrastructure
- Multi-device support (CUDA, Apple Silicon MPS, CPU)
- Designed to work with Kaggle environment
- **Location**: `training\predictor-trainer-regression.ipynb`
- Uses Comprehensive datasets from MMLU, SQuAD, AmbigQA, and domain-specific sources

## Quick Start

### Interactive CLI Testing
```bash
# Test individual agents
python predict_adaptive.py "What is machine learning?"
python predict_adaptive.py --interactive

# Evaluate agents with datasets
python evaluation/evaluate_single_agent.py --dataset data.json --agent hallucination
python evaluation/evaluate_single_agent.py --dataset data.json --agent specialization
```

### Programmatic Usage
```python
from agents.adaptive_wrapper import AdaptiveWrapper

# Initialize the wrapper agent (loads trained models automatically)
wrapper = AdaptiveWrapper()

# Analyze a query and get routing decision
query = "What was the GDP of France in 2019?"
decision = await wrapper.analyze_query(query)
# Access prediction results
print(f"Hallucination Risk: {decision.hallucination_risk.score:.3f}")
print(f"Specialization Need: {decision.specialization_need.score:.3f}")
print(f"Strategy: {decision.strategy}")
# Access classifier result
classifier_decision = await wrapper.predict_query_complexity(query)
print(f"Classifier Decision: {classifier_decision.complexity.label}")
```

## Evaluation

The system includes comprehensive evaluation tools:

- **`evaluate_single_agent.py`**: Focused evaluation for individual agents
- **`individual_model_evaluation.py`**: Comprehensive evaluation framework
- **`test_models.py`**: Quick model testing and verification

### Example Evaluation
```bash
# Evaluate hallucination predictor with your dataset
python evaluation/evaluate_single_agent.py \
    --dataset evaluation_data.json \
    --agent hallucination \
    --output-dir results/

# Output includes:
# - Performance metrics (MSE, MAE, R², correlation)
# - Threshold analysis for binary classification
# - Visualization plots (scatter, residuals, distributions)
# - Detailed results in JSON and CSV formats
```

This system provides intelligent query routing to optimize both accuracy and performance of RAG systems using production-ready trained models.