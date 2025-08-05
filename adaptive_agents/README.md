# Adaptive RAG Agents

A standalone, plug-and-play system implementing the **Dual Question Framework** for adaptive retrieval-augmented generation.

## Architecture Overview

The system consists of two specialized agents that analyze queries **before** they are sent to the core RAG system:

### 1. Hallucination Predictor Agent
- **Purpose**: Predicts if the LLM in the core system might hallucinate given the input query
- **Timing**: Pre-response analysis (before query goes to core LLM)
- **Model**: DeBERTa-based fine-tuned classifier
- **Output**: Risk score (0.0-1.0) indicating likelihood of hallucination

### 2. Specialization Affordance Predictor Agent  
- **Purpose**: Determines if the query requires domain-specific/course-specific knowledge
- **Timing**: Pre-response analysis (concurrent with hallucination prediction)
- **Model**: T5-based fine-tuned classifier
- **Output**: Need score (0.0-1.0) indicating requirement for specialized context

### 3. Wrapper Agent
- **Purpose**: Coordinates both predictors and makes final routing decision
- **Logic**: Routes to RAG (single/multi-step) vs direct LLM based on predictions
- **Interface**: Clean API for integration with any RAG system

## Directory Structure

```
adaptive_agents/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── config/
│   ├── model_config.yaml              # Model configurations
│   └── training_config.yaml           # Training parameters
├── agents/
│   ├── __init__.py
│   ├── hallucination_predictor.py     # DeBERTa-based hallucination predictor
│   ├── specialization_predictor.py    # T5-based specialization predictor
│   └── adaptive_wrapper.py            # Main wrapper agent
├── models/
│   ├── __init__.py
│   ├── base_predictor.py              # Base class for predictors
│   └── saved_models/                  # Trained model checkpoints
├── datasets/
│   ├── __init__.py
│   ├── hallucination_dataset.py       # Dataset generation for hallucination detection
│   ├── specialization_dataset.py      # Dataset generation for specialization prediction
│   └── data_transformers.py           # Data preprocessing utilities
├── training/
│   ├── __init__.py
│   ├── train_hallucination.py         # Training script for hallucination predictor
│   ├── train_specialization.py        # Training script for specialization predictor
│   └── train_all.py                   # Complete training pipeline
├── evaluation/
│   ├── __init__.py
│   ├── evaluate_agents.py             # Evaluation framework
│   └── metrics.py                     # Custom evaluation metrics
└── utils/
    ├── __init__.py
    ├── model_utils.py                  # Model loading/saving utilities
    └── integration.py                  # Integration helpers for core RAG system
```

## Key Features

### 🎯 Pre-Response Analysis
- Analyzes queries **before** they reach the core LLM
- Prevents hallucinations by routing high-risk queries to RAG
- Optimizes performance by using direct LLM for simple queries

### 🔄 Plug-and-Play Design
- Standalone system with clean API
- Easy integration with any RAG system
- No dependencies on specific RAG implementations

### 🤖 Dual Question Framework
1. **Question 1**: Will the core LLM likely hallucinate on this query?
2. **Question 2**: Does this query need specialized domain knowledge?

### 📊 Model Recommendations
- **Hallucination Predictor**: DeBERTa-v3-base (fine-tuned)
- **Specialization Predictor**: T5-base (fine-tuned)
- **Alternative Models**: DistilBERT, RoBERTa options included

### 🎓 Training Pipeline
- Automated dataset generation with domain-specific keywords
- Transfer learning from pre-trained models
- Comprehensive evaluation metrics

## Quick Start

```python
from adaptive_agents import AdaptiveWrapper

# Initialize the wrapper agent
wrapper = AdaptiveWrapper()

# Analyze a query before sending to core RAG system
query = "What was the GDP of France in 2019?"
decision = await wrapper.analyze_query(query)

if decision.use_rag:
    # Route to RAG system with context retrieval
    response = core_rag_system.query_with_context(query)
else:
    # Route directly to LLM without context
    response = core_llm.query_direct(query)
```

## Integration Example

```python
# Easy integration with existing RAG systems
from adaptive_agents import AdaptiveWrapper

class YourRAGSystem:
    def __init__(self):
        self.adaptive_agent = AdaptiveWrapper()
        
    async def process_query(self, query: str):
        # Pre-response analysis
        decision = await self.adaptive_agent.analyze_query(query)
        
        if decision.use_rag:
            if decision.use_complex_reasoning:
                return await self.multi_step_rag(query)
            else:
                return await self.single_step_rag(query)
        else:
            return await self.direct_llm(query)
```

This system provides intelligent query routing to optimize both accuracy and performance of RAG systems.