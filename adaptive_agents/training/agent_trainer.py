"""
Hallucination Predictor Training Script

Trains a transformer-based hallucination predictor using processed datasets.
The model learns to assess query complexity and predict hallucination risk scores.

Features:
- Uses processed_hallucination_dataset.json with train/val/test splits
- Supports multiple architectures (DeBERTa, RoBERTa, DistilBERT, BERT)
- Comprehensive evaluation metrics and model checkpointing
- Configurable training parameters with early stopping
- Detailed logging and regression reports

Alternative model suggestions:
- microsoft/deberta-v3-base: Best performance, recommended for production
- microsoft/deberta-v3-small: Faster training, good balance
- roberta-base: Excellent general-purpose performance
- distilbert-base-uncased: Fastest training, good for development
- bert-base-uncased: Classic baseline, reliable performance
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sklearn.utils import compute_class_weight
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
    AutoModelForSequenceClassification,
)
from datasets import Dataset, DatasetDict

sys.path.append(str(Path(__file__).parent.parent))
from checkpoint_utils import CheckpointManager
from utils_training import (
    WeightedTrainer,
    log_device_info,
    create_data_splits,
    get_testing_config,
    get_production_config,
    setup_argument_parser,
    merge_configs,
    setup_training_logging,
    cleanup_logging_and_exit,
)

logger, log_file = setup_training_logging("hallucination")
warnings.filterwarnings("ignore", category=FutureWarning)

class HallucinationPredictor:
    """
    Advanced hallucination prediction model trainer.

    Trains transformer models to predict hallucination risk from query text.
    Supports regression with comprehensive evaluation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        output_dir: str = None,
        num_labels: int = 1,    # Number of labels (1 for regression)
        max_length: int = 512,  # Maximum sequence length for tokenization
        cache_dir: str = None,  # Directory for caching downloaded models
        seed: int = 42,         # Random seed for reproducibility
        resume_from_checkpoint: bool = True,
        class_weights: List[float] = None,
        agent_type: str = "hallucination",
        loss_type: str = "ce",
    ):
        """
        Initialize the hallucination predictor trainer.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.seed = seed
        set_seed(seed)
        if output_dir is None:
            output_dir = f"../models/saved_models/{agent_type}_predictor_test_{loss_type}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.datasets_dir = Path(__file__).parent.parent / "datasets"
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_collator = None
        self.training_history = {}
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir, resume_from_checkpoint=resume_from_checkpoint
        )
        logger.info("[HALLUCINATION-PREDICTOR] Initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Labels: {num_labels}")
        logger.info(f"  Max Length: {max_length}")
        logger.info(f"  Output: {self.output_dir}")

    def load_dataset(self, agent_type: str) -> List[Dict]:
        """Load hallucination dataset."""
        data_file = "domain_relevance_dataset.json" if agent_type == "specialization" else "hallucination_risk_dataset.json"
        dataset_path = self.datasets_dir / data_file
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"[DATASET] Loaded {len(data)} samples")
        return data

    def setup_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.data_collator = DataCollatorWithPadding(
        #     tokenizer=self.tokenizer, padding=True
        # ) # Tokenizer set to max_length, padding=True
        logger.info(f"[TOKENIZER] Initialized: {self.tokenizer.__class__.__name__}")
        logger.info(f"  Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"  Max length: {self.max_length}")

    def tokenize_data(self, data):
        """Tokenize text data for model input."""
        tokenized_data = self.tokenizer(
            data["question"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        return {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
            "labels": [int(score * 10) for score in data["score"]]
        }

    def prepare_datasets(
        self, agent_type: str, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15
    ) -> DatasetDict:
        """
        Load and prepare datasets for training.

        Returns:
            DatasetDict with tokenized train/validation/test splits
        """
        raw_data = self.load_dataset(agent_type)
        train_data, val_data, test_data = create_data_splits(
            data=raw_data,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            stratify=True,
            label_key="score",
            seed=self.seed,
        )
        
        self.setup_tokenizer()
        
        train_dataset = Dataset.from_list(train_data)
        train_labels = [int(data["score"] * 10) for data in train_dataset]
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(11),
            y=train_labels
        )
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        logger.info("[TOKENIZATION] Tokenizing datasets...")
        remove_columns = ["id","question", "answer", "reason", "source", "score"]
        train_dataset = train_dataset.map(
            self.tokenize_data,
            batched=True,
            desc="Tokenizing train",
            remove_columns=remove_columns,
        )
        val_dataset = val_dataset.map(
            self.tokenize_data,
            batched=True,
            desc="Tokenizing validation",
            remove_columns=remove_columns,
        )
        test_dataset = test_dataset.map(
            self.tokenize_data,
            batched=True,
            desc="Tokenizing test", 
            remove_columns=remove_columns,
        )

        datasets = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )
        logger.info(f"[DATASETS] Prepared tokenized datasets: {datasets}")
        return datasets

    def compute_metrics(self, eval_pred):
        """Compute regression evaluation metrics for 0-1 score prediction."""
        predictions, labels = eval_pred
        predictions = predictions.flatten() # Convert to 1D array
        labels = labels.flatten() 
        metrics = {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "rmse": np.sqrt(mean_squared_error(labels, predictions)),
            "r2": 1 - (np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
        }
        return metrics

    def setup_model(self):
        """Initialize the regression model."""
        if self.model is not None:
            return
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
            problem_type="regression"
            cache_dir=self.cache_dir
        )
        if self.tokenizer is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f"[MODEL] Loaded regression model: {self.model_name}")
        logger.info(f"  Parameters: {self.model.num_parameters():,}")
        logger.info("  Score outputs: 1 (regression [0-1])")
        logger.info("  Problem type: Regression")

    def train(
        self,
        datasets: DatasetDict,  # DatasetDict with train/validation/test splits
        learning_rate: float = 1e-5,   # Learning rate for optimization
        num_epochs: int = 15,    # Number of training epochs
        batch_size: int = 4,    # Training batch size
        eval_batch_size: int = 4,  # Evaluation batch size
        warmup_ratio: float = 0.1,  # Proportion of steps for learning rate warmup
        weight_decay: float = 0.01,  # L2 regularization strength
        save_steps: int = 20,  # Steps between model checkpoints
        eval_steps: int = 20,  # Steps between evaluations
        logging_steps: int = 5,  # Steps between logging
        early_stopping_patience: int = 3,  # Patience for early stopping
        load_best_model: bool = True,  # Whether to load best model at end
        fp16: bool = None,  # Enable mixed precision training (auto-detect if None)
        dataloader_num_workers: int = 0,  # Number of workers for data loading
        resume_from_checkpoint: Optional[str] = None,  # Resume from checkpoint if provided
        loss_type: str = "ce",  # 'ce' or 'emd'
    ) -> Dict:
        """
        Train the hallucination prediction model.

        Returns:
            Dictionary with training results and metrics
        """
        self.setup_tokenizer()
        self.setup_model()
        if resume_from_checkpoint is None:
            should_resume, checkpoint_path = (
                self.checkpoint_manager.should_resume_training()
            )
            if should_resume:
                resume_from_checkpoint = checkpoint_path
        if fp16 is None:
            if torch.cuda.is_available():
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                fp16 = torch.cuda.get_device_capability()[0] >= 7
            else:
                fp16 = False

        if loss_type.lower() == "emd":
            logger.info("[LOSS] Using ordinal EMD loss")
        else:
            logger.info("[LOSS] Using Cross-Entropy loss")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            metric_for_best_model="eval_los",
            greater_is_better=False,
            load_best_model_at_end=load_best_model,
            report_to="none",
            push_to_hub=False,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=False,
            remove_unused_columns=True,
            seed=self.seed,
            data_seed=self.seed,
        )
        
        self.trainer = WeightedTrainer(
            class_weights=self.class_weights,
            loss_type=loss_type,
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
        )

        self.checkpoint_manager.set_trainer(self.trainer)
        logger.info("[TRAINING] Starting model training")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Device: {training_args.device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size} (train) / {eval_batch_size} (eval)")
        logger.info(f"  Warmup ratio: {warmup_ratio}")
        logger.info(f"  Weight decay: {weight_decay}")
        if resume_from_checkpoint:
            logger.info(
                f"[TRAINING] Resuming from checkpoint: {resume_from_checkpoint}"
            )
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )
        else:
            train_result = self.trainer.train()
        if self.checkpoint_manager.training_interrupted:
            logger.info("[TRAINING] Training was interrupted - checkpoint saved")
            return {
                "status": "interrupted",
                "message": "Training interrupted by user - checkpoint saved",
                "checkpoint_path": str(self.output_dir),
            }
        logger.info("[TRAINING] Saving model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("[EVALUATION] Evaluating on test set...")
        test_results = self.trainer.evaluate(datasets["test"], metric_key_prefix="test")
        results = {
            "model_name": self.model_name,
            "task_type": "hallucination_prediction_classification",
            "num_labels": self.num_labels,
            "training_args": training_args.to_dict(),
            "train_results": train_result.metrics,
            "test_results": test_results,
            "dataset_sizes": {
                "train": len(datasets["train"]),
                "validation": len(datasets["validation"]),
                "test": len(datasets["test"]),
            },
            "training_time": train_result.metrics.get("train_runtime", 0),
            "trained_at": datetime.now().isoformat(),
            "seed": self.seed,
        }

        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("[TRAINING] Training completed successfully!")
        logger.info(f"  Test MSE: {test_results.get('test_mse', 0):.4f}")
        logger.info(f"  Test MAE: {test_results.get('test_mae', 0):.4f}")
        logger.info(f"  Test RMSE: {test_results.get('test_rmse', 0):.4f}")
        logger.info(f"  Test R2: {test_results.get('test_r2', 0):.4f}")
        logger.info(
            f"  Training time: {train_result.metrics.get('train_runtime', 0):.1f}s"
        )
        logger.info(f"  Results saved: {results_path}")
        return results

def train_hallucination_predictor(
    model_name: str = "microsoft/deberta-v3-base",
    learning_rate: float = 2e-05,
    num_epochs: int = 4,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    save_steps: int = 500,
    eval_steps: int = 250,
    logging_steps: int = 50,
    resume_from_checkpoint: bool = True,
    loss_type: str = "ce",
    **kwargs
) -> Dict:
    """
    Convenience function to train hallucination predictor.

    Alternative models to try:
    - microsoft/deberta-v3-base: Best performance (recommended)
    - microsoft/deberta-v3-small: Faster, good balance
    - roberta-base: Excellent general performance
    - distilbert-base-uncased: Fastest training
    - bert-base-uncased: Reliable baseline
    """

    init_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in ["output_dir", "num_labels", "max_length", "cache_dir", "seed", "agent_type"]
    }
    dataset_kwargs = {
        k: v for k, v in kwargs.items() if k in ["train_size", "val_size", "test_size", "agent_type"]
    }
    excluded_params = {
        "output_dir",
        "fp16",
        "adam_beta2",
        "optim",
        "max_length",
        "cache_dir",
        "dataloader_pin_memory",
        "val_size",
        "adam_beta1",
        "resume_from_checkpoint",
        "train_size",
        "num_labels",
        "dataloader_num_workers",
        "seed",
        "test_size",
    }
    train_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}
    predictor = HallucinationPredictor(
        model_name=model_name,
        resume_from_checkpoint=resume_from_checkpoint,
        loss_type=loss_type,
        **init_kwargs,
    )
    logger.info("Preparing datasets...")
    datasets = predictor.prepare_datasets(
        agent_type=dataset_kwargs.get("agent_type", "hallucination"),
        train_size=dataset_kwargs.get("train_size", 0.7),
        val_size=dataset_kwargs.get("val_size", 0.15),
        test_size=dataset_kwargs.get("test_size", 0.15),
    )

    # Train model
"""     results = predictor.train(
          datasets=datasets,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        loss_type=loss_type,
        **train_kwargs,
    ) """

    #return results


def main():
    """Main training function with argument parsing."""
    try:
        parser = setup_argument_parser("Hallucination Predictor Training")
        args = parser.parse_args()
        logger.info(
            "======================================================================"
        )
        logger.info("HALLUCINATION PREDICTOR TRAINING")
        logger.info(
            "======================================================================"
        )
        log_device_info()
        base_config = {
            "model_name": "microsoft/deberta-v3-base",
            "num_labels": 1,
            "learning_rate": 1e-05,
        }
        if args.test:
            logger.info("🧪 TESTING MODE: Using lower values for quick testing")
            mode_config = get_testing_config()
        else:
            logger.info("🚀 PRODUCTION MODE: Using production training configuration")
            mode_config = get_production_config()
        config = merge_configs(base_config, mode_config, args)
        results = train_hallucination_predictor(**config)
        logger.info(
            "======================================================================"
        )
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info(
            "======================================================================"
        )
        logger.info(f"Final Test MSE: {results['test_results'].get('test_mse', 0):.4f}")
        # MSE 0.01-0.02
        logger.info(f"Final Test R2: {results['test_results'].get('test_r2', 0):.4f}")
        # R2 0.7-0.8
        logger.info(f"Final Test MAE: {results['test_results'].get('test_mae', 0):.4f}")
        # MAE 0.1-0.2
        logger.info(f"Final Test RMSE: {results['test_results'].get('test_rmse', 0):.4f}")
        # RMSE 0.1-0.2
        
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


if __name__ == "__main__":
    try:
        # Usage: python agent_trainer.py --test --agent-type specialization/hallucination
        main()
        logger.info(f"Log file saved: {log_file}")
    finally:
        cleanup_logging_and_exit()
