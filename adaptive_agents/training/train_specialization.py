"""
Specialization Predictor Training Script

Trains transformer-based specialization predictors using processed datasets.
The model learns to assess query specialization needs and predict domain requirements.

Features:
- Uses processed_specialization_dataset.json with train/val/test splits
- Supports both T5 (text-to-text) and classification approaches
- Multiple architecture options (T5, DeBERTa, RoBERTa, DistilBERT)
- Advanced training optimizations and evaluation metrics
- Comprehensive logging and model checkpointing

Alternative model suggestions:
T5 models (text-to-text generation):
- google/t5-small: Faster training, good for prototyping
- google/t5-base: Balanced performance and speed (recommended)
- google/t5-large: Better performance, requires more resources
- google/flan-t5-base: Instruction-tuned variant

Classification models:
- microsoft/deberta-v3-base: Best classification performance
- roberta-base: Excellent general-purpose classifier
- distilbert-base-uncased: Fastest training, good efficiency
- bert-base-uncased: Reliable baseline performance
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Disable wandb and threading issues
os.environ["WANDB_DISABLED"] = "true"
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding, DataCollatorForSeq2Seq,
    set_seed, GenerationConfig
)
from datasets import Dataset, DatasetDict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import checkpoint utilities and shared training utilities
from checkpoint_utils import CheckpointManager
from utils_training import (
    log_device_info, create_data_splits, get_testing_config, get_production_config, 
    setup_argument_parser, merge_configs, setup_training_logging, cleanup_logging_and_exit
)

# Setup logging
logger, log_file = setup_training_logging("specialization")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# log_device_info function moved to training_utils.py

class SpecializationPredictor:
    """
    Advanced specialization prediction model trainer.
    
    Trains transformer models to predict specialization needs from query text.
    Supports both text-to-text generation (T5) and classification approaches.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",  # Default to regression model
        model_type: str = "regression",  # "t5" or "regression"
        output_dir: str = None,
        num_labels: int = 1,  # Regression output (0-1 score)
        max_input_length: int = 512,
        max_target_length: int = 64,
        cache_dir: str = None,
        seed: int = 42,
        resume_from_checkpoint: bool = True
    ):
        """
        Initialize the specialization predictor trainer.
        
        Args:
            model_name: HuggingFace model identifier
            model_type: "t5" for text-to-text or "classification" for classification
            output_dir: Directory to save trained model
            num_labels: Number of classification labels (for classification models)
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length (T5 only)
            cache_dir: Directory for caching downloaded models
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.model_type = model_type.lower()
        self.num_labels = num_labels
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.seed = seed
        
        # Validate model type
        if self.model_type not in ["t5", "regression"]:
            raise ValueError("model_type must be 't5' or 'regression'")
        
        # Set random seeds
        set_seed(seed)
        
        # Setup directories
        if output_dir is None:
            output_dir = f"../models/saved_models/specialization_predictor_updated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.datasets_dir = Path(__file__).parent.parent / "datasets"
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_collator = None
        self.generation_config = None
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Specialization text mappings
        self.score_to_text_map = {
            0.0: "general knowledge",
            0.2: "low specialization",
            0.4: "medium specialization", 
            0.6: "high specialization",
            0.8: "very high specialization"
        }
        
        self.text_to_score_map = {v: k for k, v in self.score_to_text_map.items()}
        
        logger.info(f"[SPECIALIZATION-PREDICTOR] Initialized")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Labels: {num_labels}")
        logger.info(f"  Input Length: {max_input_length}")
        logger.info(f"  Output: {self.output_dir}")
    
    def load_processed_dataset(self) -> List[Dict]:
        """Load processed specialization dataset."""
        dataset_path = self.datasets_dir / "processed_specialization_dataset.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"[DATASET] Loaded {len(data)} samples")
        return data
    
    def score_to_text(self, score: float) -> str:
        """Convert specialization score to text description."""
        # Find closest score mapping
        closest_score = min(self.score_to_text_map.keys(), key=lambda x: abs(x - score))
        return self.score_to_text_map[closest_score]
    
    def text_to_score(self, text: str) -> float:
        """Convert text description back to specialization score."""
        text_clean = text.lower().strip()
        return self.text_to_score_map.get(text_clean, 0.5)  # Default to medium
    
    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        Preprocess raw data for training.
        
        Converts scores to labels/text and prepares inputs.
        """
        processed = []
        score_distribution = []
        
        for item in data:
            question = item.get("question", "").strip()
            if not question:
                continue
                
            # Extract metadata
            answer = item.get("answer", "")
            source = item.get("source", "unknown")
            spec_score = float(item.get("score", 0.0))
            domain = item.get("domain", "unknown")
            
            score_distribution.append(spec_score)
            
            # Use actual score (0.0 to 1.0) for regression
            target = float(spec_score)
            # Ensure score is in valid range
            target = max(0.0, min(1.0, target))
            
            # Prepare input/target based on model type
            if self.model_type == "t5":
                # T5 with question only (no domain context)
                input_text = f"classify specialization: {question}"
                target_text = f"score: {target:.1f}"
            else:
                # Regression format with question only (no domain context)
                input_text = question
                target_text = target
            
            processed.append({
                "input_text": input_text,
                "target_text": target_text,
                "label": target,  # Regression score (0-1)
                "spec_score": spec_score,
                "domain_text": domain,  # Keep domain for analysis
                "source": source,
                "answer": answer,
                "question": question
            })
        
        # Log statistics
        scores = np.array(score_distribution)
        logger.info(f"[PREPROCESSING] Processed {len(processed)} samples")
        logger.info(f"  Score statistics: mean={scores.mean():.3f}, std={scores.std():.3f}")
        logger.info(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Label distribution
        labels = [item['label'] for item in processed]
        label_counts = pd.Series(labels).value_counts().sort_index()
        logger.info(f"  Label distribution: {label_counts.to_dict()}")
        
        return processed
    
    def create_splits(
        self,
        data: List[Dict],
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified train/validation/test splits using shared utility.
        """
        return create_data_splits(
            data=data,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            stratify=stratify,
            label_key='label',
            seed=self.seed
        )
    
    def setup_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        if self.tokenizer is not None:
            return
        
        if self.model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                legacy=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Setup data collator
        if self.model_type == "t5":
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                #max_length=self.max_input_length,
                pad_to_multiple_of=8
            )
        else:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                # max_length=self.max_input_length
            )
        
        logger.info(f"[TOKENIZER] Initialized: {self.tokenizer.__class__.__name__}")
        logger.info(f"  Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"  Max input length: {self.max_input_length}")
        if self.model_type == "t5":
            logger.info(f"  Max target length: {self.max_target_length}")
    
    def tokenize_data(self, examples):
        """Tokenize text data for model input."""
        if self.model_type == "t5":
            # T5 text-to-text format
            model_inputs = self.tokenizer(
                examples["input_text"],
                truncation=True,
                padding=False,  # Handle by data collator
                max_length=self.max_input_length,
                return_tensors=None
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples["target_text"],
                    truncation=True,
                    padding=False,
                    max_length=self.max_target_length,
                    return_tensors=None
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        else:
            # Classification model
            return self.tokenizer(
                examples["input_text"],
                truncation=True,
                padding=False,  # Handle by data collator
                max_length=self.max_input_length,
                return_tensors=None
            )
    
    def prepare_datasets(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> DatasetDict:
        """
        Load and prepare datasets for training.
        """
        # Load and preprocess data
        raw_data = self.load_processed_dataset()
        processed_data = self.preprocess_data(raw_data)
        
        # Create splits
        train_data, val_data, test_data = self.create_splits(
            processed_data, train_size, val_size, test_size
        )
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        
        # Define columns to remove during tokenization
        remove_columns = ["input_text", "target_text", "domain_text", "source", "answer", "question", "spec_score"]
        
        # Tokenize datasets
        logger.info("[TOKENIZATION] Tokenizing datasets...")
        train_dataset = train_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing train"
        )
        val_dataset = val_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing validation"
        )
        test_dataset = test_dataset.map(
            self.tokenize_data,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing test"
        )
        
        # Create dataset dict
        datasets = DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        })
        
        logger.info("[DATASETS] Prepared tokenized datasets")
        return datasets
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics."""
        predictions, labels = eval_pred
        
        if self.model_type == "t5":
            # Decode T5 predictions
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            # Convert text predictions to numeric labels
            pred_scores = [self.text_to_score(pred) for pred in decoded_preds]
            pred_labels = [min(int(score * self.num_labels), self.num_labels - 1) for score in pred_scores]
            
            # For T5, labels are token IDs, need to decode them first
            if isinstance(labels[0], (list, np.ndarray)):
                # Replace -100 with pad token for decoding
                labels_clean = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
                true_scores = [self.text_to_score(label) for label in decoded_labels]
            else:
                true_scores = labels
            
            # T5 regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            pred_scores_array = np.array(pred_scores)
            true_scores_array = np.array(true_scores)
            
            # Clip predictions to valid range [0, 1]
            pred_scores_clipped = np.clip(pred_scores_array, 0.0, 1.0)
            
            # Round to nearest 0.1 for interval accuracy  
            pred_scores_rounded = np.round(pred_scores_clipped, 1)
            true_scores_rounded = np.round(true_scores_array, 1)
            
            metrics = {
                'mse': mean_squared_error(true_scores_array, pred_scores_array),
                'mae': mean_absolute_error(true_scores_array, pred_scores_array), 
                'rmse': np.sqrt(mean_squared_error(true_scores_array, pred_scores_array)),
                'r2_score': r2_score(true_scores_array, pred_scores_array),
                'interval_accuracy': np.mean(pred_scores_rounded == true_scores_rounded),
                'mean_pred': np.mean(pred_scores_clipped),
                'std_pred': np.std(pred_scores_clipped)
            }
            
            
            # Add threshold-based accuracy (treat as binary at 0.5)
            binary_preds = (pred_scores_clipped > 0.5).astype(int)
            binary_true = (true_scores_array > 0.5).astype(int)
            metrics['binary_accuracy'] = accuracy_score(binary_true, binary_preds)
            
            return metrics
        else:
            # Regression model
            pred_scores = predictions.flatten()
            true_scores = labels.flatten()
            
            # Clip predictions to valid range [0, 1]
            pred_scores_clipped = np.clip(pred_scores, 0.0, 1.0)
            
            # Round to nearest 0.1 for interval accuracy
            pred_scores_rounded = np.round(pred_scores_clipped, 1)
            true_scores_rounded = np.round(true_scores, 1)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                'mse': mean_squared_error(true_scores, pred_scores),
                'mae': mean_absolute_error(true_scores, pred_scores),
                'rmse': np.sqrt(mean_squared_error(true_scores, pred_scores)),
                'r2_score': r2_score(true_scores, pred_scores),
                'interval_accuracy': np.mean(pred_scores_rounded == true_scores_rounded),
                'mean_pred': np.mean(pred_scores_clipped),
                'std_pred': np.std(pred_scores_clipped)
            }
            
            # Add threshold-based accuracy (treat as binary at 0.5)
            binary_preds = (pred_scores_clipped > 0.5).astype(int)
            binary_true = (true_scores > 0.5).astype(int)
            metrics['binary_accuracy'] = accuracy_score(binary_true, binary_preds)
            
            return metrics
    
    def setup_model(self):
        """Initialize the model."""
        if self.model is not None:
            return
        
        if self.model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Setup generation config
            self.generation_config = GenerationConfig(
                max_length=self.max_target_length,
                num_beams=1,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            # Regression model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,  # Single output for regression
                cache_dir=self.cache_dir,
                problem_type="regression"
            )
        
        # Resize token embeddings if needed
        if self.tokenizer is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"[MODEL] Loaded: {self.model_name}")
        logger.info(f"  Parameters: {self.model.num_parameters():,}")
        logger.info(f"  Problem type: {'Text-to-text generation' if self.model_type == 't5' else 'Regression (0-1 scores)'}")
    
    def compute_metrics(self, eval_pred):
        """Basic compute metrics for specialization prediction."""
        predictions, labels = eval_pred
        
        if self.model_type == "regression":
            # Regression metrics
            pred_scores = predictions.flatten()
            true_scores = labels.flatten()
            
            # Clip predictions to valid range [0, 1]
            pred_scores = np.clip(pred_scores, 0.0, 1.0)
            
            # Basic regression metrics
            mse = mean_squared_error(true_scores, pred_scores)
            mae = mean_absolute_error(true_scores, pred_scores)
            r2 = r2_score(true_scores, pred_scores)
            
            # Interval accuracy (within 0.1 of target)
            interval_accuracy = np.mean(np.abs(pred_scores - true_scores) <= 0.1)
            
            # Binary accuracy (threshold at 0.5)
            binary_accuracy = np.mean((pred_scores > 0.5) == (true_scores > 0.5))
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2_score': r2,
                'interval_accuracy': interval_accuracy,
                'binary_accuracy': binary_accuracy,
                'mean_pred': np.mean(pred_scores),
                'std_pred': np.std(pred_scores)
            }
        else:
            # T5 text generation metrics - placeholder
            return {}
    
    def compute_metrics_enhanced(self, eval_pred):
        """Enhanced compute metrics using advanced metrics from utils."""
        predictions, labels = eval_pred
        
        if self.model_type == "regression":
            # Use the advanced metrics computation from utils
            pred_scores = predictions.flatten()
            true_scores = labels.flatten()
            
            # Clip predictions to valid range [0, 1]
            pred_scores = np.clip(pred_scores, 0.0, 1.0)
            
            # Basic regression metrics
            metrics = {
                'mse': mean_squared_error(true_scores, pred_scores),
                'mae': mean_absolute_error(true_scores, pred_scores),
                'rmse': np.sqrt(mean_squared_error(true_scores, pred_scores)),
                'r2_score': r2_score(true_scores, pred_scores)
            }
            
            return metrics
        else:
            # T5 text generation metrics - placeholder
            return {}
    
    def train(
        self,
        datasets: DatasetDict,
        learning_rate: float = None,
        num_epochs: int = 3,
        batch_size: int = None,
        eval_batch_size: int = None,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        early_stopping_patience: int = 3,
        load_best_model: bool = True,
        fp16: bool = None,
        dataloader_num_workers: int = 0,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Train the specialization prediction model.
        """
        # Setup model and tokenizer
        self.setup_tokenizer()
        self.setup_model()
        
        # Check for checkpoint resume
        if resume_from_checkpoint is None:
            should_resume, checkpoint_path = self.checkpoint_manager.should_resume_training()
            if should_resume:
                resume_from_checkpoint = checkpoint_path
        
        # Set defaults based on model type
        if learning_rate is None:
            learning_rate = 3e-4 if self.model_type == "t5" else 2e-5
        
        if batch_size is None:
            batch_size = 8 if self.model_type == "t5" else 16
            
        if eval_batch_size is None:
            eval_batch_size = batch_size * 2
        
        # Auto-detect fp16 capability if not specified
        if fp16 is None:
            # Enable FP16 for CUDA GPUs with compute capability >= 7.0
            if torch.cuda.is_available():
                fp16 = torch.cuda.get_device_capability()[0] >= 7
            # MPS (Apple Silicon) doesn't support fp16 mixed precision
            else:
                fp16 = False
        
        # Use standard loss functions
        if self.model_type == "regression":
            logger.info(f"[LOSS] Using standard MSE Loss")
        else:
            logger.info(f"[LOSS] Using standard Cross-Entropy Loss")
        
        # Training arguments with improvements
        training_args_dict = {
            "output_dir": str(self.output_dir),
            "learning_rate": learning_rate,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": eval_batch_size,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "logging_steps": logging_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "save_total_limit": 3,
            "load_best_model_at_end": load_best_model,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": "none",
            "push_to_hub": False,
            "fp16": fp16,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "remove_unused_columns": True,
            "seed": self.seed,
            "data_seed": self.seed,
            "log_level": "info"
        }
        
        # Add T5-specific parameters only for T5 models
        if self.model_type == "t5":
            training_args_dict["predict_with_generate"] = True
            training_args_dict["generation_max_length"] = self.max_target_length
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Set trainer in checkpoint manager for graceful shutdown
        self.checkpoint_manager.set_trainer(self.trainer)
        
        # Log training setup
        logger.info("[TRAINING] Starting model training")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Type: {self.model_type}")
        logger.info(f"  Device: {training_args.device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size} (train) / {eval_batch_size} (eval)")
        logger.info(f"  FP16: {'✅' if fp16 else '❌'}")
        logger.info(f"  Warmup ratio: {warmup_ratio}")
        logger.info(f"  Weight decay: {weight_decay}")
        
        # Train model (with optional resume)
        if resume_from_checkpoint:
            logger.info(f"[TRAINING] Resuming from checkpoint: {resume_from_checkpoint}")
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = self.trainer.train()
        
        # Check if training was interrupted
        if self.checkpoint_manager.training_interrupted:
            logger.info("[TRAINING] Training was interrupted - checkpoint saved")
            return {
                "status": "interrupted",
                "message": "Training interrupted by user - checkpoint saved",
                "checkpoint_path": str(self.output_dir)
            }
        
        # Save model and tokenizer
        logger.info("[TRAINING] Saving model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save generation config for T5
        if self.model_type == "t5" and self.generation_config:
            self.generation_config.save_pretrained(self.output_dir)
        
        # Evaluate on test set
        logger.info("[EVALUATION] Evaluating on test set...")
        test_results = self.trainer.evaluate(datasets["test"], metric_key_prefix="test")
        
        # Generate predictions for detailed analysis
        test_predictions = self.trainer.predict(datasets["test"])
        
        # Handle predictions based on model type
        class_report = None
        conf_matrix = None
        
        if self.model_type == "t5":
            decoded_preds = self.tokenizer.batch_decode(test_predictions.predictions, skip_special_tokens=True)
            pred_scores = [self.text_to_score(pred) for pred in decoded_preds]
            pred_labels = [min(int(score * self.num_labels), self.num_labels - 1) for score in pred_scores]
            
            # Handle T5 labels
            if isinstance(test_predictions.label_ids[0], (list, np.ndarray)):
                labels_clean = np.where(test_predictions.label_ids != -100, 
                                      test_predictions.label_ids, self.tokenizer.pad_token_id)
                decoded_labels = self.tokenizer.batch_decode(labels_clean, skip_special_tokens=True)
                true_scores = [self.text_to_score(label) for label in decoded_labels]
                true_labels = [min(int(score * self.num_labels), self.num_labels - 1) for score in true_scores]
            else:
                true_labels = test_predictions.label_ids
            
            # Classification metrics only for T5
            class_report = classification_report(
                true_labels, pred_labels,
                output_dict=True,
                zero_division=0
            )
            conf_matrix = confusion_matrix(true_labels, pred_labels).tolist()
            
        elif self.model_type == "regression":
            # For regression, we don't need classification metrics
            # The test_results already contain regression metrics from compute_metrics
            pass
        
        # Compile results
        results = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "task_type": "specialization_prediction",
            "num_labels": self.num_labels,
            "training_args": training_args.to_dict(),
            "train_results": train_result.metrics,
            "test_results": test_results,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "dataset_sizes": {
                "train": len(datasets["train"]),
                "validation": len(datasets["validation"]),
                "test": len(datasets["test"])
            },
            "training_time": train_result.metrics.get("train_runtime", 0),
            "trained_at": datetime.now().isoformat(),
            "seed": self.seed
        }
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Log final results
        logger.info("[TRAINING] Training completed successfully!")
        if self.model_type == "regression":
            logger.info(f"  Test MSE: {test_results.get('test_mse', 0):.4f}")
            logger.info(f"  Test MAE: {test_results.get('test_mae', 0):.4f}")
            logger.info(f"  Test R²: {test_results.get('test_r2_score', 0):.4f}")
            logger.info(f"  Test Interval Accuracy: {test_results.get('test_interval_accuracy', 0):.4f}")
        else:
            logger.info(f"  Test Accuracy: {test_results.get('test_accuracy', 0):.4f}")
            logger.info(f"  Test F1 (weighted): {test_results.get('test_f1_weighted', 0):.4f}")
        logger.info(f"  Training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
        logger.info(f"  Results saved: {results_path}")
        
        return results


def train_specialization_predictor(
    model_name: str = "google/t5-base",
    model_type: str = "t5",
    num_labels: int = 5,
    learning_rate: float = None,
    num_epochs: int = 3,
    batch_size: int = None,
    eval_batch_size: int = None,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 100,
    resume_from_checkpoint: bool = True,
    **kwargs
) -> Dict:
    """
    Convenience function to train specialization predictor.
    
    T5 models (text-to-text):
    - google/t5-small: Faster training, good for prototyping
    - google/t5-base: Balanced performance and speed (recommended)
    - google/t5-large: Better performance, requires more resources
    - google/flan-t5-base: Instruction-tuned variant
    
    Classification models:
    - microsoft/deberta-v3-base: Best classification performance
    - roberta-base: Excellent general-purpose classifier
    - distilbert-base-uncased: Fastest training
    - bert-base-uncased: Reliable baseline
    """
    # Separate init args, dataset args, and training args
    init_kwargs = {k: v for k, v in kwargs.items() 
                   if k in ['output_dir', 'max_input_length', 'max_target_length', 'cache_dir', 'seed']}
    
    dataset_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['train_size', 'val_size', 'test_size']}
    
    # Filter out parameters that don't belong to the train method
    excluded_params = {
        'output_dir', 'max_input_length', 'max_target_length', 'cache_dir', 'seed',
        'train_size', 'val_size', 'test_size', 'optim', 'adam_beta1', 
        'adam_beta2', 'adam_epsilon', 'fp16', 'dataloader_num_workers', 
        'dataloader_pin_memory', 'resume_from_checkpoint'
    }
    train_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}
    
    predictor = SpecializationPredictor(
        model_name=model_name,
        model_type=model_type,
        num_labels=num_labels,
        resume_from_checkpoint=resume_from_checkpoint,
        **init_kwargs
    )
    
    # Prepare datasets with configuration values
    datasets = predictor.prepare_datasets(
        train_size=dataset_kwargs.get('train_size', 0.7),
        val_size=dataset_kwargs.get('val_size', 0.15),
        test_size=dataset_kwargs.get('test_size', 0.15)
    )
    
    # Train model
    results = predictor.train(
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
        **train_kwargs
    )
    
    return results


def main():
    """Main training function with argument parsing."""
    try:
        # Setup argument parser
        parser = setup_argument_parser("Specialization Predictor Training")
        parser.add_argument(
            "--model-type", 
            type=str, 
            choices=["t5", "regression"],
            default="regression",
            help="Model type: t5 for text-to-text or regression for regression model"
        )
        args = parser.parse_args()
        
        logger.info("=" * 70)
        logger.info("SPECIALIZATION PREDICTOR TRAINING")
        logger.info("=" * 70)
        
        # Log available device information
        log_device_info()
        
        # Base configurations for different model types
        if args.model_type == "t5":
            base_config = {
                "model_name": "google/t5-base",
                "model_type": "t5",
                "num_labels": 5,
                "learning_rate": 3e-4
            }
        else:  # regression
            base_config = {
                "model_name": "microsoft/deberta-v3-base",
                "model_type": "regression",
                "num_labels": 1,                    # Regression output (0-1 score)
                "learning_rate": 2e-5
            }
        
        # Get configuration based on mode
        if args.test:
            logger.info("🧪 TESTING MODE: Using lower values for quick testing")
            mode_config = get_testing_config()
        else:
            logger.info("🚀 PRODUCTION MODE: Using production training configuration")
            mode_config = get_production_config()
        
        # Merge configurations with command line overrides
        config = merge_configs(base_config, mode_config, args)
        
        logger.info(f"Training configuration: {config}")
        
        # Train model
        results = train_specialization_predictor(**config)
        
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        
        # Show appropriate metrics based on model type
        model_type = config.get('model_type', 'regression')
        if model_type == 'regression':
            logger.info(f"Final Test MSE: {results['test_results'].get('test_mse', 0):.4f}")
            logger.info(f"Final Test R²: {results['test_results'].get('test_r2_score', 0):.4f}")
        else:
            logger.info(f"Final Test Accuracy: {results['test_results'].get('test_accuracy', 0):.4f}")
            logger.info(f"Final Test F1: {results['test_results'].get('test_f1_weighted', 0):.4f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


if __name__ == "__main__":
    try:
        main()
        logger.info(f"Log file saved: {log_file}")
    finally:
        cleanup_logging_and_exit()
