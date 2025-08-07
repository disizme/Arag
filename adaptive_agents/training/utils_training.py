"""
Shared Training Utilities

Common functions and utilities used by both hallucination and specialization
training scripts to reduce code duplication.
"""
import argparse
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

logger = logging.getLogger(__name__)

# Global flag to prevent multiple logging setups
_logging_configured = False

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Calculate class weights if not provided
        if self.class_weights is None:
            # You can compute weights from your training data
            loss_fct = torch.nn.CrossEntropyLoss()
        else:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        
        loss = loss_fct(logits.view(-1, 11), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def setup_training_logging(script_name: str):
    """
    Setup logging to both console and file. Ensures only one log configuration per run.
    
    Args:
        script_name: Name of the training script (e.g., 'hallucination', 'specialization')
    
    Returns:
        tuple: (logger, log_file_path)
    """
    global _logging_configured
    
    # Prevent multiple logging configurations in the same run
    if _logging_configured:
        logger = logging.getLogger(__name__)
        # Find existing log file from current handlers
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                return logger, Path(handler.baseFilename)
        return logger, None
    
    # Create logs directory
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file (single file per script type)
    date_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{date_prefix}_{script_name}_training.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Override existing configuration
    )
    
    _logging_configured = True
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    return logger, log_file


def cleanup_logging_and_exit():
    """Cleanup logging and exit."""
    global _logging_configured
    logger.info("Training session completed. Closing log file.")
    logging.shutdown()
    _logging_configured = False  # Reset flag for next run
    

def log_device_info():
    """
    Log information about the available device for training.
    TrainingArguments will handle device selection automatically.
    """
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_capability = torch.cuda.get_device_capability()[0]
        fp16_supported = compute_capability >= 7
        
        logger.info(f"🚀 CUDA GPU detected: {device_name}")
        logger.info(f"   GPU Memory: {gpu_memory:.1f} GB")
        logger.info(f"   Compute Capability: {compute_capability}.x")
        
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info(f"🍎 Apple Silicon GPU (MPS) detected")
        
    else:
        logger.warning(f"⚠️  No GPU detected - training will use CPU (much slower)")

def create_data_splits(
    data: List[Dict],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    label_key: str = 'label',
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create stratified train/validation/test splits with support for testing mode.
    
    Args:
        data: Preprocessed data
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        stratify: Whether to stratify splits by label
        label_key: Key name for labels in data dict
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # For testing with smaller datasets, allow splits that don't sum to 1.0
    total_size = train_size + val_size + test_size
    if total_size <= 0.1:  # Testing mode with small splits
        stratify = False
        logger.info(f"[TESTING MODE] Using small dataset splits: {total_size:.3f} of total data")
    elif abs(total_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to 1.0")
    
    df = pd.DataFrame(data)

    # For testing mode with small splits, sample from the dataset first
    if total_size < 1.0:
        # Sample the required portion of data first
        sample_size = max(10, int(len(df) * total_size))  # Minimum 10 samples
        sample_size = min(sample_size, len(df))  # Don't exceed available data
        
        if sample_size < 10:
            logger.warning(f"Dataset too small for splits. Using all {len(df)} samples.")
            sample_size = len(df)
            
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        
        # Normalize split sizes to work with sampled data
        norm_factor = 1.0 / total_size
        train_size *= norm_factor
        val_size *= norm_factor
        test_size *= norm_factor
    
    # Simple stratification
    if stratify:
        try:
            # 11 unique scores in the dataset 0.0 to 1.0
            stratify_col = pd.qcut(df[label_key], q=11, labels=False, duplicates='drop')
        except (ValueError, TypeError):
            stratify_col = None
    else:
        stratify_col = None
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=stratify_col,
        shuffle=True
    )
    
    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=temp_df[label_key] if stratify else None,
        shuffle=True
    )
    
    # Convert to records
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    test_data = test_df.to_dict('records')
    
    # Ensure minimum sizes for each split
    min_samples = 2  # Minimum samples per split
    if len(train_data) < min_samples:
        logger.warning(f"Train set too small ({len(train_data)}). Consider using more data.")
    if len(val_data) < min_samples:
        logger.warning(f"Validation set too small ({len(val_data)}). Consider using more data.")
    if len(test_data) < min_samples:
        logger.warning(f"Test set too small ({len(test_data)}). Consider using more data.")
    
    # Log split info with score distribution analysis
    logger.info(f"[SPLITS] Created dataset splits:")
    logger.info(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    # Label distributions per split
    for name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        labels = [item[label_key] for item in split_data]
        counts = pd.Series(labels).value_counts().sort_index()
        logger.info(f"  {name} labels: {counts.to_dict()}")
    
    # Detailed score distribution logging (only in testing mode for brevity)
    if total_size <= 0.1:  # Testing mode
        for name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if len(split_data) > 0:
                labels = [item[label_key] for item in split_data]
                counts = pd.Series(labels).value_counts().sort_index()
                count_dict = {float(k): int(v) for k, v in counts.items()}
                logger.info(f"  {name} score distribution: {count_dict}")
    
    return train_data, val_data, test_data

def get_testing_config():
    """
    Get configuration parameters optimized for testing mode.
    
    Returns:
        Dictionary with lower values suitable for quick testing
    """
    return {
        # Dataset splits
        "train_size": 0.05,    # lower for testing (was 0.7)
        "val_size": 0.01,      # lower for testing (was 0.15)
        "test_size": 0.01,     # lower for testing (was 0.15)
        
        # Training parameters
        "num_epochs": 1,       # lower for testing (was 3-4)
        "batch_size": 4,       # lower for testing (was 16)
        "eval_batch_size": 4,  # lower for testing (was 32)
        "save_steps": 20,      # lower for testing (was 500)
        "eval_steps": 20,      # lower for testing (was 250-500)
        "logging_steps": 5,    # lower for testing (was 50-100)
        
        # Other parameters
        "early_stopping_patience": 1,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01
    }

def get_production_config():
    """
    Get configuration parameters optimized for production training.
    
    Returns:
        Dictionary with full values for production training
    """
    return {
        # Dataset splits
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        
        # Training parameters
        "num_epochs": 30,
        "batch_size": 16,
        "eval_batch_size": 16,
        "save_steps": 100,
        "eval_steps": 50,
        "logging_steps": 50,
        
        
        # Other parameters
        "early_stopping_patience": 5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01
    }

def setup_argument_parser(script_name: str = "training_script"):
    """
    Setup argument parser with common arguments for training scripts.
    
    Args:
        script_name: Name of the script for help text
        
    Returns:
        Configured argument parser
    """
    
    parser = argparse.ArgumentParser(
        description=f"{script_name} - Train model with configurable parameters"
    )
    
    parser.add_argument(
        "--agent-type",
        type=str,
        default="hallucination",
        help="Type of agent to train (hallucination or specialization)"
    )
    
    # Mode selection
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Use testing mode with lower values for quick testing"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name to use for training"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=2e-5,
        help="Learning rate for training"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--eval-batch-size", 
        type=int, 
        default=None,
        help="Evaluation batch size"
    )
    
    
    # Other options
    parser.add_argument(
        "--no-resume", 
        action="store_true",
        help="Don't resume from checkpoint"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for saved model"
    )
    
    return parser

def merge_configs(base_config: Dict, override_config: Dict, args) -> Dict:
    """
    Merge configuration dictionaries with command line argument overrides.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration (e.g., testing config)
        args: Parsed command line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Start with base config
    config = base_config.copy()
    
    # Apply override config
    config.update(override_config)
    
    # Apply command line arguments (if provided)
    if args.epochs is not None:
        config["num_epochs"] = args.epochs
    
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
        
    if args.model is not None:
        config["model_name"] = args.model
        
    if args.learning_rate != 2e-5:  # If different from default
        config["learning_rate"] = args.learning_rate
        
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
        
    # Handle model_type for specialization training
    if hasattr(args, 'model_type') and args.model_type is not None:
        config["model_type"] = args.model_type
        
    config["resume_from_checkpoint"] = not args.no_resume
    
    return config









