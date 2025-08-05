"""
Hallucination Predictor Training Script

Trains a transformer-based hallucination predictor using processed datasets.
The model learns to assess query complexity and predict hallucination risk scores.

Features:
- Uses processed_hallucination_dataset.json with train/val/test splits
- Supports multiple architectures (DeBERTa, RoBERTa, DistilBERT, BERT)
- Comprehensive evaluation metrics and model checkpointing
- Configurable training parameters with early stopping
- Detailed logging and classification reports

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

os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
sys.path.append(str(Path(__file__).parent.parent))
from checkpoint_utils import CheckpointManager
from utils_training import log_device_info, create_data_splits, get_testing_config, get_production_config, setup_argument_parser, merge_configs, setup_training_logging, cleanup_logging_and_exit

logger, log_file = setup_training_logging('hallucination')
warnings.filterwarnings('ignore', category=FutureWarning)

class HallucinationPredictor:
    """
    Advanced hallucination prediction model trainer.
    
    Trains transformer models to predict hallucination risk from query text.
    Supports binary and multi-class classification with comprehensive evaluation.
    """

    def __init__(self, model_name: str='microsoft/deberta-v3-base', output_dir: str=None, num_labels: int=1, max_length: int=512, cache_dir: str=None, seed: int=42, resume_from_checkpoint: bool=True):
        """
        Initialize the hallucination predictor trainer.
        
        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save trained model
            num_labels: Number of labels (1 for regression)
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory for caching downloaded models
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.seed = seed
        set_seed(seed)
        if output_dir is None:
            output_dir = f"../models/saved_models/hallucination_predictor_updated"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.datasets_dir = Path(__file__).parent.parent / 'datasets'
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_collator = None
        self.training_history = {}
        self.checkpoint_manager = CheckpointManager(output_dir=self.output_dir, resume_from_checkpoint=resume_from_checkpoint)
        logger.info('[HALLUCINATION-PREDICTOR] Initialized')
        logger.info(f'  Model: {model_name}')
        logger.info(f'  Labels: {num_labels}')
        logger.info(f'  Max Length: {max_length}')
        logger.info(f'  Output: {self.output_dir}')

    def load_processed_dataset(self) -> List[Dict]:
        """Load processed hallucination dataset."""
        dataset_path = self.datasets_dir / 'processed_hallucination_dataset.json'
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset not found: {dataset_path}')
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f'[DATASET] Loaded {len(data)} samples')
        return data

    def preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        Preprocess raw data for training.
        
        Converts scores to labels and prepares text inputs.
        """
        processed = []
        score_distribution = []
        for item in data:
            question = item.get('question', '').strip()
            if not question:
                continue
            answer = item.get('answer', '')
            source = item.get('source', 'unknown')
            risk_score = float(item.get('score', 0.0))
            score_distribution.append(risk_score)
            target = float(risk_score)
            target = max(0.0, min(1.0, target))
            processed.append({'text': question, 'label': target, 'risk_score': risk_score, 'source': source, 'answer': answer})
        scores = np.array(score_distribution)
        logger.info(f'[PREPROCESSING] Processed {len(processed)} samples')
        targets = [item['label'] for item in processed]
        targets_array = np.array(targets)
        
        # Log statistics
        logger.info(f"  Score statistics: mean={scores.mean():.3f}, std={scores.std():.3f}")
        logger.info(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        # Log label distribution
        logger.info(f"  Label distribution: {np.histogram(targets_array, bins=5, range=(0, 1))[0]}")
        
        return processed

    def create_splits(self, data: List[Dict], train_size: float=0.7, val_size: float=0.15, test_size: float=0.15, stratify: bool=True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create stratified train/validation/test splits using shared utility.
        """
        return create_data_splits(data=data, train_size=train_size, val_size=val_size, test_size=test_size, stratify=stratify, label_key='label', seed=self.seed)

    def setup_tokenizer(self):
        """Initialize tokenizer with proper configuration."""
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
        logger.info(f'[TOKENIZER] Initialized: {self.tokenizer.__class__.__name__}')
        logger.info(f'  Vocab size: {self.tokenizer.vocab_size}')
        logger.info(f'  Max length: {self.max_length}')

    def tokenize_data(self, examples):
        """Tokenize text data for model input."""
        return self.tokenizer(examples['text'], truncation=True, padding=False, max_length=self.max_length, return_tensors=None)

    def prepare_datasets(self, train_size: float=0.7, val_size: float=0.15, test_size: float=0.15) -> DatasetDict:
        """
        Load and prepare datasets for training.
        
        Returns:
            DatasetDict with tokenized train/validation/test splits
        """
        raw_data = self.load_processed_dataset()
        processed_data = self.preprocess_data(raw_data)
        train_data, val_data, test_data = self.create_splits(processed_data, train_size, val_size, test_size)
        self.setup_tokenizer()
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        test_dataset = Dataset.from_list(test_data)
        logger.info('[TOKENIZATION] Tokenizing datasets...')
        remove_columns = ['text', 'source', 'answer', 'risk_score']
        train_dataset = train_dataset.map(self.tokenize_data, batched=True, remove_columns=remove_columns, desc='Tokenizing train')
        val_dataset = val_dataset.map(self.tokenize_data, batched=True, remove_columns=remove_columns, desc='Tokenizing validation')
        test_dataset = test_dataset.map(self.tokenize_data, batched=True, remove_columns=remove_columns, desc='Tokenizing test')
        datasets = DatasetDict({'train': train_dataset, 'validation': val_dataset, 'test': test_dataset})
        logger.info('[DATASETS] Prepared tokenized datasets')
        return datasets

    def compute_metrics(self, eval_pred):
        """Compute regression evaluation metrics for 0-1 score prediction."""
        predictions, labels = eval_pred
        pred_scores = predictions.flatten()
        true_scores = labels.flatten()
        pred_scores_clipped = np.clip(pred_scores, 0.0, 1.0)
        pred_scores_rounded = np.round(pred_scores_clipped, 1)
        true_scores_rounded = np.round(true_scores, 1)
        metrics = {'mse': mean_squared_error(true_scores, pred_scores), 'mae': mean_absolute_error(true_scores, pred_scores), 'rmse': np.sqrt(mean_squared_error(true_scores, pred_scores)), 'r2_score': r2_score(true_scores, pred_scores), 'interval_accuracy': np.mean(pred_scores_rounded == true_scores_rounded), 'mean_pred': np.mean(pred_scores_clipped), 'std_pred': np.std(pred_scores_clipped)}
        binary_preds = (pred_scores_clipped > 0.5).astype(int)
        binary_true = (true_scores > 0.5).astype(int)
        metrics['binary_accuracy'] = accuracy_score(binary_true, binary_preds)
        return metrics

    def setup_model(self):
        """Initialize the regression model."""
        if self.model is not None:
            return
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1, cache_dir=self.cache_dir, problem_type='regression')
        if self.tokenizer is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        logger.info(f'[MODEL] Loaded regression model: {self.model_name}')
        logger.info(f'  Parameters: {self.model.num_parameters():,}')
        logger.info('  Score outputs: 1 (regression 0-1)')
        logger.info('  Problem type: Regression')

    def train(self, datasets: DatasetDict, learning_rate: float=2e-05, num_epochs: int=1, batch_size: int=4, eval_batch_size: int=4, warmup_ratio: float=0.1, weight_decay: float=0.01, save_steps: int=20, eval_steps: int=20, logging_steps: int=5, early_stopping_patience: int=3, load_best_model: bool=True, fp16: bool=None, dataloader_num_workers: int=0, resume_from_checkpoint: Optional[str]=None) -> Dict:
        """
        Train the hallucination prediction model.
        
        Args:
            datasets: DatasetDict with train/validation/test splits
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            warmup_ratio: Proportion of steps for learning rate warmup
            weight_decay: L2 regularization strength
            save_steps: Steps between model checkpoints
            eval_steps: Steps between evaluations
            logging_steps: Steps between logging
            early_stopping_patience: Patience for early stopping
            load_best_model: Whether to load best model at end
            fp16: Enable mixed precision training (auto-detect if None)
            dataloader_num_workers: Number of dataloader workers
        
        Returns:
            Dictionary with training results and metrics
        """
        self.setup_tokenizer()
        self.setup_model()
        if resume_from_checkpoint is None:
            should_resume, checkpoint_path = self.checkpoint_manager.should_resume_training()
            if should_resume:
                resume_from_checkpoint = checkpoint_path
        if fp16 is None:
            if torch.cuda.is_available():
                fp16 = torch.cuda.get_device_capability()[0] >= 7
            else:
                fp16 = False
        logger.info('[LOSS] Using standard MSE Loss')
        training_args = TrainingArguments(output_dir=str(self.output_dir), learning_rate=learning_rate, num_train_epochs=num_epochs, per_device_train_batch_size=batch_size, per_device_eval_batch_size=eval_batch_size, warmup_ratio=warmup_ratio, weight_decay=weight_decay, logging_steps=logging_steps, eval_steps=eval_steps, save_steps=save_steps, eval_strategy='steps', save_strategy='steps', save_total_limit=3, metric_for_best_model='eval_loss', greater_is_better=False, report_to='none', push_to_hub=False, fp16=fp16, dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=True, seed=self.seed, data_seed=self.seed)
        self.trainer = Trainer(model=self.model, args=training_args, train_dataset=datasets['train'], eval_dataset=datasets['validation'], tokenizer=self.tokenizer, data_collator=self.data_collator, compute_metrics=self.compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])
        self.checkpoint_manager.set_trainer(self.trainer)
        logger.info('[TRAINING] Starting model training')
        logger.info(f'  Model: {self.model_name}')
        logger.info(f'  Device: {training_args.device}')
        logger.info(f'  Learning rate: {learning_rate}')
        logger.info(f'  Epochs: {num_epochs}')
        logger.info(f'  Batch size: {batch_size} (train) / {eval_batch_size} (eval)')
        logger.info(f'  Warmup ratio: {warmup_ratio}')
        logger.info(f'  Weight decay: {weight_decay}')
        if resume_from_checkpoint:
            logger.info(f'[TRAINING] Resuming from checkpoint: {resume_from_checkpoint}')
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = self.trainer.train()
        if self.checkpoint_manager.training_interrupted:
            logger.info('[TRAINING] Training was interrupted - checkpoint saved')
            return {'status': 'interrupted', 'message': 'Training interrupted by user - checkpoint saved', 'checkpoint_path': str(self.output_dir)}
        logger.info('[TRAINING] Saving model...')
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info('[EVALUATION] Evaluating on test set...')
        test_results = self.trainer.evaluate(datasets['test'], metric_key_prefix='test')
        results = {'model_name': self.model_name, 'task_type': 'hallucination_prediction_regression', 'num_labels': self.num_labels, 'training_args': training_args.to_dict(), 'train_results': train_result.metrics, 'test_results': test_results, 'dataset_sizes': {'train': len(datasets['train']), 'validation': len(datasets['validation']), 'test': len(datasets['test'])}, 'training_time': train_result.metrics.get('train_runtime', 0), 'trained_at': datetime.now().isoformat(), 'seed': self.seed}
        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info('[TRAINING] Training completed successfully!')
        logger.info(f"  Test MSE: {test_results.get('test_mse', 0):.4f}")
        logger.info(f"  Test R²: {test_results.get('test_r2_score', 0):.4f}")
        logger.info(f"  Test Interval Accuracy: {test_results.get('test_interval_accuracy', 0):.4f}")
        logger.info(f"  Training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
        logger.info(f'  Results saved: {results_path}')
        return results

    def evaluate_model(self, datasets: DatasetDict) -> Dict:
        """Evaluate trained model on all splits."""
        if self.trainer is None:
            raise ValueError('Model must be trained first')
        results = {}
        for split_name, dataset in datasets.items():
            logger.info(f'[EVALUATION] Evaluating on {split_name} set...')
            eval_results = self.trainer.evaluate(dataset, metric_key_prefix=split_name)
            results[split_name] = eval_results
        return results

def train_hallucination_predictor(model_name: str='microsoft/deberta-v3-base', learning_rate: float=2e-05, num_epochs: int=4, batch_size: int=16, eval_batch_size: int=32, warmup_ratio: float=0.1, weight_decay: float=0.01, early_stopping_patience: int=3, save_steps: int=500, eval_steps: int=250, logging_steps: int=50, resume_from_checkpoint: bool=True, **kwargs) -> Dict:
    """
    Convenience function to train hallucination predictor.
    
    Alternative models to try:
    - microsoft/deberta-v3-base: Best performance (recommended)
    - microsoft/deberta-v3-small: Faster, good balance
    - roberta-base: Excellent general performance
    - distilbert-base-uncased: Fastest training
    - bert-base-uncased: Reliable baseline
    """
    init_kwargs = {k: v for k, v in kwargs.items() if k in ['output_dir', 'num_labels', 'max_length', 'cache_dir', 'seed']}
    dataset_kwargs = {k: v for k, v in kwargs.items() if k in ['train_size', 'val_size', 'test_size']}
    excluded_params = {'output_dir', 'fp16', 'adam_beta2', 'optim', 'max_length', 'cache_dir', 'dataloader_pin_memory', 'val_size', 'adam_beta1', 'resume_from_checkpoint', 'train_size', 'num_labels', 'dataloader_num_workers', 'seed', 'test_size'}
    train_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}
    predictor = HallucinationPredictor(model_name=model_name, resume_from_checkpoint=resume_from_checkpoint, **init_kwargs)
    logger.info('Preparing datasets...')
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
        parser = setup_argument_parser('Hallucination Predictor Training')
        args = parser.parse_args()
        logger.info('======================================================================')
        logger.info('HALLUCINATION PREDICTOR TRAINING')
        logger.info('======================================================================')
        log_device_info()
        base_config = {'model_name': 'microsoft/deberta-v3-base', 'num_labels': 1, 'learning_rate': 2e-05}
        if args.test:
            logger.info('🧪 TESTING MODE: Using lower values for quick testing')
            mode_config = get_testing_config()
        else:
            logger.info('🚀 PRODUCTION MODE: Using production training configuration')
            mode_config = get_production_config()
        config = merge_configs(base_config, mode_config, args)
        results = train_hallucination_predictor(**config)
        logger.info('======================================================================')
        logger.info('TRAINING COMPLETED SUCCESSFULLY')
        logger.info('======================================================================')
        logger.info(f"Final Test MSE: {results['test_results'].get('test_mse', 0):.4f}")
        logger.info(f"Final Test R²: {results['test_results'].get('test_r2_score', 0):.4f}")
        logger.info(f"Final Test Interval Accuracy: {results['test_results'].get('test_interval_accuracy', 0):.4f}")
        return results
    except Exception as e:
        logger.error(f'Training failed: {e}')
        return None


if __name__ == '__main__':
    try:
        main()
        logger.info(f'Log file saved: {log_file}')
    finally:
        cleanup_logging_and_exit()