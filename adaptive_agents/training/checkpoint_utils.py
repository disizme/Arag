"""
Checkpoint Utilities for Training Scripts

Reusable utilities for handling training checkpoints, graceful shutdown,
and automatic resume functionality across different training scripts.
"""

import os
import signal
import glob
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints with automatic resume and graceful shutdown.
    
    Features:
    - Automatic checkpoint detection and resume
    - Graceful shutdown with signal handling
    - Checkpoint validation
    - Training completion detection
    """
    
    def __init__(
        self,
        output_dir: Path,
        resume_from_checkpoint: bool = True,
        auto_signal_handling: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory where checkpoints are saved
            resume_from_checkpoint: Whether to enable automatic resume
            auto_signal_handling: Whether to setup signal handlers
        """
        self.output_dir = Path(output_dir)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.training_interrupted = False
        self.trainer = None
        
        # Setup signal handlers for graceful shutdown
        if auto_signal_handling:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"[CHECKPOINT-MANAGER] Initialized")
        logger.info(f"  Output dir: {output_dir}")
    
    def _signal_handler(self, signum, frame=None):
        """Handle training interruption signals gracefully."""
        logger.info(f"\n[INTERRUPT] Received signal {signum}. Saving checkpoint and stopping...")
        self.training_interrupted = True
        if self.trainer:
            try:
                self.trainer.save_model()
                logger.info("[INTERRUPT] Model checkpoint saved successfully")
            except Exception as e:
                logger.error(f"[INTERRUPT] Failed to save checkpoint: {e}")
        
        # Exit gracefully after saving
        logger.info("[INTERRUPT] Exiting...")
        import sys
        sys.exit(0)
    
    def set_trainer(self, trainer):
        """Set the trainer instance for checkpoint saving during interruption."""
        self.trainer = trainer
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the latest valid checkpoint in the output directory.
        
        Returns:
            Path to latest checkpoint directory, or None if no valid checkpoint found
        """
        if not self.output_dir.exists():
            return None
        
        # Look for checkpoint directories
        checkpoint_pattern = str(self.output_dir / "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            return None
        
        # Sort by checkpoint number (extract number from checkpoint-XXXX)
        def get_checkpoint_number(path):
            try:
                return int(Path(path).name.split('-')[-1])
            except (ValueError, IndexError):
                return 0
        
        latest_checkpoint = max(checkpoints, key=get_checkpoint_number)
        
        # Verify checkpoint is valid
        checkpoint_path = Path(latest_checkpoint)
        required_files = ["config.json"]
        
        # Check for either pytorch_model.bin or model.safetensors
        has_model_file = (
            (checkpoint_path / "pytorch_model.bin").exists() or
            (checkpoint_path / "model.safetensors").exists()
        )
        
        # Verify all required files exist
        valid_checkpoint = (
            has_model_file and 
            all((checkpoint_path / f).exists() for f in required_files)
        )
        
        if valid_checkpoint:
            logger.info(f"[CHECKPOINT] Found valid checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
        else:
            logger.warning(f"[CHECKPOINT] Invalid checkpoint found (missing files): {checkpoint_path}")
            return None
    
    def is_training_completed(self) -> bool:
        """
        Check if training was already completed.
        
        Returns:
            True if training_results.json exists, indicating completed training
        """
        results_file = self.output_dir / "training_results.json"
        return results_file.exists()
    
    def should_resume_training(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if training should be resumed from a checkpoint.
        
        Returns:
            Tuple of (should_resume, checkpoint_path)
        """
        if not self.resume_from_checkpoint:
            logger.info("[CHECKPOINT] Resume disabled, starting fresh training")
            return False, None
        
        # Check if training was already completed
        if self.is_training_completed():
            logger.info("[CHECKPOINT] Training appears to be completed (results file exists)")
            
            # Ask user what they want to do
            logger.info("[CHECKPOINT] Options:")
            logger.info("  - Delete 'training_results.json' to restart training")
            logger.info("  - Or disable resume_from_checkpoint parameter")
            return False, None
        
        # Look for latest checkpoint
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            checkpoint_step = Path(latest_checkpoint).name.split('-')[-1]
            logger.info(f"[CHECKPOINT] Resuming training from step {checkpoint_step}")
            return True, latest_checkpoint
        
        logger.info("[CHECKPOINT] No valid checkpoint found, starting fresh training")
        return False, None
    
    def get_checkpoint_info(self, checkpoint_path: str) -> dict:
        """
        Get information about a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_dir = Path(checkpoint_path)
        
        info = {
            "path": checkpoint_path,
            "step": checkpoint_dir.name.split('-')[-1],
            "exists": checkpoint_dir.exists(),
            "files": []
        }
        
        if checkpoint_dir.exists():
            info["files"] = [f.name for f in checkpoint_dir.iterdir() if f.is_file()]
        
        return info
    
    def clean_old_checkpoints(self, keep_latest: int = 3):
        """
        Clean old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest checkpoints to keep
        """
        if not self.output_dir.exists():
            return
        
        checkpoint_pattern = str(self.output_dir / "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) <= keep_latest:
            return
        
        # Sort by checkpoint number
        def get_checkpoint_number(path):
            try:
                return int(Path(path).name.split('-')[-1])
            except (ValueError, IndexError):
                return 0
        
        checkpoints_sorted = sorted(checkpoints, key=get_checkpoint_number, reverse=True)
        checkpoints_to_remove = checkpoints_sorted[keep_latest:]
        
        for checkpoint_path in checkpoints_to_remove:
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"[CHECKPOINT] Removed old checkpoint: {Path(checkpoint_path).name}")
            except Exception as e:
                logger.warning(f"[CHECKPOINT] Failed to remove {checkpoint_path}: {e}")
    
    def reset_training(self):
        """
        Reset training by removing all checkpoints and results.
        Use with caution!
        """
        if not self.output_dir.exists():
            return
        
        # Remove checkpoints
        checkpoint_pattern = str(self.output_dir / "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        for checkpoint_path in checkpoints:
            try:
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"[RESET] Removed checkpoint: {Path(checkpoint_path).name}")
            except Exception as e:
                logger.warning(f"[RESET] Failed to remove {checkpoint_path}: {e}")
        
        # Remove results file
        results_file = self.output_dir / "training_results.json"
        if results_file.exists():
            try:
                results_file.unlink()
                logger.info("[RESET] Removed training results file")
            except Exception as e:
                logger.warning(f"[RESET] Failed to remove results file: {e}")
        
        logger.info("[RESET] Training reset completed")


def create_checkpoint_manager(output_dir: Path, **kwargs) -> CheckpointManager:
    """
    Factory function to create a CheckpointManager instance.
    
    Args:
        output_dir: Output directory for checkpoints
        **kwargs: Additional arguments for CheckpointManager
        
    Returns:
        CheckpointManager instance
    """
    return CheckpointManager(output_dir, **kwargs)