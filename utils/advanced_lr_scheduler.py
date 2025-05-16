"""
Advanced learning rate schedulers for T2ITrainer.
Optimized for multi-GPU training on RTX 4090 GPUs.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Optimizer
from typing import List, Dict, Optional, Union, Callable, Any, Tuple
import warnings
import numpy as np

class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: Target learning rate = base lr * multiplier if multiplier > 1.0.
                    If multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: Target learning rate is reached at total_epoch.
        after_scheduler: After target_epoch, use this scheduler.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of steps
        min_lr: Minimum learning rate
        last_epoch: Last epoch (-1 for initialization)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Ensure we don't go beyond 1.0
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]


class WarmupLinearScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear decay scheduler with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total number of steps
        min_lr: Minimum learning rate
        last_epoch: Last epoch (-1 for initialization)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super(WarmupLinearScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Ensure we don't go beyond 1.0
            return [self.min_lr + (base_lr - self.min_lr) * (1.0 - progress) for base_lr in self.base_lrs]


class WarmupConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Constant learning rate after warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch (-1 for initialization)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super(WarmupConstantScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Constant learning rate
            return self.base_lrs


class StepLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """
    Step LR with warmup.
    
    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        step_size: Period of learning rate decay (in steps)
        gamma: Multiplicative factor of learning rate decay
        last_epoch: Last epoch (-1 for initialization)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        super(StepLRWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Step decay
            steps_after_warmup = self.last_epoch - self.warmup_steps
            decay_factor = self.gamma ** (steps_after_warmup // self.step_size)
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class MultiStageScheduler:
    """
    Multi-stage learning rate scheduler.
    Allows different learning rate schedulers at different training stages.
    
    Args:
        optimizer: Optimizer
        schedulers: List of scheduler tuples (scheduler, duration)
        last_epoch: Last epoch (-1 for initialization)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: List[Tuple[torch.optim.lr_scheduler._LRScheduler, int]],
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.current_scheduler_idx = 0
        self.current_epoch = 0 if last_epoch == -1 else last_epoch
        self.last_lr = None
        
        # Set up first scheduler
        self.current_scheduler, self.current_duration = self.schedulers[0]
        
        # Track stage transitions
        self.stage_starts = [0]
        total = 0
        for _, duration in schedulers[:-1]:
            total += duration
            self.stage_starts.append(total)
            
        # Initialize with the right epoch
        if last_epoch > 0:
            for _ in range(last_epoch):
                self.step()
    
    def get_last_lr(self):
        if self.last_lr is None:
            return self.current_scheduler.get_last_lr()
        return self.last_lr
    
    def step(self):
        # Check if we need to switch to the next scheduler
        stage_end = self.stage_starts[self.current_scheduler_idx] + self.current_duration
        
        if self.current_epoch >= stage_end and self.current_scheduler_idx < len(self.schedulers) - 1:
            # Move to next stage
            self.current_scheduler_idx += 1
            self.current_scheduler, self.current_duration = self.schedulers[self.current_scheduler_idx]
            
            # Initialize new scheduler with the right base lr
            self.current_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        # Step the current scheduler
        if hasattr(self.current_scheduler, 'step'):
            self.current_scheduler.step()
        
        # Update last_lr
        self.last_lr = self.current_scheduler.get_last_lr()
        
        # Increment epoch
        self.current_epoch += 1
        
        return self.last_lr


class ProgressiveBatchScheduler:
    """
    Progressive batch size scheduler that changes the batch size during training.
    Can work with any learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        base_scheduler: Base learning rate scheduler
        batch_schedule: List of (batch_size, step) tuples
        adjust_lr: Whether to adjust learning rate proportionally to batch size
        dataloader_update_fn: Function to update dataloader batch size
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler,
        batch_schedule: List[Tuple[int, int]],
        adjust_lr: bool = True,
        dataloader_update_fn: Optional[Callable[[int], None]] = None
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.batch_schedule = sorted(batch_schedule, key=lambda x: x[1])
        self.adjust_lr = adjust_lr
        self.dataloader_update_fn = dataloader_update_fn
        
        self.current_step = 0
        self.current_batch_idx = 0
        self.initial_batch_size = batch_schedule[0][0]
        self.current_batch_size = self.initial_batch_size
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def get_last_lr(self):
        return self.base_scheduler.get_last_lr()
    
    def step(self):
        # Check if we need to update batch size
        if self.current_batch_idx < len(self.batch_schedule) - 1:
            next_batch_size, next_step = self.batch_schedule[self.current_batch_idx + 1]
            
            if self.current_step >= next_step:
                # Update batch size
                self.current_batch_idx += 1
                prev_batch_size = self.current_batch_size
                self.current_batch_size = next_batch_size
                
                # Adjust learning rate if needed
                if self.adjust_lr:
                    batch_ratio = self.current_batch_size / prev_batch_size
                    for i, group in enumerate(self.optimizer.param_groups):
                        group['lr'] = group['lr'] * batch_ratio
                
                # Update dataloader batch size if function provided
                if self.dataloader_update_fn is not None:
                    self.dataloader_update_fn(self.current_batch_size)
                
                print(f"Updating batch size to {self.current_batch_size} at step {self.current_step}")
        
        # Step the base scheduler
        self.base_scheduler.step()
        
        # Increment step
        self.current_step += 1
        
        return self.get_last_lr()


class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler that adjusts learning rate based on gradient statistics.
    
    Args:
        optimizer: Optimizer
        base_scheduler: Base learning rate scheduler
        model: Model to track gradients
        update_interval: How often to update learning rate (in steps)
        scaling_factor: How much to scale learning rate based on gradient statistics
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        smoothing: Exponential moving average smoothing factor
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler,
        model: torch.nn.Module,
        update_interval: int = 100,
        scaling_factor: float = 0.1,
        min_lr: float = 1e-7,
        max_lr: Optional[float] = None,
        smoothing: float = 0.9
    ):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.model = model
        self.update_interval = update_interval
        self.scaling_factor = scaling_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smoothing = smoothing
        
        self.current_step = 0
        self.grad_norms_ema = None
    
    def get_last_lr(self):
        return self.base_scheduler.get_last_lr()
    
    def step(self):
        # Step the base scheduler
        self.base_scheduler.step()
        
        # Collect gradient statistics
        grad_norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            
            # Update EMA
            if self.grad_norms_ema is None:
                self.grad_norms_ema = avg_grad_norm
            else:
                self.grad_norms_ema = self.smoothing * self.grad_norms_ema + (1 - self.smoothing) * avg_grad_norm
            
            # Adjust learning rate based on gradient statistics
            if self.current_step > 0 and self.current_step % self.update_interval == 0:
                for i, group in enumerate(self.optimizer.param_groups):
                    # Compute adjustment factor
                    factor = 1.0
                    if self.grad_norms_ema > 10.0:
                        # Gradients are large, reduce learning rate
                        factor = 1.0 - self.scaling_factor
                    elif self.grad_norms_ema < 0.1:
                        # Gradients are small, increase learning rate
                        factor = 1.0 + self.scaling_factor
                    
                    # Apply adjustment
                    new_lr = group['lr'] * factor
                    
                    # Clip to min/max if specified
                    if self.min_lr is not None:
                        new_lr = max(new_lr, self.min_lr)
                    if self.max_lr is not None:
                        new_lr = min(new_lr, self.max_lr)
                    
                    group['lr'] = new_lr
        
        # Increment step
        self.current_step += 1
        
        return self.get_last_lr()


class CustomSNRWeightedLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler that scales learning rate based on SNR of diffusion process.
    Higher learning rates for high SNR (low noise) and lower learning rates for low SNR (high noise).
    
    Args:
        optimizer: Optimizer
        snr_gamma: SNR gamma value for scaling
        max_steps: Total number of steps
        last_epoch: Last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        snr_gamma: float = 5.0,
        max_steps: int = 1000,
        last_epoch: int = -1
    ):
        self.snr_gamma = snr_gamma
        self.max_steps = max_steps
        super(CustomSNRWeightedLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Simulate SNR weights based on training progress
        progress = self.last_epoch / self.max_steps
        # Higher weight for early steps (high SNR)
        weight = (1.0 - progress) ** self.snr_gamma
        
        return [base_lr * weight for base_lr in self.base_lrs]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: float = 0.5,
    power: float = 1.0,
    last_epoch: int = -1,
    min_lr: float = 0.0,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler based on the scheduler name.
    
    Args:
        name: Name of the scheduler
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine scheduler
        power: Power factor for polynomial scheduler
        last_epoch: Last epoch
        min_lr: Minimum learning rate
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Learning rate scheduler
    """
    name = name.lower()
    
    if name == "linear":
        return WarmupLinearScheduler(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            max_steps=num_training_steps,
            min_lr=min_lr,
            last_epoch=last_epoch,
        )
    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=num_training_steps - (num_warmup_steps or 0),
            eta_min=min_lr,
            last_epoch=last_epoch
        )
    elif name == "cosine_with_restarts":
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=(num_training_steps - (num_warmup_steps or 0)) // int(1 / num_cycles),
            eta_min=min_lr,
            last_epoch=last_epoch
        )
    elif name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            max_steps=num_training_steps,
            min_lr=min_lr,
            last_epoch=last_epoch,
        )
    elif name == "constant":
        return WarmupConstantScheduler(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            last_epoch=last_epoch,
        )
    elif name == "step":
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.1)
        return StepLRWithWarmup(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            step_size=step_size,
            gamma=gamma,
            last_epoch=last_epoch,
        )
    elif name == "adaptive":
        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model must be provided for adaptive scheduler")
        
        update_interval = kwargs.get("update_interval", 100)
        scaling_factor = kwargs.get("scaling_factor", 0.1)
        
        base_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            max_steps=num_training_steps,
            min_lr=min_lr,
            last_epoch=last_epoch,
        )
        
        return AdaptiveLearningRateScheduler(
            optimizer=optimizer,
            base_scheduler=base_scheduler,
            model=model,
            update_interval=update_interval,
            scaling_factor=scaling_factor,
            min_lr=min_lr,
            max_lr=kwargs.get("max_lr"),
            smoothing=kwargs.get("smoothing", 0.9),
        )
    elif name == "progressive_batching":
        batch_schedule = kwargs.get("batch_schedule")
        if batch_schedule is None:
            raise ValueError("batch_schedule must be provided for progressive_batching scheduler")
        
        base_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=num_warmup_steps or 0,
            max_steps=num_training_steps,
            min_lr=min_lr,
            last_epoch=last_epoch,
        )
        
        return ProgressiveBatchScheduler(
            optimizer=optimizer,
            base_scheduler=base_scheduler,
            batch_schedule=batch_schedule,
            adjust_lr=kwargs.get("adjust_lr", True),
            dataloader_update_fn=kwargs.get("dataloader_update_fn"),
        )
    elif name == "snr_weighted":
        snr_gamma = kwargs.get("snr_gamma", 5.0)
        
        return CustomSNRWeightedLR(
            optimizer=optimizer,
            snr_gamma=snr_gamma,
            max_steps=num_training_steps,
            last_epoch=last_epoch,
        )
    elif name == "multi_stage":
        schedulers = kwargs.get("schedulers")
        if schedulers is None:
            raise ValueError("schedulers must be provided for multi_stage scheduler")
        
        return MultiStageScheduler(
            optimizer=optimizer,
            schedulers=schedulers,
            last_epoch=last_epoch,
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}") 