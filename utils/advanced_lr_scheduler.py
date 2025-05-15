"""
Advanced learning rate scheduling utilities.
Optimized for diffusion model training on RTX 4090 GPUs.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, LambdaLR
import numpy as np
from typing import List, Dict, Optional, Union, Callable


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm up the learning rate from 0 to base_lr over multiple epochs.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of epochs for warmup
        after_scheduler: Scheduler to use after warmup
        last_epoch: The index of last epoch
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(self, 
                 optimizer, 
                 warmup_epochs: int, 
                 after_scheduler, 
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = [base_lr for base_lr in self.base_lrs]
                self.finished = True
            return self.after_scheduler.get_lr()
        
        return [base_lr * ((self.last_epoch + 1) / self.warmup_epochs) for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if self.finished and epoch is None:
            if epoch is None:
                self.after_scheduler.step(None)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
                self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    
    Args:
        optimizer: Optimizer
        first_cycle_steps: Number of steps in the first cycle
        cycle_mult: Multiplier for cycle length after each restart
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate
        warmup_steps: Number of warmup steps in each cycle
        gamma: Decay factor for max_lr after each restart
        last_epoch: The index of last epoch
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.0,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.0,
                 last_epoch: int = -1,
                 verbose: bool = False):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cycle = 0
        self.cycle_steps = first_cycle_steps
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch, verbose)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            # Linear warmup
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / 
                                   (self.cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cycle_steps = int(self.cycle_steps * self.cycle_mult)
                self.max_lr = self.max_lr * self.gamma
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cycle_steps = int(self.first_cycle_steps * self.cycle_mult ** (n))
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
            else:
                self.cycle = 0
                self.step_in_cycle = epoch
                self.cycle_steps = self.first_cycle_steps
        
        self.last_epoch = math.floor(epoch)
        self._last_lr = self.get_lr()
        
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr


class ChainedScheduler(_LRScheduler):
    """
    Chain multiple schedulers in sequence for multi-phase training.
    
    Args:
        optimizer: Optimizer
        schedulers: List of scheduler configurations
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 schedulers: List[Dict],
                 verbose: bool = False):
        self.optimizer = optimizer
        self.schedulers = []
        self.phase_steps = []
        self.current_phase = 0
        self.current_step = 0
        
        total_steps = 0
        for scheduler_config in schedulers:
            scheduler_type = scheduler_config.pop('type')
            steps = scheduler_config.pop('steps')
            
            # Create the scheduler
            if scheduler_type == 'constant':
                scheduler = LambdaLR(optimizer, lambda _: 1.0)
            elif scheduler_type == 'linear':
                start_factor = scheduler_config.pop('start_factor', 1.0)
                end_factor = scheduler_config.pop('end_factor', 0.0)
                scheduler = LambdaLR(
                    optimizer,
                    lambda step: start_factor + (end_factor - start_factor) * step / steps
                )
            elif scheduler_type == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=steps, **scheduler_config)
            elif scheduler_type == 'warmup_cosine':
                warmup_steps = scheduler_config.pop('warmup_steps')
                scheduler = CosineAnnealingWarmupRestarts(
                    optimizer,
                    first_cycle_steps=steps,
                    warmup_steps=warmup_steps,
                    **scheduler_config
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
            self.schedulers.append(scheduler)
            self.phase_steps.append(steps)
            total_steps += steps
        
        self.total_steps = total_steps
        super(ChainedScheduler, self).__init__(optimizer, -1, verbose)
    
    def get_lr(self):
        # Use the current phase's scheduler
        return self.schedulers[self.current_phase].get_lr()
    
    def step(self, epoch=None):
        if epoch is not None:
            raise ValueError("ChainedScheduler doesn't support epoch-based stepping")
        
        # Increment step and check if we need to move to the next phase
        self.current_step += 1
        steps_in_current_phase = sum(self.phase_steps[:self.current_phase])
        
        if self.current_step - steps_in_current_phase >= self.phase_steps[self.current_phase]:
            # Move to the next phase
            self.current_phase += 1
            if self.current_phase >= len(self.schedulers):
                # We've completed all phases, stay at the last LR
                self.current_phase = len(self.schedulers) - 1
            
            # Reset the current scheduler
            steps_in_current_phase = sum(self.phase_steps[:self.current_phase])
        
        # Step the current scheduler
        for _ in range(self.current_step - steps_in_current_phase):
            self.schedulers[self.current_phase].step()
        
        # Update _last_lr
        self._last_lr = self.get_lr()
        return self._last_lr


class ProgressiveLRScheduler(_LRScheduler):
    """
    Progressive learning rate scheduler that integrates with progressive batch size.
    The learning rate is scaled based on the square root of the batch size ratio.
    
    Args:
        optimizer: Optimizer
        batch_schedules: List of (batch_size, steps) tuples
        lr_base: Base learning rate
        last_epoch: The index of last epoch
        scale_mode: How to scale the learning rate ('sqrt' or 'linear')
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 batch_schedules: List[tuple], 
                 lr_base: float,
                 last_epoch: int = -1,
                 scale_mode: str = 'sqrt',
                 verbose: bool = False):
        self.optimizer = optimizer
        self.batch_schedules = batch_schedules
        self.lr_base = lr_base
        self.scale_mode = scale_mode
        
        # Calculate phase boundaries
        self.phase_boundaries = []
        step_sum = 0
        for _, steps in batch_schedules:
            step_sum += steps
            self.phase_boundaries.append(step_sum)
        
        self.current_phase = 0
        self.total_steps = sum(steps for _, steps in batch_schedules)
        
        super(ProgressiveLRScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        # Determine current phase
        current_step = self.last_epoch
        for i, boundary in enumerate(self.phase_boundaries):
            if current_step < boundary:
                self.current_phase = i
                break
        else:
            self.current_phase = len(self.batch_schedules) - 1
        
        # Get current batch size
        batch_size, _ = self.batch_schedules[self.current_phase]
        base_batch_size, _ = self.batch_schedules[0]
        
        # Scale learning rate based on batch size
        if self.scale_mode == 'sqrt':
            scale = math.sqrt(batch_size / base_batch_size)
        else:  # linear
            scale = batch_size / base_batch_size
        
        return [self.lr_base * scale for _ in self.base_lrs]
    
    def get_current_batch_size(self):
        """Get the current batch size based on the training step."""
        batch_size, _ = self.batch_schedules[self.current_phase]
        return batch_size


class AdaptiveLRScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler that adjusts the learning rate based on the gradient statistics.
    
    Args:
        optimizer: Optimizer
        base_lr: Base learning rate
        grad_stats_window: Window size for gradient statistics
        scale_factor: Factor to scale the learning rate
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
        adjust_every: Adjust the learning rate every N steps
        last_epoch: The index of last epoch
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 grad_stats_window: int = 100,
                 scale_factor: float = 0.1,
                 min_scale: float = 0.1,
                 max_scale: float = 10.0,
                 adjust_every: int = 100,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.grad_stats_window = grad_stats_window
        self.scale_factor = scale_factor
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adjust_every = adjust_every
        
        # Store gradient statistics
        self.grad_norms = []
        self.lr_scales = [1.0]  # Start with no scaling
        
        super(AdaptiveLRScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        return [self.base_lr * self.lr_scales[-1] for _ in self.base_lrs]
    
    def record_grad_norm(self, grad_norm: float):
        """Record the gradient norm for adaptive adjustment."""
        self.grad_norms.append(grad_norm)
        if len(self.grad_norms) > self.grad_stats_window:
            self.grad_norms.pop(0)
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        
        # Adjust learning rate if needed
        if epoch % self.adjust_every == 0 and epoch > 0 and len(self.grad_norms) >= self.grad_stats_window:
            # Compute mean and std of gradient norms
            mean_norm = np.mean(self.grad_norms)
            std_norm = np.std(self.grad_norms)
            
            # Compute coefficient of variation (CV)
            cv = std_norm / mean_norm if mean_norm > 0 else 0.0
            
            # Adjust learning rate based on CV
            if cv > 1.5:  # High variance, reduce learning rate
                new_scale = max(self.lr_scales[-1] * (1 - self.scale_factor), self.min_scale)
            elif cv < 0.1:  # Low variance, increase learning rate
                new_scale = min(self.lr_scales[-1] * (1 + self.scale_factor), self.max_scale)
            else:
                new_scale = self.lr_scales[-1]
            
            self.lr_scales.append(new_scale)
        
        self.last_epoch = epoch
        self._last_lr = self.get_lr()
        
        for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
            param_group['lr'] = lr
        
        return self._last_lr


def create_scheduler_from_config(optimizer, config: Dict, steps: int) -> _LRScheduler:
    """
    Create a learning rate scheduler from a configuration dict.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        steps: Total number of training steps
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'constant')
    
    if scheduler_type == 'constant':
        return LambdaLR(optimizer, lambda _: 1.0)
    
    elif scheduler_type == 'linear':
        start_factor = config.get('start_factor', 1.0)
        end_factor = config.get('end_factor', 0.0)
        return LambdaLR(
            optimizer,
            lambda step: start_factor + (end_factor - start_factor) * step / steps
        )
    
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer, 
            T_max=steps,
            eta_min=config.get('min_lr', 0.0)
        )
    
    elif scheduler_type == 'warmup_cosine':
        warmup_steps = config.get('warmup_steps', int(0.1 * steps))
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=steps,
            warmup_steps=warmup_steps,
            max_lr=config.get('max_lr', optimizer.param_groups[0]['lr']),
            min_lr=config.get('min_lr', 0.0)
        )
    
    elif scheduler_type == 'multi_phase':
        schedulers = config.get('schedulers', [])
        return ChainedScheduler(optimizer, schedulers)
    
    elif scheduler_type == 'progressive':
        batch_schedules = config.get('batch_schedules', [(1, steps)])
        return ProgressiveLRScheduler(
            optimizer,
            batch_schedules=batch_schedules,
            lr_base=config.get('lr_base', optimizer.param_groups[0]['lr']),
            scale_mode=config.get('scale_mode', 'sqrt')
        )
    
    elif scheduler_type == 'adaptive':
        return AdaptiveLRScheduler(
            optimizer,
            base_lr=config.get('base_lr', optimizer.param_groups[0]['lr']),
            grad_stats_window=config.get('grad_stats_window', 100),
            scale_factor=config.get('scale_factor', 0.1),
            min_scale=config.get('min_scale', 0.1),
            max_scale=config.get('max_scale', 10.0),
            adjust_every=config.get('adjust_every', 100)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_rtx4090_optimized_scheduler(optimizer, total_steps: int, batch_size: int) -> _LRScheduler:
    """
    Get a learning rate scheduler optimized for RTX 4090 with typical diffusion model training.
    
    Args:
        optimizer: Optimizer
        total_steps: Total number of training steps
        batch_size: Batch size per GPU
        
    Returns:
        Optimized learning rate scheduler
    """
    # For RTX 4090, a progressive batch size approach works well
    warmup_steps = max(100, int(0.05 * total_steps))
    
    # Create a scheduler with warmup and cosine annealing
    return CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=total_steps,
        warmup_steps=warmup_steps,
        max_lr=optimizer.param_groups[0]['lr'],
        min_lr=1e-6
    ) 