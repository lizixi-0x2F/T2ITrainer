"""
Distributed evaluation utilities for diffusion models.
Optimized for RTX 4090 GPUs.
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import gc
import time
from tqdm import tqdm

def is_distributed() -> bool:
    """Check if code is running in a distributed environment."""
    return dist.is_initialized() and dist.get_world_size() > 1

def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    """Get number of processes in the distributed group."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0

def synchronize():
    """Synchronize all processes."""
    if not is_distributed():
        return
    dist.barrier()

def all_gather_object(obj: Any) -> List[Any]:
    """
    Gather objects from all processes.
    
    Args:
        obj: Object to gather
        
    Returns:
        List of objects from all processes
    """
    if not is_distributed():
        return [obj]
    
    gathered_objects = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered_objects, obj)
    return gathered_objects

def reduce_dict(input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Reduce dictionary of tensors from all processes.
    
    Args:
        input_dict: Dictionary of tensors to reduce
        
    Returns:
        Reduced dictionary on rank 0, empty dictionary on other ranks
    """
    if not is_distributed():
        return input_dict
    
    # Gather keys and values
    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys]
    
    # Reduce the values
    values = torch.stack(values)
    dist.reduce(values, dst=0)
    
    # Create the reduced dictionary on rank 0
    if is_main_process():
        reduced_dict = {k: v for k, v in zip(keys, values)}
        return reduced_dict
    else:
        return {}

class DistributedEvaluator:
    """
    Utility class for distributed evaluation of diffusion models.
    
    Args:
        model: The model to evaluate
        device: The device to use for evaluation
        is_fsdp: Whether the model is wrapped with FSDP
    """
    
    def __init__(
        self,
        model: Union[torch.nn.Module, DDP, FSDP],
        device: torch.device = None,
        is_fsdp: bool = False
    ):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.is_fsdp = is_fsdp
        self.is_distributed = is_distributed()
        self.rank = get_rank()
        self.world_size = get_world_size()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        metrics: Dict[str, Callable] = None,
        postprocess_fn: Callable = None,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model using the given dataloader and metrics.
        
        Args:
            dataloader: DataLoader for evaluation
            metrics: Dictionary of metric functions
            postprocess_fn: Function to apply to model outputs
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Store outputs and targets
        all_outputs = []
        all_targets = []
        
        # Create progress bar if on main process
        progress_bar = tqdm(total=len(dataloader), disable=not is_main_process())
        
        # Evaluate
        for batch_idx, batch in enumerate(dataloader):
            if max_samples is not None and batch_idx * len(batch) >= max_samples:
                break
            
            # Move batch to device
            if isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            if self.is_fsdp:
                with FSDP.summon_full_params(self.model, writeback=False):
                    outputs = self.model(batch)
            else:
                outputs = self.model(batch)
            
            # Apply postprocessing if provided
            if postprocess_fn is not None:
                outputs, targets = postprocess_fn(outputs, batch)
            else:
                targets = batch
            
            # Collect outputs and targets
            all_outputs.append(outputs)
            all_targets.append(targets)
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Gather all outputs and targets
        if self.is_distributed:
            # Wait for all processes to finish evaluation
            synchronize()
            
            # Gather outputs and targets from all processes
            all_outputs = all_gather_object(all_outputs)
            all_targets = all_gather_object(all_targets)
            
            # Flatten gathered lists
            if is_main_process():
                all_outputs = [item for sublist in all_outputs for item in sublist]
                all_targets = [item for sublist in all_targets for item in sublist]
        
        # Compute metrics on main process
        metric_results = {}
        if is_main_process():
            for metric_name, metric_fn in metrics.items():
                metric_results[metric_name] = metric_fn(all_outputs, all_targets)
        
        # Broadcast metric results to all processes
        if self.is_distributed:
            metric_results = all_gather_object(metric_results)[0]
        
        return metric_results

def distributed_image_generation(
    pipeline,
    prompts: List[str],
    num_inference_steps: int = 50,
    batch_size: int = 4,
    guidance_scale: float = 7.5,
    **kwargs
) -> Tuple[List[torch.Tensor], float]:
    """
    Generate images in a distributed manner for faster evaluation.
    
    Args:
        pipeline: The diffusion pipeline to use for generation
        prompts: List of prompts to generate images for
        num_inference_steps: Number of inference steps
        batch_size: Batch size for generation
        guidance_scale: Guidance scale for generation
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        Tuple of list of generated images and generation time
    """
    # Distribute prompts across processes
    rank = get_rank()
    world_size = get_world_size()
    
    if world_size > 1:
        # Calculate how many prompts each process should handle
        prompts_per_process = len(prompts) // world_size
        remainder = len(prompts) % world_size
        
        start_idx = rank * prompts_per_process + min(rank, remainder)
        end_idx = start_idx + prompts_per_process + (1 if rank < remainder else 0)
        
        # Get prompts for this process
        process_prompts = prompts[start_idx:end_idx]
    else:
        process_prompts = prompts
    
    # Generate images
    start_time = time.time()
    
    all_images = []
    for i in range(0, len(process_prompts), batch_size):
        batch_prompts = process_prompts[i:i + batch_size]
        
        # Generate
        with torch.no_grad():
            outputs = pipeline(
                batch_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )
        
        # Collect images
        if hasattr(outputs, "images"):
            images = outputs.images
        else:
            images = outputs
        
        all_images.extend(images)
    
    generation_time = time.time() - start_time
    
    # Gather images from all processes
    if world_size > 1:
        all_gathered_images = all_gather_object(all_images)
        
        if is_main_process():
            # Flatten gathered lists and reorder
            all_images = []
            for i in range(len(prompts)):
                process_idx = min(i // prompts_per_process, world_size - 1)
                if process_idx < remainder:
                    local_idx = i - process_idx * prompts_per_process
                else:
                    offset = remainder * (prompts_per_process + 1)
                    remaining = i - offset
                    process_idx = remainder + remaining // prompts_per_process
                    local_idx = remaining % prompts_per_process
                
                if process_idx < len(all_gathered_images) and local_idx < len(all_gathered_images[process_idx]):
                    all_images.append(all_gathered_images[process_idx][local_idx])
        
        # Reduce generation time (average)
        all_times = all_gather_object(generation_time)
        generation_time = sum(all_times) / len(all_times)
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_images, generation_time

def evaluate_fid_distributed(
    real_images_dir: str,
    generated_images: List[torch.Tensor],
    batch_size: int = 32,
    device: torch.device = None
) -> float:
    """
    Compute FID score in a distributed manner.
    
    Args:
        real_images_dir: Directory containing real images
        generated_images: List of generated images
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        FID score
    """
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("torch-fidelity is required for FID computation. Install with: pip install torch-fidelity")
    
    # Use CPU for feature extraction in distributed setting to avoid CUDA errors
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Only compute FID on main process
    if not is_main_process():
        synchronize()
        return 0.0
    
    # Save generated images to temporary directory
    import tempfile
    import os
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save generated images
        for i, img in enumerate(generated_images):
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:  # CHW to HWC
                    img = img.transpose(1, 2, 0)
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                continue
            
            img.save(os.path.join(tmpdir, f"img_{i:05d}.png"))
        
        # Compute FID
        fid = torch_fidelity.calculate_metrics(
            input1=tmpdir,
            input2=real_images_dir,
            batch_size=batch_size,
            device=device,
            metrics=['fid']
        )['fid']
    
    # Broadcast FID to all processes
    if is_distributed():
        fid = all_gather_object(fid)[0]
    
    return fid

def distributed_model_evaluation(
    model, test_data, metric_fn, batch_size=16, device=None
):
    """
    Evaluates a model in a distributed fashion.
    
    Args:
        model: Model to evaluate
        test_data: Test dataset
        metric_fn: Function to compute metrics
        batch_size: Batch size
        device: Device to use
    
    Returns:
        Evaluation results
    """
    # Set up device
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    # Set up distributed sampler
    if is_distributed():
        sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    else:
        sampler = None
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Create evaluator
    evaluator = DistributedEvaluator(
        model=model,
        device=device,
        is_fsdp=isinstance(model, FSDP)
    )
    
    # Evaluate model
    results = evaluator.evaluate(
        dataloader=dataloader,
        metrics={"metric": metric_fn}
    )
    
    return results 