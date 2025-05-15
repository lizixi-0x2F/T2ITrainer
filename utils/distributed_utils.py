"""
Utility functions for distributed training.
Optimized for RTX 4090 GPUs.
"""

import os
import torch
import torch.distributed as dist
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple

def setup_distributed_device(local_rank: int) -> torch.device:
    """
    Set up the device for the current process in distributed training.
    
    Args:
        local_rank: The local rank of the current process.
    
    Returns:
        torch.device: The device to use for this process.
    """
    if torch.cuda.is_available():
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device("cpu")

def setup_distributed_environment(local_rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Set up the distributed environment.
    
    Args:
        local_rank: The local rank of the current process.
        world_size: The total number of processes.
        backend: The backend to use for distributed training.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=local_rank, world_size=world_size)

def cleanup_distributed_environment() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_initialized()

def get_world_size() -> int:
    """Get the world size for distributed training."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return get_rank() == 0

def synchronize() -> None:
    """Synchronize all processes (barrier)."""
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return
    dist.barrier()

def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    Reduce the values in the dictionary from all processes.
    
    Args:
        input_dict: Dictionary with values to reduce.
        average: Whether to average or sum the values.
    
    Returns:
        Dict[str, torch.Tensor]: Reduced dictionary.
    """
    world_size = get_world_size()
    if world_size <= 1:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        
        # Sort the keys for consistency across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        
        if dist.get_rank() == 0 and average:
            values /= world_size
            
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict

def all_gather(data: Any) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data.
    
    Args:
        data: Any picklable object.
    
    Returns:
        List[Any]: List of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size <= 1:
        return [data]
    
    # Serialize the data to a Tensor
    buffer = torch.ByteStorage.from_buffer(torch.pickle.dumps(data))
    tensor = torch.ByteTensor(buffer).to('cuda')
    
    # Obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    # Gather all tensors
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,), device='cuda'))
    
    # Pad if necessary
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,), device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    
    dist.all_gather(tensor_list, tensor)
    
    # Deserialize the gathered tensors
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(torch.pickle.loads(buffer))
    
    return data_list

def configure_optimizer_for_rtx4090(optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    Configure the optimizer for optimal performance on RTX 4090.
    
    Args:
        optimizer: The optimizer to configure.
    
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    # Set optimizer state to use CUDA streams
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                state = optimizer.state[p]
                # Initialize state with device placement
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(non_blocking=True)
    
    return optimizer

def setup_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set up random seed for reproducibility.
    
    Args:
        seed: Random seed.
        deterministic: Whether to enable deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable cuDNN benchmark for RTX 4090
        torch.backends.cudnn.benchmark = True

def enable_tf32() -> None:
    """Enable TF32 mode for faster computation on RTX 4090."""
    if torch.cuda.is_available():
        # Only available on Ampere (RTX 30 series) and newer (RTX 40 series)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def distributed_sampler_set_epoch(dataloader: torch.utils.data.DataLoader, epoch: int) -> None:
    """
    Set the epoch for the sampler in the dataloader for distributed training.
    
    Args:
        dataloader: DataLoader with a DistributedSampler.
        epoch: Current epoch.
    """
    if isinstance(dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

def get_rtx4090_optimized_memory_config() -> dict:
    """
    Get memory configurations optimized for RTX 4090.
    
    Returns:
        dict: Memory configuration.
    """
    # RTX 4090 has 24GB VRAM - optimize memory usage
    return {
        "max_split_size_mb": 512,
        "activation_checkpointing": True,
        "offload_optimizer": False,  # Usually not needed for RTX 4090
        "gradient_accumulation_steps": 1,  # Can be adjusted based on batch size
        "mixed_precision": "bf16",  # bf16 is fast on RTX 4090
        "gradient_clipping": 1.0
    }

def setup_rtx4090_environment() -> None:
    """Configure environment variables for optimal RTX 4090 performance."""
    # Enable cuDNN benchmark mode
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 mode for faster computation
    enable_tf32()
    
    # Set memory allocation strategy for maximum throughput
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Set OpenMP threads for optimal CPU-GPU transfer
    os.environ["OMP_NUM_THREADS"] = str(max(os.cpu_count() // 2, 1))
    
    # Use NVIDIA NCCL for communication
    os.environ["NCCL_DEBUG"] = "INFO"  # Set to WARN for less verbose output
    os.environ["NCCL_P2P_LEVEL"] = "NVL" 