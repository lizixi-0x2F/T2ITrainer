"""
FSDP (Fully Sharded Data Parallel) utilities for distributed training.
Optimized for RTX 4090 GPUs.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
    LocalStateDictConfig,
    ShardedStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import functools
from typing import Dict, Any, Optional, Tuple, List
from transformers.modeling_utils import PreTrainedModel

def get_mixed_precision_config(precision: str = "bf16") -> Optional[MixedPrecision]:
    """
    Get mixed precision configuration for FSDP.
    
    Args:
        precision: Mixed precision mode ("bf16", "fp16", or None)
    
    Returns:
        MixedPrecision configuration or None if not using mixed precision
    """
    if precision == "bf16":
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    elif precision == "fp16":
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
    else:
        return None

def get_auto_wrap_policy(transformer_layer_cls=None, min_params: int = 1e8):
    """
    Get auto wrap policy for FSDP.
    
    Args:
        transformer_layer_cls: Transformer layer class or list of classes
        min_params: Minimum number of parameters for a module to be wrapped
                   
    Returns:
        Auto wrap policy function
    """
    if transformer_layer_cls is not None:
        if not isinstance(transformer_layer_cls, list):
            transformer_layer_cls = [transformer_layer_cls]
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls
        )
    else:
        return functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=min_params
        )

def get_sharding_strategy(strategy_name: str = "full") -> ShardingStrategy:
    """
    Get sharding strategy for FSDP.
    
    Args:
        strategy_name: Sharding strategy ("full", "grad_op", "shard_grad_op")
    
    Returns:
        ShardingStrategy
    """
    if strategy_name == "full":
        return ShardingStrategy.FULL_SHARD
    elif strategy_name == "grad_op":
        return ShardingStrategy.SHARD_GRAD_OP
    elif strategy_name == "shard_grad_op":
        return ShardingStrategy.HYBRID_SHARD
    else:
        return ShardingStrategy.FULL_SHARD  # Default

def get_cpu_offload_config(offload: bool = False) -> CPUOffload:
    """
    Get CPU offload configuration for FSDP.
    
    Args:
        offload: Whether to offload parameters to CPU
    
    Returns:
        CPUOffload configuration
    """
    return CPUOffload(offload_params=offload)

def get_fsdp_kwargs(
    mixed_precision: str = "bf16",
    sharding_strategy: str = "full",
    cpu_offload: bool = False,
    transformer_layer_cls = None,
    min_params: int = 1e8,
    backward_prefetch: bool = True,
    use_orig_params: bool = True
) -> Dict[str, Any]:
    """
    Get FSDP initialization arguments.
    
    Args:
        mixed_precision: Mixed precision mode ("bf16", "fp16", or None)
        sharding_strategy: Sharding strategy ("full", "grad_op", "shard_grad_op")
        cpu_offload: Whether to offload parameters to CPU
        transformer_layer_cls: Transformer layer class or list of classes
        min_params: Minimum number of parameters for a module to be wrapped
        backward_prefetch: Whether to prefetch gradients during backward pass
        use_orig_params: Whether to use original model parameters
        
    Returns:
        Dictionary of FSDP initialization arguments
    """
    fsdp_kwargs = {
        "mixed_precision": get_mixed_precision_config(mixed_precision),
        "sharding_strategy": get_sharding_strategy(sharding_strategy),
        "cpu_offload": get_cpu_offload_config(cpu_offload),
        "auto_wrap_policy": get_auto_wrap_policy(transformer_layer_cls, min_params),
        "device_id": torch.cuda.current_device(),
        "use_orig_params": use_orig_params,
    }
    
    if backward_prefetch:
        fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
    
    return fsdp_kwargs

def wrap_model_with_fsdp(
    model: nn.Module,
    **fsdp_kwargs
) -> FSDP:
    """
    Wrap a model with FSDP.
    
    Args:
        model: Model to wrap
        **fsdp_kwargs: FSDP initialization arguments
    
    Returns:
        FSDP-wrapped model
    """
    if "device_id" not in fsdp_kwargs:
        fsdp_kwargs["device_id"] = torch.cuda.current_device()

    # Move model to current device before wrapping with FSDP
    model = model.to(torch.cuda.current_device())
    
    # Apply FSDP wrapping
    fsdp_model = FSDP(model, **fsdp_kwargs)
    
    return fsdp_model

def load_model_with_fsdp_checkpoint(
    model_class,
    checkpoint_path: str,
    model_config=None,
    map_location=None,
    **fsdp_kwargs
) -> FSDP:
    """
    Load a model from a checkpoint and wrap it with FSDP.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to the checkpoint
        model_config: Model configuration
        map_location: Location to map checkpoint tensors to
        **fsdp_kwargs: FSDP initialization arguments
    
    Returns:
        FSDP-wrapped model loaded from checkpoint
    """
    if map_location is None:
        map_location = f"cuda:{torch.cuda.current_device()}"
    
    # Create model
    if model_config is not None:
        model = model_class(model_config)
    else:
        model = model_class()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # Check if it's a HuggingFace model
    if isinstance(model, PreTrainedModel):
        model.load_state_dict(checkpoint)
    else:
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    # Wrap with FSDP
    fsdp_model = wrap_model_with_fsdp(model, **fsdp_kwargs)
    
    return fsdp_model

def save_fsdp_model_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    global_step: int,
    state_dict_type: str = "full"
):
    """
    Save an FSDP model checkpoint.
    
    Args:
        model: FSDP-wrapped model
        optimizer: Optimizer
        save_path: Path to save the checkpoint
        global_step: Global training step
        state_dict_type: Type of state dict to save ("full", "local", "sharded")
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Choose the right state dict type
    if state_dict_type == "full":
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True)):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    elif state_dict_type == "local":
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig()):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    elif state_dict_type == "sharded":
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig()):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        raise ValueError(f"Unknown state_dict_type: {state_dict_type}")
    
    # Save state dicts
    if torch.distributed.get_rank() == 0:  # Only save on main process
        checkpoint = {
            "model": model_state,
            "optimizer": optim_state,
            "global_step": global_step,
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

def get_transformer_layer_cls_from_model(model):
    """
    Attempt to automatically detect transformer layer classes from a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        List of detected transformer layer classes
    """
    # Common transformer layer class names
    potential_layer_names = [
        "TransformerBlock", "TransformerLayer", "TransformerEncoderLayer", 
        "TransformerDecoderLayer", "EncoderLayer", "DecoderLayer",
        "BertLayer", "GPTLayer", "T5Layer", "T5Block"
    ]
    
    layer_classes = []
    for name, module in model.named_modules():
        for layer_name in potential_layer_names:
            if layer_name.lower() in type(module).__name__.lower():
                layer_classes.append(type(module))
                break
    
    # Return unique classes
    return list(set(layer_classes))

def configure_fsdp_for_diffusion(model):
    """
    Configure FSDP specifically for diffusion models.
    Automatically detects appropriate layer classes and returns optimized FSDP kwargs.
    
    Args:
        model: Diffusion model
        
    Returns:
        Dictionary of FSDP initialization arguments optimized for diffusion models
    """
    # Try to detect transformer layers
    transformer_layers = get_transformer_layer_cls_from_model(model)
    
    # For RTX 4090, BF16 is optimal
    return get_fsdp_kwargs(
        mixed_precision="bf16",
        sharding_strategy="full",
        cpu_offload=False,  # RTX 4090 has plenty of memory
        transformer_layer_cls=transformer_layers if transformer_layers else None,
        min_params=1e8 if not transformer_layers else 0,  # Use size-based policy if no transformer layers found
        backward_prefetch=True,
        use_orig_params=True
    ) 