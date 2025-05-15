#!/usr/bin/env python
# coding=utf-8

"""
Distributed training launcher for T2ITrainer.
This script launches the training script with proper distributed setup,
with support for both DDP and FSDP, optimized for RTX 4090 GPUs.
"""

import os
import sys
import argparse
import json
import time
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import numpy as np

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Distributed training launcher for T2ITrainer")
    parser.add_argument(
        "--num_gpus", type=int, default=torch.cuda.device_count(),
        help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--train_script", type=str, default="train_flux_lora_ui.py",
        help="Training script to run in distributed mode"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost",
        help="Master node address"
    )
    parser.add_argument(
        "--master_port", type=str, default=None,
        help="Master port (random if not specified)"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        help="Distributed backend to use (nccl, gloo, etc.)"
    )
    parser.add_argument(
        "--parallelize_mode", type=str, default="fsdp", choices=["ddp", "fsdp"],
        help="Parallelization mode to use (ddp or fsdp)"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
        help="Mixed precision mode for training"
    )
    parser.add_argument(
        "--enable_tf32", action="store_true", default=True,
        help="Enable TF32 mode on RTX GPUs for faster training"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true", default=False,
        help="Enable CPU offloading (usually not needed for RTX 4090)"
    )
    parser.add_argument(
        "--batch_size_per_gpu", type=int, default=None,
        help="Batch size per GPU. If not specified, will use the batch size from the training script"
    )
    parser.add_argument(
        "--grad_accumulation_steps", type=int, default=None,
        help="Gradient accumulation steps. If not specified, will use the value from the training script"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sharding_strategy", type=str, default="full", choices=["full", "grad_op", "shard_grad_op"],
        help="FSDP sharding strategy (only applicable if parallelize_mode is fsdp)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable profiling to identify performance bottlenecks"
    )
    parser.add_argument(
        "--distributed_evaluation", action="store_true", default=True,
        help="Enable distributed evaluation to accelerate validation"
    )
    parser.add_argument(
        "--distributed_checkpointing", action="store_true", default=True,
        help="Enable distributed checkpointing for faster model saving"
    )
    parser.add_argument(
        "--compile_model", action="store_true", default=False,
        help="Use torch.compile() to speed up model (requires PyTorch 2.0+)"
    )
    
    # Add all remaining arguments to pass to the training script
    parser.add_argument('training_args', nargs='*')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from a JSON file."""
    if config_path is None:
        return {}
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using default settings.")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_environment(rank, world_size, args, config):
    """Set up the environment for distributed training."""
    os.environ['MASTER_ADDR'] = args.master_addr
    
    if args.master_port is None:
        args.master_port = str(random.randint(10000, 20000))
    os.environ['MASTER_PORT'] = args.master_port
    
    # Set random seed for reproducibility
    random.seed(args.seed + rank)
    np.seed = args.seed + rank
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    
    # Enable TF32 for better performance on Ampere+ GPUs
    if args.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set the device and initialize the process group
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"Using GPU: {torch.cuda.get_device_name(rank)}")
    
    # Initialize distributed training
    init_method = "env://"
    dist.init_process_group(
        backend=args.backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    # Apply any additional environment settings from config
    env_config = config.get('environment', {})
    for key, value in env_config.items():
        os.environ[key] = str(value)
    
    # Set up memory optimization for RTX 4090
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Print configuration for debugging
    if rank == 0:
        print(f"Distributed training setup complete:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {args.backend}")
        print(f"  Mixed precision: {args.mixed_precision}")
        print(f"  TF32 enabled: {args.enable_tf32}")
        print(f"  Parallelization mode: {args.parallelize_mode}")
        if args.parallelize_mode == "fsdp":
            print(f"  Sharding strategy: {args.sharding_strategy}")
        
    return rank, world_size

def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def build_training_args(rank, world_size, args):
    """Build command line arguments for the training script."""
    # Start with the user-provided training arguments
    training_args = args.training_args.copy()
    
    # Add distributed training arguments
    training_args.extend([
        f"--local_rank={rank}",
        f"--world_size={world_size}",
    ])
    
    # Set batch size per GPU if provided
    if args.batch_size_per_gpu is not None:
        training_args.append(f"--train_batch_size={args.batch_size_per_gpu}")
    
    # Set gradient accumulation steps if provided
    if args.grad_accumulation_steps is not None:
        training_args.append(f"--gradient_accumulation_steps={args.grad_accumulation_steps}")
    
    # Set mixed precision mode
    if args.mixed_precision != "no":
        training_args.append(f"--mixed_precision={args.mixed_precision}")
    
    # Add distributed evaluation flag if enabled
    if args.distributed_evaluation:
        training_args.append("--distributed_evaluation")
    
    # Add distributed checkpointing flag if enabled
    if args.distributed_checkpointing:
        training_args.append("--distributed_checkpointing")
    
    # Add compile flag if enabled
    if args.compile_model:
        training_args.append("--compile_model")
    
    # Add FSDP-specific args if using FSDP
    if args.parallelize_mode == "fsdp":
        training_args.append("--use_fsdp")
        training_args.append(f"--fsdp_sharding_strategy={args.sharding_strategy}")
        if args.cpu_offload:
            training_args.append("--fsdp_cpu_offload")
    
    return training_args

def setup_rtx4090_optimization(args, config):
    """Apply optimizations specific to RTX 4090 GPUs."""
    optimizations = {
        "memory_efficient_fusion": True,
        "cudnn_benchmark": True,
        "flash_attention": True,
        "channels_last_memory_format": True,
    }
    
    # Override with config values if provided
    rtx_config = config.get('rtx4090_optimizations', {})
    optimizations.update(rtx_config)
    
    # Apply the optimizations
    if optimizations["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    
    # Add more RTX 4090 specific optimizations as needed
    return optimizations

def run_training(rank, world_size, args, config):
    """Run the training script in distributed mode."""
    try:
        # Set up environment
        rank, world_size = setup_environment(rank, world_size, args, config)
        
        # Apply RTX 4090 optimizations
        rtx_optimizations = setup_rtx4090_optimization(args, config)
        
        # Build training args
        training_args = build_training_args(rank, world_size, args)
        
        if rank == 0:
            print(f"Starting training with args: {' '.join(training_args)}")
        
        # Import the training script dynamically
        training_module = args.train_script.replace(".py", "")
        sys.path.append(os.getcwd())
        
        # Import module and run
        try:
            module = __import__(training_module)
            # Parse args and run the main function
            train_args = module.parse_args(training_args)
            
            # Enable profiling if requested
            if args.profile and rank == 0:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=10,
                        warmup=10,
                        active=20,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./profiling/{training_module}"),
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    module.main(train_args, profiler=prof)
            else:
                module.main(train_args)
                
        except Exception as e:
            print(f"Error in rank {rank}: {e}")
            import traceback
            traceback.print_exc()
            raise e
            
    except Exception as e:
        print(f"Error in setup for rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    finally:
        # Clean up
        cleanup()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration if provided
    config = load_config(args.config)
    
    # Determine world size (number of GPUs to use)
    world_size = min(args.num_gpus, torch.cuda.device_count())
    
    if world_size <= 0:
        raise ValueError(f"No GPUs available for training, found {torch.cuda.device_count()} GPUs")
    
    if world_size == 1:
        print("Only one GPU detected, running in single GPU mode")
        run_training(0, 1, args, config)
    else:
        print(f"Running distributed training on {world_size} GPUs")
        mp.spawn(
            run_training,
            args=(world_size, args, config),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    # For RTX 4090, set memory efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    main() 