#!/usr/bin/env python
# coding=utf-8

"""
Distributed training launcher for T2ITrainer.
This script launches the training script with proper distributed setup.
"""

import os
import sys
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch
import random

def parse_args():
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
    
    # Add all remaining arguments to pass to the training script
    parser.add_argument('training_args', nargs='*')
    
    return parser.parse_args()

def setup(rank, world_size, args):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = args.master_addr
    if args.master_port is None:
        args.master_port = str(random.randint(10000, 20000))
    os.environ['MASTER_PORT'] = args.master_port
    
    # Initialize the process group
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)
    
    print(f"Setup complete for rank {rank} of {world_size}")

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def run_training(rank, world_size, args):
    """Run the training script in distributed mode."""
    setup(rank, world_size, args)
    
    # Construct the command line arguments for the training script
    cmd_args = [
        f"--local_rank={rank}",
        f"--world_size={world_size}",
    ] + args.training_args
    
    # Import the training script dynamically
    training_module = args.train_script.replace(".py", "")
    sys.path.append(os.getcwd())
    
    try:
        # Import the module and run the main function
        module = __import__(training_module)
        # Parse args and run the main function
        train_args = module.parse_args(cmd_args)
        module.main(train_args)
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise e
    finally:
        cleanup()

def main():
    args = parse_args()
    world_size = min(args.num_gpus, torch.cuda.device_count())
    
    if world_size <= 0:
        raise ValueError(f"No GPUs available for training, found {torch.cuda.device_count()} GPUs")
    
    if world_size == 1:
        print("Only one GPU detected, running in single GPU mode")
        run_training(0, 1, args)
    else:
        print(f"Running distributed training on {world_size} GPUs")
        mp.spawn(
            run_training,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    # For RTX 4090, set memory efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    main() 