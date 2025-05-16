"""
Distributed evaluation utilities for T2ITrainer.
Optimized for RTX 4090 GPUs.
"""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from PIL import Image
import time
from torch.cuda.amp import autocast
import random

from utils.distributed_utils import (
    get_rank, get_world_size, is_main_process, 
    all_gather, synchronize
)

def distributed_image_generation(
    pipeline,
    prompts: List[str],
    batch_size: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> Tuple[List[Image.Image], float]:
    """
    Distributed image generation across multiple GPUs.
    
    Args:
        pipeline: Diffusion pipeline to use for generation
        prompts: List of prompts to generate images for
        batch_size: Batch size per GPU
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for classifier-free guidance
        negative_prompt: Negative prompt for classifier-free guidance
        seed: Random seed for reproducibility
        height: Height of generated images (if None, use pipeline default)
        width: Width of generated images (if None, use pipeline default)
        
    Returns:
        Tuple of (list of generated images, generation time)
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed + rank)
    else:
        generator = None
    
    # Distribute prompts across GPUs
    num_prompts = len(prompts)
    prompts_per_rank = (num_prompts + world_size - 1) // world_size
    start_idx = rank * prompts_per_rank
    end_idx = min(start_idx + prompts_per_rank, num_prompts)
    rank_prompts = prompts[start_idx:end_idx]
    
    # Create negative prompts if provided
    if negative_prompt is not None:
        if isinstance(negative_prompt, str):
            rank_negative_prompts = [negative_prompt] * len(rank_prompts)
        else:
            rank_negative_prompts = negative_prompt[start_idx:end_idx]
    else:
        rank_negative_prompts = None
    
    # Generate images in batches
    images = []
    total_gen_time = 0
    
    for i in range(0, len(rank_prompts), batch_size):
        batch_prompts = rank_prompts[i:i+batch_size]
        
        if rank_negative_prompts is not None:
            batch_negative_prompts = rank_negative_prompts[i:i+batch_size]
        else:
            batch_negative_prompts = None
        
        # Measure generation time
        start_time = time.time()
        
        # Generate images with mixed precision for RTX 4090
        with autocast(dtype=torch.bfloat16):
            batch_images = pipeline(
                prompt=batch_prompts,
                negative_prompt=batch_negative_prompts,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width
            ).images
        
        end_time = time.time()
        batch_gen_time = end_time - start_time
        total_gen_time += batch_gen_time
        
        images.extend(batch_images)
    
    # Gather images from all ranks
    all_images = all_gather(images)
    
    # Flatten the list of lists
    if is_main_process():
        all_images_flat = []
        for img_list in all_images:
            all_images_flat.extend(img_list)
        
        # Preserve original order based on prompts
        # Create (index, image) pairs based on the distributed chunks
        indexed_images = []
        for proc_rank, img_list in enumerate(all_images):
            start = proc_rank * prompts_per_rank
            for i, img in enumerate(img_list):
                if start + i < num_prompts:  # Make sure we don't go out of bounds
                    indexed_images.append((start + i, img))
        
        # Sort by index and extract just the images
        all_images_flat = [img for _, img in sorted(indexed_images)]
        
        # Get average generation time across all ranks
        all_gen_times = all_gather(total_gen_time)
        avg_gen_time = sum(all_gen_times) / len(all_gen_times)
        
        return all_images_flat, avg_gen_time
    else:
        return [], 0.0

def distributed_validation(
    pipeline,
    validation_prompts: List[str],
    validation_seeds: List[int],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> List[Image.Image]:
    """
    Run distributed validation with fixed seeds for reproducible evaluation.
    
    Args:
        pipeline: Diffusion pipeline to use for generation
        validation_prompts: List of validation prompts
        validation_seeds: List of seeds to use for each prompt
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for classifier-free guidance
        negative_prompt: Negative prompt for classifier-free guidance
        height: Height of generated images (if None, use pipeline default)
        width: Width of generated images (if None, use pipeline default)
        
    Returns:
        List of generated validation images
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Ensure we have a seed for each prompt
    assert len(validation_prompts) == len(validation_seeds), \
        "Number of validation prompts must match number of validation seeds"
    
    # Distribute prompts and seeds across GPUs
    num_prompts = len(validation_prompts)
    prompts_per_rank = (num_prompts + world_size - 1) // world_size
    start_idx = rank * prompts_per_rank
    end_idx = min(start_idx + prompts_per_rank, num_prompts)
    
    rank_prompts = validation_prompts[start_idx:end_idx]
    rank_seeds = validation_seeds[start_idx:end_idx]
    
    # Generate validation images
    images = []
    
    for prompt, seed in zip(rank_prompts, rank_seeds):
        # Set seed for reproducibility
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # Generate image with mixed precision for RTX 4090
        with autocast(dtype=torch.bfloat16):
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=height,
                width=width
            )
        
        images.append(result.images[0])
    
    # Gather images from all ranks
    all_images = all_gather(images)
    
    # Flatten and reorder the images
    if is_main_process():
        all_images_flat = []
        for proc_rank, img_list in enumerate(all_images):
            start = proc_rank * prompts_per_rank
            for i, img in enumerate(img_list):
                if start + i < num_prompts:
                    all_images_flat.append((start + i, img))
        
        # Sort by index and extract just the images
        all_images_flat = [img for _, img in sorted(all_images_flat)]
        
        return all_images_flat
    else:
        return []

def evaluate_fid_distributed(
    real_images_dir: str,
    generated_images: List[Image.Image],
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> float:
    """
    Calculate FID score using distributed computation.
    
    Args:
        real_images_dir: Directory containing real images
        generated_images: List of generated images
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        FID score
    """
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import InceptionV3
    except ImportError:
        if is_main_process():
            print("pytorch-fid not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "pytorch-fid"])
            from pytorch_fid import fid_score
            from pytorch_fid.inception import InceptionV3
    
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # Only run FID calculation on the main process
    if not is_main_process():
        synchronize()
        return 0.0
    
    # Save generated images to a temporary directory
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    
    for i, img in enumerate(generated_images):
        img.save(os.path.join(temp_dir, f"gen_{i:04d}.png"))
    
    # Calculate FID score
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_dir, temp_dir],
            batch_size,
            device,
            dims=2048
        )
    except Exception as e:
        print(f"Error calculating FID score: {e}")
        fid_value = float('nan')
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)
    
    synchronize()
    return fid_value

def evaluate_clip_score_distributed(
    pipeline,
    prompts: List[str],
    generated_images: List[Image.Image],
    batch_size: int = 16,
    device: Optional[torch.device] = None
) -> float:
    """
    Calculate CLIP score using distributed computation.
    
    Args:
        pipeline: Pipeline with CLIP model
        prompts: Text prompts corresponding to generated images
        generated_images: List of generated images
        batch_size: Batch size for feature extraction
        device: Device to use for computation
        
    Returns:
        Average CLIP score
    """
    world_size = get_world_size()
    rank = get_rank()
    
    if device is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # Check if we have access to the CLIP model
    try:
        import clip
    except ImportError:
        if is_main_process():
            print("OpenAI CLIP not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "git+https://github.com/openai/CLIP.git"])
            import clip
    
    # Load CLIP model
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except:
        # If we couldn't get OpenAI CLIP, try using the one from pipeline
        if hasattr(pipeline, "text_encoder") and hasattr(pipeline, "tokenizer"):
            model = pipeline.text_encoder
            preprocess = pipeline.feature_extractor
        else:
            if is_main_process():
                print("Could not load CLIP model. Skipping CLIP score evaluation.")
            synchronize()
            return 0.0
    
    # Distribute images and prompts across GPUs
    num_images = len(generated_images)
    images_per_rank = (num_images + world_size - 1) // world_size
    start_idx = rank * images_per_rank
    end_idx = min(start_idx + images_per_rank, num_images)
    
    rank_images = generated_images[start_idx:end_idx]
    rank_prompts = [prompts[i % len(prompts)] for i in range(start_idx, end_idx)]
    
    # Calculate CLIP scores
    clip_scores = []
    
    for i in range(0, len(rank_images), batch_size):
        batch_images = rank_images[i:i+batch_size]
        batch_prompts = rank_prompts[i:i+batch_size]
        
        # Preprocess images
        image_inputs = torch.stack([preprocess(img).to(device) for img in batch_images])
        
        # Encode images and text
        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                image_features = model.encode_image(image_inputs)
                text_features = model.encode_text(clip.tokenize(batch_prompts).to(device))
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).diag()
        clip_scores.extend(similarity.tolist())
    
    # Gather scores from all ranks
    all_scores = all_gather(clip_scores)
    
    # Calculate average CLIP score
    if is_main_process():
        all_scores_flat = []
        for score_list in all_scores:
            all_scores_flat.extend(score_list)
        
        # Make sure we only consider the number of actual images
        all_scores_flat = all_scores_flat[:num_images]
        avg_clip_score = sum(all_scores_flat) / len(all_scores_flat)
        
        return avg_clip_score
    else:
        return 0.0

def distributed_infer_batch(
    model,
    batch,
    inference_fn: Callable,
    use_mixed_precision: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Run inference on a batch using distributed processing.
    
    Args:
        model: Model to use for inference
        batch: Input batch
        inference_fn: Function to perform inference (model, batch) -> outputs
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Dictionary of inference outputs
    """
    # Move batch to current device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Run inference with mixed precision if requested
    if use_mixed_precision:
        with autocast(dtype=torch.bfloat16):
            outputs = inference_fn(model, batch)
    else:
        outputs = inference_fn(model, batch)
    
    # Gather outputs from all processes
    output_keys = list(outputs.keys())
    gathered_outputs = {}
    
    for key in output_keys:
        if isinstance(outputs[key], torch.Tensor):
            all_tensors = all_gather(outputs[key])
            if is_main_process():
                gathered_outputs[key] = torch.cat(all_tensors, dim=0)
        else:
            gathered_values = all_gather(outputs[key])
            if is_main_process():
                if isinstance(gathered_values[0], list):
                    gathered_outputs[key] = []
                    for val_list in gathered_values:
                        gathered_outputs[key].extend(val_list)
                else:
                    gathered_outputs[key] = gathered_values
    
    return gathered_outputs

def validate_model_distributed(
    model,
    val_dataloader,
    inference_fn: Callable,
    metrics_fn: Callable,
    use_mixed_precision: bool = True
) -> Dict[str, float]:
    """
    Validate a model using distributed processing.
    
    Args:
        model: Model to validate
        val_dataloader: Validation dataloader
        inference_fn: Function to perform inference (model, batch) -> outputs
        metrics_fn: Function to compute metrics (outputs) -> metrics
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Dictionary of validation metrics
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Set model to eval mode
    model.eval()
    
    # Create a distributed sampler for the validation dataloader if not already wrapped
    if not hasattr(val_dataloader, 'sampler') or not isinstance(val_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataloader.dataset,
            batch_size=val_dataloader.batch_size,
            sampler=sampler,
            num_workers=val_dataloader.num_workers,
            collate_fn=val_dataloader.collate_fn if hasattr(val_dataloader, 'collate_fn') else None,
            pin_memory=True
        )
    
    # Run validation
    all_outputs = {}
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", disable=not is_main_process()):
            # Run inference
            outputs = distributed_infer_batch(
                model,
                batch,
                inference_fn,
                use_mixed_precision
            )
            
            # Collect outputs
            for key, value in outputs.items():
                if key not in all_outputs:
                    all_outputs[key] = []
                all_outputs[key].append(value)
    
    # Concatenate outputs
    for key in all_outputs.keys():
        if isinstance(all_outputs[key][0], torch.Tensor):
            all_outputs[key] = torch.cat(all_outputs[key], dim=0)
        elif isinstance(all_outputs[key][0], list):
            combined = []
            for val_list in all_outputs[key]:
                combined.extend(val_list)
            all_outputs[key] = combined
    
    # Compute metrics
    metrics = metrics_fn(all_outputs) if is_main_process() else {}
    
    # Synchronize before returning
    synchronize()
    
    return metrics 