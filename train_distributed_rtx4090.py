#!/usr/bin/env python
# coding=utf-8

"""
RTX 4090优化的分布式训练启动器
支持FSDP和DDP两种模式，针对RTX 4090特性优化
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
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
import numpy as np
import traceback
from datetime import datetime

# 梯度压缩算法实现
try:
    import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
    HAS_POWERSGD = True
except ImportError:
    HAS_POWERSGD = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RTX 4090优化的分布式训练启动器")
    parser.add_argument(
        "--num_gpus", type=int, default=torch.cuda.device_count(),
        help="用于训练的GPU数量"
    )
    parser.add_argument(
        "--train_script", type=str, default="train_flux_lora_ui.py",
        help="要运行的训练脚本"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost",
        help="主节点地址"
    )
    parser.add_argument(
        "--master_port", type=str, default=None,
        help="主节点端口(未指定则随机)"
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        help="分布式后端 (nccl, gloo等)"
    )
    parser.add_argument(
        "--parallelize_mode", type=str, default="fsdp", choices=["ddp", "fsdp"],
        help="并行化模式 (ddp或fsdp)"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
        help="混合精度训练模式"
    )
    parser.add_argument(
        "--enable_tf32", action="store_true", default=True,
        help="在RTX GPU上启用TF32模式以加速训练"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true", default=False,
        help="启用CPU卸载(通常RTX 4090不需要)"
    )
    parser.add_argument(
        "--batch_size_per_gpu", type=int, default=None,
        help="每GPU批次大小，未指定则使用训练脚本的值"
    )
    parser.add_argument(
        "--grad_accumulation_steps", type=int, default=None,
        help="梯度累积步数，未指定则使用训练脚本的值"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--sharding_strategy", type=str, default="full", 
        choices=["full", "grad_op", "shard_grad_op"],
        help="FSDP分片策略(仅当parallelize_mode为fsdp时有效)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="JSON配置文件路径"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="启用性能分析以识别性能瓶颈"
    )
    parser.add_argument(
        "--distributed_evaluation", action="store_true", default=True,
        help="启用分布式评估以加速验证"
    )
    parser.add_argument(
        "--distributed_checkpointing", action="store_true", default=True,
        help="启用分布式检查点保存以加速模型保存"
    )
    parser.add_argument(
        "--compile_model", action="store_true", default=False,
        help="使用torch.compile()加速模型(需要PyTorch 2.0+)"
    )
    parser.add_argument(
        "--grad_compression", action="store_true", default=False,
        help="启用梯度压缩(PowerSGD)以减少通信开销"
    )
    parser.add_argument(
        "--adaptive_lr", action="store_true", default=False,
        help="启用自适应学习率调整"
    )
    parser.add_argument(
        "--use_progressive_batching", action="store_true", default=False, 
        help="使用渐进式批次大小"
    )
    parser.add_argument(
        "--backward_prefetch", type=str, default="BACKWARD_PRE", 
        choices=["BACKWARD_PRE", "BACKWARD_POST", "NO_PREFETCH"],
        help="FSDP反向预取策略"
    )
    
    # 添加所有剩余参数以传递给训练脚本
    parser.add_argument('training_args', nargs='*')
    
    return parser.parse_args()

def load_config(config_path):
    """从JSON文件加载配置"""
    if config_path is None:
        return {}
    
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 {config_path} 未找到。使用默认设置。")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_environment(rank, world_size, args, config):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = args.master_addr
    
    if args.master_port is None:
        args.master_port = str(random.randint(10000, 20000))
    os.environ['MASTER_PORT'] = args.master_port
    
    # 设置随机种子以实现可重复性
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    
    # 在Ampere+GPU上启用TF32以提高性能
    if args.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 设置设备并初始化进程组
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"使用GPU: {torch.cuda.get_device_name(rank)}")
    
    # 初始化分布式训练
    init_method = "env://"
    dist.init_process_group(
        backend=args.backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    # 应用来自配置的任何其他环境设置
    env_config = config.get('environment', {})
    for key, value in env_config.items():
        os.environ[key] = str(value)
    
    # 设置RTX 4090内存优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 为RTX 4090设置NCCL优化
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    
    # 打印调试信息
    if rank == 0:
        print(f"分布式训练设置完成:")
        print(f"  世界大小: {world_size}")
        print(f"  后端: {args.backend}")
        print(f"  混合精度: {args.mixed_precision}")
        print(f"  TF32启用: {args.enable_tf32}")
        print(f"  并行化模式: {args.parallelize_mode}")
        if args.parallelize_mode == "fsdp":
            print(f"  分片策略: {args.sharding_strategy}")
        
    return rank, world_size

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def build_training_args(rank, world_size, args):
    """为训练脚本构建命令行参数"""
    # 从用户提供的训练参数开始
    training_args = args.training_args.copy()
    
    # 添加分布式训练参数
    training_args.extend([
        f"--local_rank={rank}",
        f"--world_size={world_size}",
    ])
    
    # 如果提供了每GPU批次大小，则设置
    if args.batch_size_per_gpu is not None:
        training_args.append(f"--train_batch_size={args.batch_size_per_gpu}")
    
    # 如果提供了梯度累积步数，则设置
    if args.grad_accumulation_steps is not None:
        training_args.append(f"--gradient_accumulation_steps={args.grad_accumulation_steps}")
    
    # 设置混合精度模式
    if args.mixed_precision != "no":
        training_args.append(f"--mixed_precision={args.mixed_precision}")
    
    # 添加分布式评估标志（如果启用）
    if args.distributed_evaluation:
        training_args.append("--distributed_evaluation")
    
    # 添加分布式检查点标志（如果启用）
    if args.distributed_checkpointing:
        training_args.append("--distributed_checkpointing")
    
    # 添加编译标志（如果启用）
    if args.compile_model:
        training_args.append("--compile_model")
    
    # 添加自适应学习率标志（如果启用）
    if args.adaptive_lr:
        training_args.append("--adaptive_learning_rate")
    
    # 添加渐进式批次大小标志（如果启用）
    if args.use_progressive_batching:
        training_args.append("--use_progressive_batching")
    
    # 如果使用FSDP，添加FSDP特定参数
    if args.parallelize_mode == "fsdp":
        training_args.append("--use_fsdp")
        training_args.append(f"--fsdp_sharding_strategy={args.sharding_strategy}")
        training_args.append(f"--fsdp_backward_prefetch={args.backward_prefetch}")
        if args.cpu_offload:
            training_args.append("--fsdp_cpu_offload")
    
    return training_args

def setup_rtx4090_optimization(args, config):
    """应用特定于RTX 4090 GPU的优化"""
    optimizations = {
        "memory_efficient_fusion": True,
        "cudnn_benchmark": True,
        "flash_attention": True,
        "channels_last_memory_format": True,
    }
    
    # 如果提供了配置值，则覆盖
    rtx_config = config.get('rtx4090_optimizations', {})
    optimizations.update(rtx_config)
    
    # 应用优化
    if optimizations["cudnn_benchmark"]:
        torch.backends.cudnn.benchmark = True
    
    # 额外的RTX 4090特定优化
    if torch.cuda.get_device_capability()[0] >= 8:  # Ampere及更高版本
        # 启用内存节省技术
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        # 启用快速GEMM算法
        if hasattr(torch.backends.cuda, 'enable_math_sdp') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
    
    return optimizations

def apply_grad_compression(model, rank, args):
    """应用梯度压缩以减少通信开销"""
    if not HAS_POWERSGD:
        print(f"警告: PowerSGD不可用，跳过梯度压缩")
        return
    
    if isinstance(model, DDP) and args.grad_compression:
        # 配置PowerSGD状态
        state = powerSGD.PowerSGDState(
            process_group=None,  # 默认进程组
            matrix_approximation_rank=1,  # 低秩近似，较小的值=更多压缩
            start_powerSGD_iter=10,  # 开始使用PowerSGD之前的预热迭代
            use_error_feedback=True,  # 使用错误反馈以提高精度
        )
        # 注册钩子
        model.register_comm_hook(state, powerSGD.powerSGD_hook)
        if rank == 0:
            print("已启用PowerSGD梯度压缩")

def setup_mixed_precision(args):
    """设置混合精度训练"""
    # 为RTX 4090优化混合精度设置
    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    
    # 返回适用于model.to()的数据类型
    return dtype

def run_training(rank, world_size, args, config):
    """在分布式模式下运行训练脚本"""
    try:
        # 设置环境
        rank, world_size = setup_environment(rank, world_size, args, config)
        
        # 应用RTX 4090优化
        rtx_optimizations = setup_rtx4090_optimization(args, config)
        
        # 设置混合精度
        weight_dtype = setup_mixed_precision(args)
        
        # 构建训练参数
        training_args = build_training_args(rank, world_size, args)
        
        if rank == 0:
            print(f"开始使用参数训练: {' '.join(training_args)}")
        
        # 动态导入训练脚本
        training_module = args.train_script.replace(".py", "")
        sys.path.append(os.getcwd())
        
        # 导入模块并运行
        try:
            module = __import__(training_module)
            # 解析参数并运行主函数
            train_args = module.parse_args(training_args)
            
            # 如果需要，启用性能分析
            if args.profile and rank == 0:
                log_dir = "profiling"
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
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
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{log_dir}/{training_module}_{timestamp}"),
                    record_shapes=True,
                    profile_memory=True,
                ) as prof:
                    module.main(train_args, profiler=prof)
            else:
                module.main(train_args)
                
        except Exception as e:
            print(f"在rank {rank}中出错: {e}")
            traceback.print_exc()
            raise e
            
    except Exception as e:
        print(f"在rank {rank}的设置中出错: {e}")
        traceback.print_exc()
        raise e
        
    finally:
        # 清理
        cleanup()

def main():
    """主入口点"""
    start_time = time.time()
    args = parse_args()
    
    # 如果提供了配置，则加载
    config = load_config(args.config)
    
    # 确定世界大小（使用的GPU数量）
    world_size = min(args.num_gpus, torch.cuda.device_count())
    
    if world_size <= 0:
        raise ValueError(f"没有可用的GPU进行训练，找到 {torch.cuda.device_count()} 个GPU")
    
    if world_size == 1:
        print("仅检测到一个GPU，以单GPU模式运行")
        run_training(0, 1, args, config)
    else:
        print(f"在 {world_size} 个GPU上运行分布式训练")
        mp.spawn(
            run_training,
            args=(world_size, args, config),
            nprocs=world_size,
            join=True
        )
    
    # 打印总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"训练完成! 总运行时间: {int(hours)}时 {int(minutes)}分 {seconds:.2f}秒")

if __name__ == "__main__":
    # 为RTX 4090设置内存效率
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    main() 