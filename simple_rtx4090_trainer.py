#!/usr/bin/env python
# coding=utf-8

"""
简单的RTX 4090优化分布式训练启动器
无需Gradio UI，通过命令行配置和运行
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime

# 默认配置
default_config = {
    "script": "train_flux_lora_ui.py",
    "output_dir": "/home/lizixi/models/flux",
    "save_name": "flux-lora-rtx4090",
    "pretrained_model_name_or_path": "flux_models", 
    "train_data_dir": "/home/lizixi/datasets/flux_test", 
    "rank": 32,
    "train_batch_size": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "num_train_epochs": 20,
    "seed": 4321,
    
    # 分布式训练参数
    "num_gpus": 2,
    "parallelize_mode": "fsdp",  # fsdp 或 ddp
    "sharding_strategy": "full", # full, grad_op, shard_grad_op
    "batch_size_per_gpu": 2,
    "enable_tf32": True,
    "enable_flash_attention": True,
}

def print_banner():
    """打印启动横幅"""
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║  RTX 4090 优化的分布式训练启动器                           ║
    ║  针对RTX 4090进行特别优化 - 简化命令行版本                 ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)

def save_config(config, config_path="config_rtx4090.json"):
    """保存配置到文件"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到 {config_path}")

def load_config(config_path="config_rtx4090.json"):
    """从文件加载配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return default_config

def check_gpu_info():
    """检查GPU信息"""
    try:
        # 检查GPU型号和数量
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], 
                              check=True, capture_output=True, text=True)
        print("\n" + result.stdout)
        
        # 检测是否有RTX 4090
        if "4090" in result.stdout:
            print("✅ 检测到RTX 4090，将应用优化配置")
        else:
            print("⚠️ 未检测到RTX 4090，但仍会应用优化配置")
    except:
        print("❌ 无法获取GPU信息，请确保NVIDIA驱动正确安装")

def edit_config(config):
    """交互式编辑配置"""
    print("\n=== 编辑配置 ===")
    print("按Enter保留当前值，或输入新值")
    
    # 基本配置
    config["script"] = input(f"训练脚本 [{config['script']}]: ") or config["script"]
    config["output_dir"] = input(f"输出目录 [{config['output_dir']}]: ") or config["output_dir"]
    config["save_name"] = input(f"保存名称 [{config['save_name']}]: ") or config["save_name"]
    config["pretrained_model_name_or_path"] = input(f"预训练模型路径 [{config['pretrained_model_name_or_path']}]: ") or config["pretrained_model_name_or_path"]
    config["train_data_dir"] = input(f"训练数据目录 [{config['train_data_dir']}]: ") or config["train_data_dir"]
    
    # 训练参数
    tmp = input(f"训练批次大小 [{config['train_batch_size']}]: ") or str(config['train_batch_size'])
    config["train_batch_size"] = int(tmp)
    
    tmp = input(f"混合精度 (bf16/fp16) [{config['mixed_precision']}]: ") or config["mixed_precision"]
    config["mixed_precision"] = tmp
    
    tmp = input(f"训练轮次 [{config['num_train_epochs']}]: ") or str(config['num_train_epochs'])
    config["num_train_epochs"] = int(tmp)
    
    # 分布式参数
    tmp = input(f"GPU数量 [{config['num_gpus']}]: ") or str(config['num_gpus'])
    config["num_gpus"] = int(tmp)
    
    tmp = input(f"每GPU批次大小 [{config['batch_size_per_gpu']}]: ") or str(config['batch_size_per_gpu'])
    config["batch_size_per_gpu"] = int(tmp)
    
    tmp = input(f"并行化模式 (fsdp/ddp) [{config['parallelize_mode']}]: ") or config["parallelize_mode"]
    config["parallelize_mode"] = tmp
    
    if config["parallelize_mode"] == "fsdp":
        tmp = input(f"分片策略 (full/grad_op/shard_grad_op) [{config['sharding_strategy']}]: ") or config["sharding_strategy"]
        config["sharding_strategy"] = tmp
    
    tmp = input(f"启用TF32 (true/false) [{str(config['enable_tf32']).lower()}]: ") or str(config['enable_tf32']).lower()
    config["enable_tf32"] = tmp.lower() == "true"
    
    tmp = input(f"启用Flash Attention (true/false) [{str(config['enable_flash_attention']).lower()}]: ") or str(config['enable_flash_attention']).lower()
    config["enable_flash_attention"] = tmp.lower() == "true"
    
    return config

def run_training(config):
    """运行训练"""
    print("\n=== 开始训练 ===")
    
    # 构建命令
    cmd = ["./run_distributed_rtx4090.sh"]
    
    # 添加分布式训练参数
    cmd.extend(["-g", str(config["num_gpus"])])
    cmd.extend(["-s", config["script"]])
    cmd.extend(["-m", config["parallelize_mode"]])
    cmd.extend(["-x", config["mixed_precision"]])
    cmd.extend(["-b", str(config["batch_size_per_gpu"])])
    
    if config["parallelize_mode"] == "fsdp":
        cmd.extend(["--sharding", config["sharding_strategy"]])
    
    if not config["enable_tf32"]:
        cmd.append("--no-tf32")
    
    if not config["enable_flash_attention"]:
        cmd.append("--no-flash-attention")
    
    # 添加训练脚本的参数（使用--分隔）
    cmd.append("--")
    
    # 添加标准训练参数
    train_params = {
        "seed": config["seed"],
        "mixed_precision": config["mixed_precision"],
        "output_dir": config["output_dir"],
        "save_name": config["save_name"],
        "train_data_dir": config["train_data_dir"],
        "train_batch_size": config["train_batch_size"],
        "num_train_epochs": config["num_train_epochs"],
        "pretrained_model_name_or_path": config["pretrained_model_name_or_path"],
        "rank": config["rank"],
    }
    
    # 添加训练参数
    for key, value in train_params.items():
        if value is not None and value != "":
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # 添加布尔参数
    if config.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")
    
    # 将命令行参数转换为字符串
    cmd_str = " ".join(cmd)
    print(f"运行命令: {cmd_str}")
    
    # 创建日志目录
    log_dir = "logs/rtx4090"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_{timestamp}.log"
    
    print(f"日志将保存到: {log_file}")
    
    # 确认运行
    confirm = input("\n确认开始训练? (y/n): ")
    if confirm.lower() != 'y':
        print("已取消训练")
        return
    
    try:
        # 检查脚本是否存在并可执行
        if not os.path.exists("run_distributed_rtx4090.sh"):
            print("创建run_distributed_rtx4090.sh的符号链接...")
            os.symlink("run_distributed.sh", "run_distributed_rtx4090.sh")
            os.chmod("run_distributed_rtx4090.sh", 0o755)
        
        print("开始训练，按Ctrl+C取消...")
        process = subprocess.Popen(cmd)
        process.wait()
        print(f"训练完成！查看日志: {log_file}")
    except KeyboardInterrupt:
        print("\n训练已中断")
    except Exception as e:
        print(f"训练出错: {e}")

def main():
    """主函数"""
    print_banner()
    
    # 检查GPU信息
    check_gpu_info()
    
    # 加载配置
    config = load_config()
    
    while True:
        print("\n=== RTX 4090分布式训练菜单 ===")
        print("1. 编辑配置")
        print("2. 开始训练")
        print("3. 保存配置")
        print("4. 加载配置")
        print("5. 退出")
        
        choice = input("\n请选择操作 (1-5): ")
        
        if choice == "1":
            config = edit_config(config)
        elif choice == "2":
            run_training(config)
        elif choice == "3":
            path = input("配置文件路径 [config_rtx4090.json]: ") or "config_rtx4090.json"
            save_config(config, path)
        elif choice == "4":
            path = input("配置文件路径 [config_rtx4090.json]: ") or "config_rtx4090.json"
            config = load_config(path)
            print(f"已加载配置: {path}")
        elif choice == "5":
            print("退出程序")
            break
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    main() 