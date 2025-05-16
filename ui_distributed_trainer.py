import gradio as gr
import subprocess
import json
import sys
import os
import glob
import shutil

# 默认配置
default_config = {
    "script": "train_flux_lora_ui.py",
    "script_choices": [
                        "train_flux_lora_ui.py",
                        "train_kolors_lora_ui.py",
                       ],
    "output_dir": "/home/lizixi/models/flux",
    "save_name": "flux-lora-rtx4090",
    "pretrained_model_name_or_path": "flux_models", 
    "train_data_dir": "/home/lizixi/datasets/flux_test", 
    "vae_path": None,
    "resume_from_checkpoint": None,
    "model_path": None, 
    "report_to": "wandb", 
    "rank": 32,
    "train_batch_size": 1,
    "repeats": 10,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "optimizer": "adamw",
    "lr_scheduler": "cosine", 
    "learning_rate": 1e-4,
    "lr_warmup_steps": 100,
    "seed": 4321,
    "num_train_epochs": 20,
    "save_model_epochs": 1, 
    "validation_epochs": 1, 
    "skip_epoch": 0, 
    "skip_step": 0, 
    "validation_ratio": 0.1, 
    "recreate_cache": False,
    "caption_dropout": 0.1,
    "config_path": "config_rtx4090.json",
    "resolution": "512",
    "resolution_choices": ["1024", "512"],
    "use_debias": False,
    "snr_gamma": 5,
    "cosine_restarts": 1,
    "max_time_steps": 0,
    "noise_offset": 0.01,
    "blocks_to_swap": 10,
    "weighting_scheme": "logit_normal",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    
    # 分布式训练参数
    "distributed_config": {
        "num_gpus": 2,
        "parallelize_mode": "fsdp",  # fsdp 或 ddp
        "sharding_strategy": "full", # full, grad_op, shard_grad_op
        "batch_size_per_gpu": 2,
        "enable_tf32": True,
        "enable_flash_attention": True,
        "distributed_evaluation": True,
        "cpu_offload": False,
        "compile_model": False,
        "profile": False
    },
    "advanced_training": {
        "progressive_batching": True,
        "batch_schedule": [
            {"batch_size": 2, "training_steps": 500},
            {"batch_size": 4, "training_steps": 2000}
        ],
        "gradient_compression": False,
        "adaptive_learning_rate": True
    }
}

# 保存配置到指定目录
def save_config( 
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        vae_path,
        resolution,
        use_debias,
        snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        noise_offset,
        blocks_to_swap,
        weighting_scheme,
        logit_mean,
        logit_std,
        # 分布式参数
        num_gpus,
        parallelize_mode,
        sharding_strategy,
        batch_size_per_gpu,
        enable_tf32,
        enable_flash_attention,
        distributed_evaluation,
        cpu_offload,
        compile_model,
        profile,
        progressive_batching,
        gradient_compression,
        adaptive_learning_rate
    ):
    config = {
        "script": script,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "report_to": report_to,
        "lr_warmup_steps": lr_warmup_steps,
        "output_dir": output_dir,
        "save_name": save_name,
        "train_data_dir": train_data_dir,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "repeats": repeats,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "save_model_epochs": save_model_epochs,
        "validation_epochs": validation_epochs,
        "rank": rank,
        "skip_epoch": skip_epoch,
        "skip_step": skip_step,
        "gradient_checkpointing": gradient_checkpointing,
        "validation_ratio": validation_ratio,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "model_path": model_path,
        "resume_from_checkpoint": resume_from_checkpoint,
        "recreate_cache": recreate_cache,
        "vae_path": vae_path,
        "config_path": config_path,
        "resolution": resolution,
        "use_debias": use_debias,
        'snr_gamma': snr_gamma,
        "caption_dropout": caption_dropout,
        "cosine_restarts": cosine_restarts,
        "max_time_steps": max_time_steps,
        "noise_offset": noise_offset,
        "blocks_to_swap": blocks_to_swap,
        "weighting_scheme": weighting_scheme,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "distributed_config": {
            "num_gpus": num_gpus,
            "parallelize_mode": parallelize_mode,
            "sharding_strategy": sharding_strategy,
            "batch_size_per_gpu": batch_size_per_gpu,
            "enable_tf32": enable_tf32,
            "enable_flash_attention": enable_flash_attention,
            "distributed_evaluation": distributed_evaluation,
            "cpu_offload": cpu_offload,
            "compile_model": compile_model,
            "profile": profile
        },
        "advanced_training": {
            "progressive_batching": progressive_batching,
            "gradient_compression": gradient_compression,
            "adaptive_learning_rate": adaptive_learning_rate
        }
    }
    
    # 创建配置文件目录
    os.makedirs(os.path.dirname(os.path.abspath(config_path)) or ".", exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到 {config_path}")
    print(f"更新默认配置")
    with open("config_rtx4090.json", 'w') as f:
        json.dump(config, f, indent=4)

# 从指定目录加载配置
def load_config(config_path):
    if not config_path.endswith(".json"):
        print("!!!文件不是json格式。")
        print("加载默认配置")
        config_path = "config_rtx4090.json"
    if not os.path.exists(config_path):
        # 创建默认配置
        os.makedirs(os.path.dirname(os.path.abspath(config_path)) or ".", exist_ok=True)
        with open(config_path, 'w') as f:
            config = {}
            for key in default_config.keys():
                config[key] = default_config[key]
            json.dump(config, f, indent=4)
        return config
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        config_path = "config_rtx4090.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    print(f"从 {config_path} 加载配置")
    
    # 更新默认配置
    for key in config.keys():
        if key not in ["distributed_config", "advanced_training"]:
            default_config[key] = config[key]
    
    # 处理嵌套的分布式配置
    if "distributed_config" in config:
        for key in config["distributed_config"]:
            default_config["distributed_config"][key] = config["distributed_config"][key]
    
    # 处理高级训练配置
    if "advanced_training" in config:
        for key in config["advanced_training"]:
            default_config["advanced_training"][key] = config["advanced_training"][key]
    
    # 提取分布式配置
    dist_config = default_config["distributed_config"]
    
    # 确保高级训练参数存在
    if "gradient_compression" not in default_config["advanced_training"]:
        default_config["advanced_training"]["gradient_compression"] = False
    if "adaptive_learning_rate" not in default_config["advanced_training"]:
        default_config["advanced_training"]["adaptive_learning_rate"] = False
    
    return (config_path, default_config['script'], default_config['seed'],
            default_config['mixed_precision'], default_config['report_to'], default_config['lr_warmup_steps'],
            default_config['output_dir'], default_config['save_name'], default_config['train_data_dir'],
            default_config['optimizer'], default_config['lr_scheduler'], default_config['learning_rate'],
            default_config['train_batch_size'], default_config['repeats'], default_config['gradient_accumulation_steps'],
            default_config['num_train_epochs'], default_config['save_model_epochs'], default_config['validation_epochs'],
            default_config['rank'], default_config['skip_epoch'],
            default_config['skip_step'], default_config['gradient_checkpointing'], default_config['validation_ratio'],
            default_config['pretrained_model_name_or_path'], default_config['model_path'], default_config['resume_from_checkpoint'],
            default_config['recreate_cache'], default_config['vae_path'], default_config['resolution'],
            default_config['use_debias'], default_config['snr_gamma'], default_config['caption_dropout'],
            default_config['cosine_restarts'], default_config['max_time_steps'],
            default_config['noise_offset'], default_config['blocks_to_swap'],
            default_config['weighting_scheme'], default_config['logit_mean'], default_config['logit_std'],
            dist_config['num_gpus'], dist_config['parallelize_mode'], dist_config['sharding_strategy'],
            dist_config['batch_size_per_gpu'], dist_config['enable_tf32'], dist_config['enable_flash_attention'],
            dist_config['distributed_evaluation'], dist_config['cpu_offload'], dist_config['compile_model'],
            dist_config['profile'], default_config['advanced_training']['progressive_batching'],
            default_config['advanced_training']['gradient_compression'], 
            default_config['advanced_training']['adaptive_learning_rate'])

# 加载默认配置
try:
    load_config("config_rtx4090.json")
except:
    print("使用默认配置")

def run_distributed(
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        vae_path,
        resolution,
        use_debias,
        snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        noise_offset,
        blocks_to_swap,
        weighting_scheme,
        logit_mean,
        logit_std,
        num_gpus,
        parallelize_mode,
        sharding_strategy,
        batch_size_per_gpu,
        enable_tf32,
        enable_flash_attention,
        distributed_evaluation,
        cpu_offload,
        compile_model,
        profile,
        progressive_batching,
        gradient_compression,
        adaptive_learning_rate
    ):
    # 检查VAE路径
    if vae_path is not None and vae_path != "":
        if not vae_path.endswith('.safetensors'):
            msg = "VAE文件需要以.safetensors结尾。建议使用来自 https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main 的fp16修复版VAE"
            gr.Warning(msg)
            return msg
    
    # 将训练参数保存到配置文件
    save_config(
        config_path, script, seed, mixed_precision, report_to, lr_warmup_steps,
        output_dir, save_name, train_data_dir, optimizer, lr_scheduler, learning_rate,
        train_batch_size, repeats, gradient_accumulation_steps, num_train_epochs,
        save_model_epochs, validation_epochs, rank, skip_epoch, skip_step,
        gradient_checkpointing, validation_ratio, pretrained_model_name_or_path,
        model_path, resume_from_checkpoint, recreate_cache, vae_path, resolution,
        use_debias, snr_gamma, caption_dropout, cosine_restarts, max_time_steps,
        noise_offset, blocks_to_swap, weighting_scheme, logit_mean, logit_std,
        num_gpus, parallelize_mode, sharding_strategy, batch_size_per_gpu,
        enable_tf32, enable_flash_attention, distributed_evaluation,
        cpu_offload, compile_model, profile, progressive_batching,
        gradient_compression, adaptive_learning_rate
    )
    
    # 构建运行分布式训练的命令行参数
    args = ["./run_distributed.sh"]
    
    # 添加分布式训练参数
    args.extend(["-g", str(num_gpus)])
    args.extend(["-s", script])
    args.extend(["-m", parallelize_mode])
    args.extend(["-x", mixed_precision])
    args.extend(["-b", str(batch_size_per_gpu)])
    
    if parallelize_mode == "fsdp":
        args.extend(["--sharding", sharding_strategy])
    
    if not enable_tf32:
        args.append("--no-tf32")
    
    if not enable_flash_attention:
        args.append("--no-flash-attention")
    
    if compile_model:
        args.append("--enable-compile")
    
    if profile:
        args.append("--profile")
    
    if cpu_offload:
        args.append("--cpu-offload")
        
    if gradient_compression:
        args.append("--grad-compression")
        
    if adaptive_learning_rate:
        args.append("--adaptive-lr")
    
    # 添加训练脚本的参数（使用--分隔）
    args.append("--")
    
    # 添加标准训练参数
    train_params = {
        "seed": seed,
        "mixed_precision": mixed_precision,
        "report_to": report_to,
        "lr_warmup_steps": lr_warmup_steps,
        "output_dir": output_dir,
        "save_name": save_name,
        "train_data_dir": train_data_dir,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "repeats": repeats,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_train_epochs,
        "save_model_epochs": save_model_epochs,
        "validation_epochs": validation_epochs,
        "rank": rank,
        "skip_epoch": skip_epoch,
        "skip_step": skip_step,
        "validation_ratio": validation_ratio,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "model_path": model_path,
        "resume_from_checkpoint": resume_from_checkpoint,
        "vae_path": vae_path,
        "resolution": resolution,
        "snr_gamma": snr_gamma,
        "caption_dropout": caption_dropout,
        "cosine_restarts": cosine_restarts,
        "max_time_steps": max_time_steps,
        "noise_offset": noise_offset,
        "blocks_to_swap": blocks_to_swap,
        "weighting_scheme": weighting_scheme,
        "logit_mean": logit_mean,
        "logit_std": logit_std
    }
    
    # 添加布尔参数
    boolean_params = {
        "gradient_checkpointing": gradient_checkpointing,
        "recreate_cache": recreate_cache,
        "use_debias": use_debias,
        "distributed_evaluation": distributed_evaluation
    }
    
    # 添加训练参数
    for key, value in train_params.items():
        if value is not None and value != "":
            args.append(f"--{key}")
            args.append(str(value))
    
    # 添加布尔参数
    for key, value in boolean_params.items():
        if value:
            args.append(f"--{key}")
    
    # 添加高级训练参数
    if progressive_batching:
        args.append("--use_progressive_batching")
    
    # 将命令行参数转换为字符串
    cmd_str = " ".join(args)
    print(f"运行命令: {cmd_str}")
    
    # 创建日志目录
    log_dir = "logs/distributed"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/training_{save_name}_{script}_{num_gpus}gpus.log"
    
    # 在终端中运行分布式训练命令
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        return f"命令执行成功:\n{cmd_str}\n\n输出:\n{result.stdout}\n\n错误:\n{result.stderr}\n\n日志保存在: {log_file}"
    except subprocess.CalledProcessError as e:
        return f"命令执行失败:\n{cmd_str}\n\n错误:\n{e.stderr}"
    

def check_gpu_info():
    try:
        # 检查GPU型号和数量
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"], 
                              check=True, capture_output=True, text=True)
        return result.stdout
    except:
        return "无法获取GPU信息，请确保NVIDIA驱动正确安装。"

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # RTX 4090 多GPU分布式训练界面
    ## Flux扩散模型分布式训练优化
    """
    )
    
    gpu_info = gr.Textbox(label="GPU信息", value=check_gpu_info())
    script = gr.Dropdown(label="训练脚本", value=default_config["script"], choices=default_config["script_choices"])
    
    with gr.Row(equal_height=True):
        config_path = gr.Textbox(scale=3, label="配置文件路径 (.json)", value=default_config["config_path"], placeholder="配置文件保存/加载路径")
        save_config_btn = gr.Button("保存配置", scale=1)
        load_config_btn = gr.Button("加载配置", scale=1)

    with gr.Accordion("基本设置", open=True):
        # 目录部分
        with gr.Row():
            output_dir = gr.Textbox(label="输出目录", value=default_config["output_dir"], placeholder="模型保存位置")
            save_name = gr.Textbox(label="保存名称", value=default_config["save_name"], placeholder="模型保存名称前缀")
        with gr.Row():
            pretrained_model_name_or_path = gr.Textbox(label="预训练模型路径", 
                value=default_config["pretrained_model_name_or_path"], 
                placeholder="仓库名称或包含diffusers模型结构的目录"
            )
            vae_path = gr.Textbox(label="FP16修复VAE路径", value=default_config["vae_path"], placeholder="单独的FP16修复VAE文件路径，需以.safetensors结尾")
        with gr.Row():
            model_path = gr.Textbox(label="模型路径", value=default_config["model_path"], placeholder="单一权重文件(如果不从官方权重训练)")
            resume_from_checkpoint = gr.Textbox(label="从检查点恢复", value=default_config["resume_from_checkpoint"], placeholder="从选定目录恢复LoRA权重")
        with gr.Row():
            train_data_dir = gr.Textbox(label="训练数据目录", value=default_config["train_data_dir"], placeholder="包含数据集的目录")
            report_to = gr.Dropdown(label="报告工具", value=default_config["report_to"], choices=["wandb", "tensorboard", "none"])
        
        # LoRA配置
        with gr.Row():
            rank = gr.Number(label="LoRA秩", value=default_config["rank"])
            train_batch_size = gr.Number(label="训练批次大小", value=default_config["train_batch_size"])
        with gr.Row():
            repeats = gr.Number(label="数据集重复次数", value=default_config["repeats"])
            gradient_accumulation_steps = gr.Number(label="梯度累积步数", value=default_config["gradient_accumulation_steps"])
            mixed_precision = gr.Radio(label="混合精度", value=default_config["mixed_precision"], choices=["fp16", "bf16"])
            gradient_checkpointing = gr.Checkbox(label="梯度检查点", value=default_config["gradient_checkpointing"])
        with gr.Row():
            optimizer = gr.Dropdown(label="优化器", value=default_config["optimizer"], choices=["adamw","prodigy","lion"])
            lr_scheduler = gr.Dropdown(label="学习率调度器", value=default_config["lr_scheduler"], 
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup", "warmup_cosine"])
            cosine_restarts = gr.Number(label="余弦重启次数", value=default_config["cosine_restarts"], info="仅适用于cosine_with_restarts调度器", minimum=1)
        with gr.Row():
            learning_rate = gr.Number(label="学习率", value=default_config["learning_rate"], info="推荐: 1e-4 或使用prodigy时为1")
            lr_warmup_steps = gr.Number(label="预热步数", value=default_config["lr_warmup_steps"])
            seed = gr.Number(label="随机种子", value=default_config["seed"])

    with gr.Accordion("训练控制", open=True):
        with gr.Row():
            num_train_epochs = gr.Number(label="训练总轮次", value=default_config["num_train_epochs"], info="训练的总轮次")
            save_model_epochs = gr.Number(label="保存模型轮次", value=default_config["save_model_epochs"], info="每隔多少轮保存一次检查点")
            validation_epochs = gr.Number(label="验证轮次", value=default_config["validation_epochs"], info="每隔多少轮进行一次验证")
        with gr.Row():
            skip_epoch = gr.Number(label="跳过验证轮次", value=default_config["skip_epoch"], info="跳过前多少轮的验证和检查点保存")
            skip_step = gr.Number(label="跳过验证步数", value=default_config["skip_step"], info="跳过前多少步的验证和检查点保存")
            validation_ratio = gr.Number(label="验证集比例", value=default_config["validation_ratio"], info="用于验证的数据集比例")
            
        with gr.Row():
            recreate_cache = gr.Checkbox(label="重新创建缓存", value=default_config["recreate_cache"])
            use_debias = gr.Checkbox(label="使用无偏估计", value=default_config["use_debias"])
            snr_gamma = gr.Number(label="SNR Gamma权重", value=default_config["snr_gamma"], info="推荐值: 5.0，用于计算SNR加权损失", maximum=10, minimum=0)
        with gr.Row():
            caption_dropout = gr.Number(label="标题丢弃率", value=default_config["caption_dropout"], info="文本提示丢弃率", maximum=1, minimum=0)
            max_time_steps = gr.Number(label="最大时间步数限制", value=default_config["max_time_steps"], info="训练时间步限制，0-1100", maximum=1100, minimum=0)
            resolution = gr.Dropdown(label="分辨率", value=default_config["resolution"], choices=default_config["resolution_choices"])
        with gr.Row():
            noise_offset = gr.Number(label="噪声偏移", value=default_config["noise_offset"], info="初始噪声偏移值", minimum=0)
            blocks_to_swap = gr.Number(label="交换块数量", value=default_config["blocks_to_swap"], info="根据VRAM大小建议10-20", minimum=0)

    with gr.Accordion("Flux特定参数", open=True):
        with gr.Row():
            weighting_scheme = gr.Dropdown(
                label="权重方案", 
                value=default_config["weighting_scheme"],
                choices=["logit_normal", "sigma_sqrt", "mode", "cosmap", "logit_snr"],
                info="Flux训练权重方案"
            )
        with gr.Row():
            logit_mean = gr.Number(label="Logit均值", value=default_config["logit_mean"], info="logit_normal方案的均值参数")
            logit_std = gr.Number(label="Logit标准差", value=default_config["logit_std"], info="logit_normal方案的标准差参数")

    with gr.Accordion("分布式训练设置", open=True):
        dist_config = default_config["distributed_config"]
        with gr.Row():
            num_gpus = gr.Number(label="GPU数量", value=dist_config["num_gpus"], minimum=1, info="用于训练的GPU数量")
            batch_size_per_gpu = gr.Number(label="每GPU批次大小", value=dist_config["batch_size_per_gpu"], minimum=1, info="每个GPU的批次大小")
        with gr.Row():
            parallelize_mode = gr.Radio(label="并行化模式", value=dist_config["parallelize_mode"], choices=["fsdp", "ddp"], info="FSDP支持更大模型，DDP更简单")
            sharding_strategy = gr.Dropdown(label="分片策略", value=dist_config["sharding_strategy"], choices=["full", "grad_op", "shard_grad_op"], info="仅当使用FSDP时有效")
        with gr.Row():
            enable_tf32 = gr.Checkbox(label="启用TF32", value=dist_config["enable_tf32"], info="在RTX 30/40系列GPU上提高性能")
            enable_flash_attention = gr.Checkbox(label="启用Flash Attention", value=dist_config["enable_flash_attention"], info="加速注意力计算")
            distributed_evaluation = gr.Checkbox(label="分布式评估", value=dist_config["distributed_evaluation"], info="加速验证过程")
        with gr.Row():
            cpu_offload = gr.Checkbox(label="CPU卸载", value=dist_config["cpu_offload"], info="使用CPU内存(会降低性能)")
            compile_model = gr.Checkbox(label="编译模型", value=dist_config["compile_model"], info="使用torch.compile()加速(需要PyTorch 2.0+)")
            profile = gr.Checkbox(label="性能分析", value=dist_config["profile"], info="启用性能分析(仅用于调试)")
        
    with gr.Accordion("高级训练策略", open=True):
        advanced_config = default_config["advanced_training"]
        with gr.Row():
            progressive_batching = gr.Checkbox(
                label="渐进式批次大小", 
                value=advanced_config["progressive_batching"], 
                info="从小批次开始，逐步增加批次大小"
            )
            gradient_compression = gr.Checkbox(
                label="梯度压缩", 
                value=advanced_config.get("gradient_compression", False), 
                info="减少通信开销，加速分布式训练"
            )
            adaptive_learning_rate = gr.Checkbox(
                label="自适应学习率", 
                value=advanced_config.get("adaptive_learning_rate", False), 
                info="根据每个GPU的梯度统计自动调整学习率"
            )

    inputs = [
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        vae_path,
        resolution,
        use_debias,
        snr_gamma,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        noise_offset,
        blocks_to_swap,
        weighting_scheme,
        logit_mean,
        logit_std,
        num_gpus,
        parallelize_mode,
        sharding_strategy,
        batch_size_per_gpu,
        enable_tf32,
        enable_flash_attention,
        distributed_evaluation,
        cpu_offload,
        compile_model,
        profile,
        progressive_batching,
        gradient_compression,
        adaptive_learning_rate
    ]
    
    output = gr.Textbox(label="输出信息", lines=10)
    run_btn = gr.Button("启动RTX 4090分布式训练", variant="primary")
    run_btn.click(fn=run_distributed, inputs=inputs, outputs=output, api_name="run_distributed")
    save_config_btn.click(fn=save_config, inputs=inputs)
    load_config_btn.click(fn=load_config, inputs=[config_path], outputs=inputs)

    gr.Markdown(
    """
    ## RTX 4090优化指南
    
    ### 硬件优势
    - RTX 4090拥有16,384个CUDA核心，24GB GDDR6X显存
    - 强大的第四代Tensor核心，TF32/BF16性能卓越
    - 支持CUDA 12和当前最新的深度学习库
    
    ### 性能优化建议
    1. **使用BF16混合精度**：在RTX 4090上，BF16比FP16提供更好的稳定性和相似的速度
    2. **启用TF32计算**：能提供接近FP32精度的同时大幅提升性能
    3. **Flash Attention 2**：为transformer注意力机制提供2-4倍加速
    4. **每GPU批次大小2-4**：RTX 4090显存充足，允许更大的批次
    5. **渐进式批次调度**：从小批次开始训练，稳定后增大
    6. **FSDP分片策略**：使用"full"获得最佳内存效率
    
    ### VRAM优化
    - 使用`blocks_to_swap=10`在CPU和GPU之间交换模型块
    - 预热阶段使用梯度检查点（gradient_checkpointing）
    - 使用混合精度训练减少内存需求
    
    ### 多GPU通信优化
    - 启用NCCL P2P优化提高通信效率
    - 使用梯度压缩减少通信开销
    - 设置CUDA_DEVICE_MAX_CONNECTIONS=1以提高稳定性
    
    详细技术文档请参阅[MULTI_GPU_README.md](MULTI_GPU_README.md)
    """
    )

if __name__ == "__main__":
    demo.launch() 