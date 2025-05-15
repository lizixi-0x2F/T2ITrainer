# T2ITrainer 多GPU训练指南

本文档介绍如何使用T2ITrainer的多GPU分布式训练功能，该功能专为RTX 4090显卡进行了优化。

## 功能特点

- **FSDP集成**: 使用PyTorch的全分片数据并行(FSDP)，支持更大的模型和批次大小
- **高效混合精度训练**: 支持bf16、fp16和TF32，充分利用RTX 4090的张量核心
- **优化的内存管理**: 针对RTX 4090的24GB显存进行了专门优化
- **高级学习率调度**: 支持预热、渐进式批次大小、多阶段训练等
- **分布式评估**: 加速模型验证和生成过程
- **易用的启动脚本**: 简化多GPU训练的配置和启动

## 安装依赖

确保已安装以下依赖:

```bash
pip install -r requirements.txt
pip install flash-attn  # 可选，但建议安装以提高性能
```

## 快速开始

### 使用默认配置启动分布式训练

```bash
# 使用所有可用GPU
./run_distributed.sh

# 指定使用2个GPU
./run_distributed.sh -g 2
```

### 指定训练脚本和批次大小

```bash
./run_distributed.sh -s train_flux_lora_ui.py -b 4
```

### 使用高级选项

```bash
./run_distributed.sh -g 2 -m fsdp -x bf16 -b 4 --enable-compile
```

### 传递训练参数

```bash
./run_distributed.sh -- --learning_rate 1e-4 --output_dir outputs/my_model
```

## 配置文件

您可以使用JSON配置文件来设置更多高级选项:

```bash
./run_distributed.sh --config configs/rtx4090_config.json
```

示例配置文件位于`configs/rtx4090_config.json`，包含了针对RTX 4090优化的各种设置。

## 参数说明

### 启动脚本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-g, --num-gpus` | 使用的GPU数量 | 所有可用GPU |
| `-s, --script` | 要运行的训练脚本 | train_flux_lora_ui.py |
| `-m, --mode` | 并行化模式(fsdp或ddp) | fsdp |
| `-x, --mixed-precision` | 混合精度模式(bf16, fp16, no) | bf16 |
| `-b, --batch-size` | 每GPU的批次大小 | 2 |
| `--no-tf32` | 禁用TF32计算 | 启用 |
| `--enable-compile` | 启用torch.compile()加速 | 禁用 |
| `--profile` | 启用性能分析 | 禁用 |
| `--config` | JSON配置文件路径 | 无 |

## 高级配置

### 学习率调度

系统支持多种学习率调度策略：

1. **预热余弦退火**: 初始预热阶段后应用余弦退火
2. **渐进式批次大小**: 从小批次开始逐步增加批次大小，同时调整学习率
3. **多阶段调度**: 在不同训练阶段应用不同的学习率策略

可以在配置文件中设置这些选项：

```json
"scheduler": {
    "type": "warmup_cosine",
    "warmup_steps": 100,
    "min_lr": 1e-6
}
```

### FSDP优化

通过FSDP设置提高训练效率：

```json
"distribution_config": {
    "parallelize_mode": "fsdp",
    "sharding_strategy": "full",
    "mixed_precision": "bf16"
}
```

### RTX 4090专用优化

```json
"rtx4090_optimizations": {
    "enable_tf32": true,
    "flash_attention": true,
    "channels_last_memory_format": true
}
```

## 分布式评估

使用`distributed_evaluation.py`中的工具进行分布式评估：

```python
from utils.distributed_evaluation import distributed_image_generation, evaluate_fid_distributed

# 分布式生成图像
images, gen_time = distributed_image_generation(
    pipeline=my_pipeline,
    prompts=prompts_list,
    batch_size=4
)

# 分布式计算FID分数
fid_score = evaluate_fid_distributed(
    real_images_dir="path/to/real/images",
    generated_images=images
)
```

## 性能优化提示

1. **批次大小优化**: RTX 4090通常可以处理较大的批次，推荐从每GPU 2-4开始测试
2. **混合精度选择**: 对于大多数扩散模型，bf16是性能和稳定性的最佳平衡
3. **梯度累积**: 如果遇到内存限制，考虑使用梯度累积而不是降低批次大小
4. **使用Flash Attention**: 安装flash-attn可以获得显著的性能提升
5. **启用TF32**: 对于RTX 4090，TF32提供了接近FP32的精度但性能接近FP16
6. **使用编译功能**: 在PyTorch 2.0+上，开启`--enable-compile`可以显著加速训练

## 故障排除

### 显存不足

如果遇到"CUDA out of memory"错误：

1. 降低每GPU批次大小
2. 启用梯度累积
3. 选择更高效的并行化策略(例如从DDP改为FSDP)
4. 启用CPU卸载(对速度有影响)

### 进程间通信问题

如果遇到"NCCL"相关错误：

1. 设置环境变量`export NCCL_DEBUG=INFO`查看详细错误信息
2. 尝试使用不同的通信后端: `--backend gloo`
3. 确保所有GPU位于同一PCIe总线上

### 训练不稳定

如果训练过程不稳定：

1. 降低学习率
2. 增加预热步数
3. 从FP16切换到BF16或FP32
4. 禁用自动混合精度，改用手动混合精度

## 示例用例

### 训练Flux Lora

```bash
./run_distributed.sh -g 2 -b 2 -- \
  --learning_rate 1e-4 \
  --output_dir outputs/flux_lora \
  --num_train_epochs 10
```

### 使用高级调度

```bash
./run_distributed.sh --config configs/rtx4090_config.json -g 2 -- \
  --output_dir outputs/advanced_training
```

## 参考

- [PyTorch FSDP文档](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [混合精度训练指南](https://pytorch.org/docs/stable/amp.html)
- [分布式训练最佳实践](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 