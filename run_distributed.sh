#!/bin/bash
# Distributed training launcher script for T2ITrainer
# Optimized for RTX 4090 GPUs

# Default settings
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
TRAIN_SCRIPT="train_flux_lora_ui.py"
PORT=$(( ( RANDOM % 10000 )  + 10000 ))
PARALLELIZE_MODE="fsdp"  # fsdp or ddp
MIXED_PRECISION="bf16"  # bf16, fp16, or no
BATCH_SIZE=2
ENABLE_TF32=true
ENABLE_FLASH_ATTENTION=true
ENABLE_COMPILE=false
PROFILE=false
SCHEDULER_TYPE="warmup_cosine"
OPTIMIZER_TYPE="adamw"

# Help message
function show_help {
    echo "使用方法: run_distributed.sh [options]"
    echo "选项:"
    echo "  -g, --num-gpus NUM       使用的GPU数量 (默认: 全部可用)"
    echo "  -s, --script SCRIPT      要运行的训练脚本 (默认: train_flux_lora_ui.py)"
    echo "  -p, --port PORT          分布式通信的端口 (默认: 随机)"
    echo "  -m, --mode MODE          并行化模式: fsdp 或 ddp (默认: fsdp)"
    echo "  -x, --mixed-precision MP 混合精度模式: bf16, fp16, no (默认: bf16)"
    echo "  -b, --batch-size BS      每GPU的批次大小 (默认: 2)"
    echo "  --no-tf32                禁用 TF32 计算 (RTX 30/40系列GPU)"
    echo "  --no-flash-attention     禁用 Flash Attention"
    echo "  --enable-compile         启用 torch.compile() 加速 (需要 PyTorch 2.0+)"
    echo "  --profile                启用性能分析"
    echo "  --optimizer TYPE         优化器类型: adamw, lion, prodigy (默认: adamw)"
    echo "  --scheduler TYPE         学习率调度器: constant, linear, cosine, warmup_cosine (默认: warmup_cosine)"
    echo "  --config FILE            JSON配置文件路径"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "附加参数 (-- 之后) 将传递给训练脚本:"
    echo "  ./run_distributed.sh -g 2 -b 4 -- --learning_rate 1e-4 --output_dir outputs/test"
}

# Parse command line options
ADDITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -g|--num-gpus)
            NUM_GPUS="$2"
            shift
            shift
            ;;
        -s|--script)
            TRAIN_SCRIPT="$2"
            shift
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift
            shift
            ;;
        -m|--mode)
            PARALLELIZE_MODE="$2"
            shift
            shift
            ;;
        -x|--mixed-precision)
            MIXED_PRECISION="$2"
            shift
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --no-tf32)
            ENABLE_TF32=false
            shift
            ;;
        --no-flash-attention)
            ENABLE_FLASH_ATTENTION=false
            shift
            ;;
        --enable-compile)
            ENABLE_COMPILE=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --optimizer)
            OPTIMIZER_TYPE="$2"
            shift
            shift
            ;;
        --scheduler)
            SCHEDULER_TYPE="$2"
            shift
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            ADDITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# RTX 4090 specific optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))

# Enable better performance on consumer GPUs
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=lo

# Check if we have RTX 30/40 series GPU and support for Flash Attention 2
if [[ "$ENABLE_FLASH_ATTENTION" == "true" ]]; then
    # Check if installed
    if python -c "import torch; import flash_attn" &>/dev/null; then
        echo "Flash Attention 2 detected, enabling optimizations..."
        export CUDA_LAUNCH_BLOCKING=0
    else
        echo "Warning: Flash Attention not detected. For best performance, install with:"
        echo "pip install flash-attn --no-build-isolation"
        ENABLE_FLASH_ATTENTION=false
    fi
fi

echo "🚀 启动分布式训练，使用 $NUM_GPUS 个GPU"
echo "🧠 训练脚本: $TRAIN_SCRIPT"
echo "🔌 端口: $PORT"
echo "💪 并行化模式: $PARALLELIZE_MODE"
echo "🎯 混合精度: $MIXED_PRECISION"
echo "📦 每GPU批次大小: $BATCH_SIZE"
echo "⚙️ 优化器: $OPTIMIZER_TYPE"
echo "📉 调度器: $SCHEDULER_TYPE"
echo "🔍 性能分析: $PROFILE"

# Check if Python and the required script exist
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ] && [ ! -f "train_distributed.py" ]; then
    echo "错误: 训练脚本 '$TRAIN_SCRIPT' 和 train_distributed.py 都未找到"
    exit 1
fi

# Prepare command line arguments
CMD_ARGS=()
CMD_ARGS+=(--num_gpus "$NUM_GPUS")
CMD_ARGS+=(--train_script "$TRAIN_SCRIPT")
CMD_ARGS+=(--master_port "$PORT")
CMD_ARGS+=(--parallelize_mode "$PARALLELIZE_MODE")
CMD_ARGS+=(--mixed_precision "$MIXED_PRECISION")
CMD_ARGS+=(--batch_size_per_gpu "$BATCH_SIZE")

# Add optional arguments
if [[ "$ENABLE_TF32" == "true" ]]; then
    CMD_ARGS+=(--enable_tf32)
fi

if [[ "$ENABLE_COMPILE" == "true" ]]; then
    CMD_ARGS+=(--compile_model)
fi

if [[ "$PROFILE" == "true" ]]; then
    CMD_ARGS+=(--profile)
fi

if [[ -n "$CONFIG_FILE" ]]; then
    CMD_ARGS+=(--config "$CONFIG_FILE")
fi

# Add user arguments
CMD_ARGS+=("${ADDITIONAL_ARGS[@]}")

# Add training script arguments
for arg in "$@"; do
    CMD_ARGS+=("$arg")
done

# Add optimizer and scheduler information
if [[ -n "$OPTIMIZER_TYPE" ]]; then
    CMD_ARGS+=(--optimizer "$OPTIMIZER_TYPE")
fi

if [[ -n "$SCHEDULER_TYPE" ]]; then
    CMD_ARGS+=(--lr_scheduler "$SCHEDULER_TYPE")
fi

# Create log directory
LOG_DIR="logs/distributed"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

# Launch the distributed training
echo "完整命令: python3 train_distributed.py ${CMD_ARGS[*]}"
echo "日志将保存到: $LOG_FILE"

python3 train_distributed.py "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "分布式训练完成" 