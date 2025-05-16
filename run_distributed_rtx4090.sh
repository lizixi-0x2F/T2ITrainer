#!/bin/bash
# RTX 4090优化的分布式训练启动脚本
# 针对RTX 4090 GPU进行特定优化

# 默认设置
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
TRAIN_SCRIPT="train_flux_lora_ui.py"
PORT=$(( ( RANDOM % 10000 )  + 10000 ))
PARALLELIZE_MODE="fsdp"  # fsdp或ddp
MIXED_PRECISION="bf16"   # bf16, fp16或no
BATCH_SIZE=2
ENABLE_TF32=true
ENABLE_FLASH_ATTENTION=true
ENABLE_COMPILE=false
PROFILE=false
SCHEDULER_TYPE="warmup_cosine"
OPTIMIZER_TYPE="adamw"
ENABLE_GRAD_COMPRESSION=false
ENABLE_ADAPTIVE_LR=false
BACKWARD_PREFETCH="BACKWARD_PRE"

# 帮助信息
function show_help {
    echo "用法: run_distributed_rtx4090.sh [选项]"
    echo "选项:"
    echo "  -g, --num-gpus NUM       使用的GPU数量 (默认: 全部可用)"
    echo "  -s, --script SCRIPT      要运行的训练脚本 (默认: train_flux_lora_ui.py)"
    echo "  -p, --port PORT          分布式通信端口 (默认: 随机)"
    echo "  -m, --mode MODE          并行化模式: fsdp或ddp (默认: fsdp)"
    echo "  -x, --mixed-precision MP 混合精度模式: bf16, fp16, no (默认: bf16)"
    echo "  -b, --batch-size BS      每GPU批次大小 (默认: 2)"
    echo "  --no-tf32                禁用TF32计算 (RTX 30/40系列GPU)"
    echo "  --no-flash-attention     禁用Flash Attention"
    echo "  --enable-compile         启用torch.compile()加速 (需要PyTorch 2.0+)"
    echo "  --profile                启用性能分析"
    echo "  --optimizer TYPE         优化器类型: adamw, lion, prodigy (默认: adamw)"
    echo "  --scheduler TYPE         学习率调度器: constant, linear, cosine, warmup_cosine (默认: warmup_cosine)"
    echo "  --config FILE            JSON配置文件路径"
    echo "  --grad-compression       启用梯度压缩以减少通信开销"
    echo "  --adaptive-lr            启用自适应学习率"
    echo "  --sharding STRATEGY      FSDP分片策略: full, grad_op, shard_grad_op (默认: full)"
    echo "  --backward-prefetch MODE FSDP反向预取模式: BACKWARD_PRE, BACKWARD_POST, NO_PREFETCH (默认: BACKWARD_PRE)"
    echo "  --cpu-offload            启用CPU卸载(不推荐用于RTX 4090)"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "附加参数 (-- 之后) 将传递给训练脚本:"
    echo "  ./run_distributed_rtx4090.sh -g 2 -b 4 -- --learning_rate 1e-4 --output_dir outputs/test"
}

# 解析命令行选项
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
        --grad-compression)
            ENABLE_GRAD_COMPRESSION=true
            shift
            ;;
        --adaptive-lr)
            ENABLE_ADAPTIVE_LR=true
            shift
            ;;
        --sharding)
            SHARDING_STRATEGY="$2"
            shift
            shift
            ;;
        --backward-prefetch)
            BACKWARD_PREFETCH="$2"
            shift
            shift
            ;;
        --cpu-offload)
            CPU_OFFLOAD=true
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

# RTX 4090特定优化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))

# 启用消费级GPU的更好性能
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1

# 检查是否有RTX 30/40系列GPU并支持Flash Attention 2
if [[ "$ENABLE_FLASH_ATTENTION" == "true" ]]; then
    # 检查是否安装
    if python -c "import torch; import flash_attn" &>/dev/null; then
        echo "检测到Flash Attention 2，启用优化..."
        export CUDA_LAUNCH_BLOCKING=0
    else
        echo "警告: 未检测到Flash Attention。为获得最佳性能，请使用以下命令安装:"
        echo "pip install flash-attn --no-build-isolation"
        ENABLE_FLASH_ATTENTION=false
    fi
fi

echo "🚀 启动RTX 4090优化的分布式训练，使用 $NUM_GPUS 个GPU"
echo "🧠 训练脚本: $TRAIN_SCRIPT"
echo "🔌 端口: $PORT"
echo "💪 并行化模式: $PARALLELIZE_MODE"
echo "🎯 混合精度: $MIXED_PRECISION"
echo "📦 每GPU批次大小: $BATCH_SIZE"
echo "⚙️ 优化器: $OPTIMIZER_TYPE"
echo "📉 调度器: $SCHEDULER_TYPE"
echo "🔍 性能分析: $PROFILE"

# 检查Python和所需脚本是否存在
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ] && [ ! -f "train_distributed_rtx4090.py" ]; then
    echo "错误: 训练脚本 '$TRAIN_SCRIPT' 和 train_distributed_rtx4090.py 都未找到"
    exit 1
fi

# 准备命令行参数
CMD_ARGS=()
CMD_ARGS+=(--num_gpus "$NUM_GPUS")
CMD_ARGS+=(--train_script "$TRAIN_SCRIPT")
CMD_ARGS+=(--master_port "$PORT")
CMD_ARGS+=(--parallelize_mode "$PARALLELIZE_MODE")
CMD_ARGS+=(--mixed_precision "$MIXED_PRECISION")
CMD_ARGS+=(--batch_size_per_gpu "$BATCH_SIZE")

# 添加FSDP特定参数
if [[ "$PARALLELIZE_MODE" == "fsdp" ]]; then
    if [[ -n "$SHARDING_STRATEGY" ]]; then
        CMD_ARGS+=(--sharding_strategy "$SHARDING_STRATEGY")
    fi
    CMD_ARGS+=(--backward_prefetch "$BACKWARD_PREFETCH")
fi

# 添加可选参数
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

if [[ "$ENABLE_GRAD_COMPRESSION" == "true" ]]; then
    CMD_ARGS+=(--grad_compression)
fi

if [[ "$ENABLE_ADAPTIVE_LR" == "true" ]]; then
    CMD_ARGS+=(--adaptive_lr)
fi

if [[ "$CPU_OFFLOAD" == "true" ]]; then
    CMD_ARGS+=(--cpu_offload)
fi

# 添加用户参数
CMD_ARGS+=("${ADDITIONAL_ARGS[@]}")

# 添加训练脚本参数
for arg in "$@"; do
    CMD_ARGS+=("$arg")
done

# 添加优化器和调度器信息
if [[ -n "$OPTIMIZER_TYPE" ]]; then
    CMD_ARGS+=(--optimizer "$OPTIMIZER_TYPE")
fi

if [[ -n "$SCHEDULER_TYPE" ]]; then
    CMD_ARGS+=(--lr_scheduler "$SCHEDULER_TYPE")
fi

# 创建日志目录
LOG_DIR="logs/rtx4090"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# 启动分布式训练
echo "完整命令: python3 train_distributed_rtx4090.py ${CMD_ARGS[*]}"
echo "日志将保存到: $LOG_FILE"

# 启用资源监控
if command -v nvidia-smi &> /dev/null; then
    # 在后台每5秒记录一次GPU使用情况
    (while true; do 
        echo "======== $(date) ========" >> "$LOG_DIR/gpu_stats_${TIMESTAMP}.log"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv >> "$LOG_DIR/gpu_stats_${TIMESTAMP}.log"
        sleep 5
    done) &
    MONITOR_PID=$!

    # 在脚本退出时停止监控
    trap "kill $MONITOR_PID" EXIT
fi

# 运行训练
python3 train_distributed_rtx4090.py "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo "RTX 4090优化的分布式训练完成！"
echo "查看结果日志: $LOG_FILE"
echo "查看GPU使用统计: $LOG_DIR/gpu_stats_${TIMESTAMP}.log" 