#!/bin/bash
# Distributed training launcher script for T2ITrainer

# Default settings
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
TRAIN_SCRIPT="train_flux_lora_ui.py"
PORT=$(( ( RANDOM % 10000 )  + 10000 ))

# Help message
function show_help {
    echo "Usage: run_distributed.sh [options]"
    echo "Options:"
    echo "  -g, --num-gpus NUM       Number of GPUs to use (default: all available)"
    echo "  -s, --script SCRIPT      Training script to run (default: train_flux_lora_ui.py)"
    echo "  -p, --port PORT          Port for distributed communication (default: random)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Additional arguments after -- will be passed to the training script:"
    echo "  ./run_distributed.sh -g 2 -- --train_batch_size 4 --learning_rate 1e-4"
}

# Parse command line options
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
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# RTX 4090 specific optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))

echo "Starting distributed training with $NUM_GPUS GPUs"
echo "Training script: $TRAIN_SCRIPT"
echo "Port: $PORT"
echo "Additional arguments: $@"

# Check if Python and the required script exist
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ] && [ ! -f "train_distributed.py" ]; then
    echo "Error: Training script '$TRAIN_SCRIPT' not found and train_distributed.py not found"
    exit 1
fi

# Launch the distributed training
python3 train_distributed.py \
    --num_gpus $NUM_GPUS \
    --train_script $TRAIN_SCRIPT \
    --master_port $PORT \
    $@

echo "Distributed training complete" 