#!/bin/bash

echo "Step 1. Setup venv"
echo "Setting up virtual environment automatically"
python3 -m venv venv || python -m venv venv
# Activate the virtual environment
source venv/bin/activate

echo "Step 2. Upgrade pip"
python -m pip install --upgrade pip

echo "Step 3. Install torch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Step 4. Install other dependencies from requirements.txt"
pip install -r requirements.txt

# 安装分布式训练依赖
echo "Step 5. Install distributed training dependencies"
pip install accelerate>=0.20.0
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install deepspeed bitsandbytes
pip install pytorch-fid

echo "Setup complete!"