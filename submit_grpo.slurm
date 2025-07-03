#!/bin/bash
#SBATCH --job-name=grpo-llama2-7b-no-decay
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=hsu_gpu_priority
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_grpo.log

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Get GPU allocation from SLURM
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

echo "Starting GRPO training..."
echo "Using all $NUM_GPUS GPUs for training ($CUDA_VISIBLE_DEVICES)"

# Create Accelerate config on the fly
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '$CUDA_VISIBLE_DEVICES'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU: run without Accelerate
    srun python train_grpo.py "${TRAIN_ARGS[@]}"
else
    srun accelerate launch \
        --multi-gpu \
        --num-processes "${NUM_GPUS}" \
        --num_machines 1 \
        --mixed_precision bf16 \
        --gpu_ids all \
        --machine_rank 0 \
        --main_training_function main \
        --rdzv_backend static \
        --same_network \
        train_grpo.py "${TRAIN_ARGS[@]}"
fi

echo "Training complete!"
