#!/bin/bash

echo "Starting GRPO training on a local 8-GPU machine..."

# --- 1. 设置环境变量 ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# 添加下面这两行来帮助 NCCL 进行通信
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- 2. 定义模型名称 ---
# 这是 train_grpo.py 唯一会读取的命令行参数
MODEL_NAME="/data/xiongtao/models/Qwen2.5-3B-Instruct" 
# 或者换成您需要的其他模型，例如 "Qwen/Qwen1.5-7B-Chat"

# --- 3. 定义训练参数 (现在这些参数会真正生效!) ---
# 您可以在这里轻松地调整超参数进行实验
TRAIN_ARGS=(
    --learning_rate 5e-6
    --lr_scheduler_type "constant_with_warmup"
    --per_device_train_batch_size 16  # <--- 修改为非常合理的小数值
    --gradient_accumulation_steps 4  # <--- 使用梯度累积来增大等效 batch size
    --num_train_epochs 1
    --gradient_checkpointing True
    --warmup_steps 10
    --logging_steps 10
    --save_steps 200
    --bf16 True
    --report_to "wandb"
    --remove_unused_columns False
    --num_generations 4
    --temperature 0.7
    --max_completion_length 512
    --max_prompt_length 512
    --disable_dropout True
    --loss_type "dr_grpo"
    --log_completions True
)

# --- 4. 使用 Accelerate 启动 8 卡训练 ---
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_grpo.py "$MODEL_NAME" "${TRAIN_ARGS[@]}"

echo "Training complete!"