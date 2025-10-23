#!/bin/bash

echo "====================================================================="
echo "Starting GRPO fine-tuning with Accelerate on 8 GPUs..."
echo "====================================================================="

# --- 1. 设置环境变量 (优化显存和多卡通信) ---
# 帮助PyTorch更好地管理显存碎片
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 如果已安装Flash Attention的预编译包，跳过即时编译以加快启动速度
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# 在某些环境中，以下两行有助于解决NCCL（多GPU通信库）的潜在问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- 2. 定义模型路径 ---
# 这是 train_grpo.py 脚本需要读取的第一个也是唯一的固定位置参数。
# 请将其修改为您自己的模型路径，可以是本地路径或Hugging Face Hub上的模型名称。
MODEL_NAME="/data/xiongtao/models/Qwen2.5-3B-Instruct" 
# 例如: "meta-llama/Llama-2-7b-chat-hf" 或 "/data/models/My-Llama-Finetune"

# --- 3. 定义训练参数 ---
# 使用一个数组来清晰地管理所有训练参数，这些参数将被传递给 train_grpo.py。
# 您可以在这里轻松地调整超参数进行实验。
TRAIN_ARGS=(
    --learning_rate 5e-6
    --lr_scheduler_type "cosine"
    --per_device_train_batch_size 16      # [重要] GRPO训练非常耗费显存，建议每个GPU的批次大小设为1或2。
    --gradient_accumulation_steps 8      # [重要] 通过梯度累积来增大有效批次大小。有效批次大小 = 8(GPUs) * 1 * 8 = 64。
    --num_train_epochs 1
    --gradient_checkpointing True        # 开启梯度检查点，用计算时间换取大量显存空间。
    --warmup_steps 15                    # 预热步数
    --logging_steps 1                    # 每隔5步记录一次日志
    --save_steps 200                     # 每隔200步保存一次模型检查点
    --bf16 True                          # 使用bfloat16混合精度进行训练
    --report_to "wandb"                  # 将训练指标上报到 Weights & Biases
    --remove_unused_columns False        # 保留数据集中的所有列
    --num_generations 4                  # 为每个Prompt生成4个候选答案进行比较
    --temperature 0.7                    # 生成时的采样温度，增加多样性
    --max_completion_length 512          # 生成答案的最大长度
    --max_prompt_length 512              # 输入提示的最大长度
    --disable_dropout True               # 在微调期间禁用dropout
    --loss_type "dr_grpo"                # 使用GRPO的特定损失函数
    --log_completions True               # 将生成的答案记录到WandB，便于分析
)

# --- 4. 使用 Accelerate 启动 8 卡分布式训练 ---
# --multi_gpu: 明确告知使用多GPU模式
# --num_processes: 指定使用的GPU数量
# --num_machines: 指定使用的机器数量（这里是单机多卡）
# --mixed_precision: 指定混合精度类型，与 --bf16 True 保持一致
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    train_grpo_reason.py "$MODEL_NAME" "${TRAIN_ARGS[@]}"

echo "====================================================================="
echo "Training script has finished."
echo "====================================================================="


# nohup bash run_reason.sh > run_reason.log 2>&1 &