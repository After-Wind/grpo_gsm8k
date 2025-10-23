import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import wandb
from math_utils import extract_answer

def main():
    # --- 1. 使用 HfArgumentParser 来解析命令行参数 ---
    # 这使得所有 GRPOConfig 的参数都可以从命令行传入，例如 --learning_rate 1e-5
    parser = HfArgumentParser(GRPOConfig)

    # 我们需要手动处理 model_name，所以让 parser 解析剩余的参数
    # sys.argv[1] 是 model_name, 之后的是训练参数
    if len(sys.argv) > 2:
        training_args = parser.parse_args_into_dataclasses(args=sys.argv[2:])[0]
    else:
        # 如果没有提供额外参数，则使用默认配置
        training_args = parser.parse_args_into_dataclasses(args=[])[0]

    # --- 2. 初始化 Accelerator ---
    accelerator = Accelerator()

    # --- 3. 模型和 Tokenizer 设置 ---
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"

    # 动态设置 output_dir
    training_args.output_dir = f"./grpo_{model_name.replace('/', '-')}_gsm8k"

    # 加载模型时，移除了 device_map 参数！
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4. WandB 初始化 ---
    if accelerator.is_main_process:
        wandb.init(project=f"{model_name.replace('/', '-')}-gsm8k-grpo", config=training_args)

    # --- 5. 数据加载和处理 ---
    train_data = []
    with open("data/gsm8k_train.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            train_data.append({
                "prompt": item["prompt"],
                "numerical_answer": item["numerical_answer"]
            })
    dataset = Dataset.from_list(train_data)

    # --- 6. Reward 函数定义 (保持不变) ---
    def accuracy_reward(prompts, completions, numerical_answer, **kwargs):
        rewards = []
        for completion, gt in zip(completions, numerical_answer):
            predicted = extract_answer(completion)
            if predicted == gt:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        return rewards

    def format_reward(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            if "####" in completion:
                rewards.append(0.2)
            else:
                rewards.append(-0.2)
        return rewards

    def length_reward(prompts, completions, **kwargs):
        rewards = []
        for completion in completions:
            if len(completion) < 30:
                rewards.append(-0.5)
            else:
                rewards.append(0.0)
        return rewards

    # --- 7. 创建 Trainer ---
    # training_args 现在是从命令行动态传入的
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[format_reward, accuracy_reward, length_reward],
        train_dataset=dataset
    )

    # --- 8. 开始训练 ---
    trainer.train()

    # --- 9. 保存最终模型 ---
    if accelerator.is_main_process:
        final_model_path = f"{training_args.output_dir}_final"
        trainer.save_model(final_model_path)
        print(f"Training complete! Model saved to {final_model_path}")

if __name__ == "__main__":
    main()