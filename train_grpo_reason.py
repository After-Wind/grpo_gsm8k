import json
import sys
import torch
import re  # 导入正则表达式库
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import Dataset, load_dataset # 导入load_dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator

# --- 1. WandB 登录与初始化 ---
# 建议将您的WandB API密钥设置为环境变量 WANDB_API_KEY，然后可以直接调用 wandb.login()
# 或者使用代码登录: wandb.login(key="YOUR_WANDB_API_KEY")
try:
    wandb.login()
    print("WandB login successful.")
except Exception as e:
    print(f"WandB login failed. Please ensure you have set the WANDB_API_KEY environment variable. Error: {e}")


# --- 2. 新增：系统提示词和数据处理函数 (来自代码B) ---
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 提取<answer>标签之间的内容
def extract_xml_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# 提取原始gsm8k数据集中的答案部分
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# 获取gsm8k数据集并处理成对话格式
def get_gsm8k_and_format_prompt(split: str = "train") -> Dataset:
    """
    加载 openai/gsm8k 数据集，并将其格式化为包含系统提示词的对话格式。
    """
    data = load_dataset('openai/gsm8k', 'main', split=split)
    
    def format_example(example):
        # 将问题包装在指定的对话格式中
        prompt_messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': example['question']}
        ]
        # 提取标准答案
        answer = extract_hash_answer(example['answer'])
        return {'prompt': prompt_messages, 'answer': answer}

    # 使用.map进行转换并移除空答案的样本
    data = data.map(format_example)
    data = data.filter(lambda x: x['answer'] is not None)
    return data


# --- 3. 新增：来自代码B的奖励函数 (已修正) ---

# 辅助函数：从 GRPOTrainer 复杂的 completions 结构中提取出纯文本列表
def extract_completion_texts(completions: list) -> list[str]:
    """从 GRPOTrainer 的 completions 数据结构中提取出纯文本字符串列表。"""
    texts = []
    for comp in completions:
        # 兼容 comp 是列表且包含字典的情况
        if isinstance(comp, list) and comp and 'content' in comp[0]:
            texts.append(comp[0]['content'])
        # 兼容 comp 本身就是字符串的简单情况（增加代码健壮性）
        elif isinstance(comp, str):
            texts.append(comp)
    return texts

# 奖励函数 1: 检查最终答案的正确性
def correctness_reward_func(prompts, completions, answer, **kwargs):
    completions_text = extract_completion_texts(completions) # <--- 修正
    extracted_responses = [extract_xml_answer(c) for c in completions_text]
    # 奖励为2.0表示正确，0.0表示错误
    return [2.0 if resp == ans else 0.0 for resp, ans in zip(extracted_responses, answer)]

# 奖励函数 2: 检查答案是否为整数
def int_reward_func(completions, **kwargs):
    completions_text = extract_completion_texts(completions) # <--- 修正
    extracted_responses = [extract_xml_answer(c) for c in completions_text]
    # 如果是纯数字，奖励0.5
    return [0.5 if resp.isdigit() else 0.0 for resp in extracted_responses]

# 奖励函数 3: 严格检查XML格式
def strict_format_reward_func(completions, **kwargs):
    completions_text = extract_completion_texts(completions) # <--- 修正
    pattern = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s*$", re.DOTALL)
    # 格式完全匹配，奖励0.5
    return [0.5 if pattern.match(c) else 0.0 for c in completions_text]

# 奖励函数 4: 宽松检查XML格式
def soft_format_reward_func(completions, **kwargs):
    completions_text = extract_completion_texts(completions) # <--- 修正
    pattern = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
    # 格式基本匹配，奖励0.5
    return [0.5 if re.search(pattern, c) else 0.0 for c in completions_text]

# 奖励函数 5: 细致地检查和统计XML标签使用情况
def count_xml(text: str) -> float:
    count = 0.0
    if text.count("<reasoning>") == 1: count += 0.125
    if text.count("</reasoning>") == 1: count += 0.125
    if text.count("<answer>") == 1: count += 0.125
    if text.count("</answer>") == 1: count += 0.125
    # 对</answer>标签后出现多余字符进行惩罚
    if "</answer>" in text:
        trailing_text = text.split("</answer>")[-1]
        count -= len(trailing_text.strip()) * 0.01 
    return count

def xmlcount_reward_func(completions, **kwargs):
    completions_text = extract_completion_texts(completions) # <--- 修正
    return [count_xml(c) for c in completions_text]

def main():
    # --- 4. 使用 HfArgumentParser 来解析命令行参数 ---
    parser = HfArgumentParser(GRPOConfig)
    if len(sys.argv) > 2:
        training_args = parser.parse_args_into_dataclasses(args=sys.argv[2:])[0]
    else:
        # 如果没有提供额外参数，则使用默认配置
        training_args = GRPOConfig(
            output_dir="./grpo_llama2-7b_gsm8k", # 默认输出目录
            learning_rate=1e-5,
            logging_steps=10,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            report_to="wandb", # 报告给wandb
        )

    # --- 5. 初始化 Accelerator ---
    accelerator = Accelerator()

    # --- 6. 模型和 Tokenizer 设置 ---
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"
    
    # 动态设置 output_dir 和 run_name
    sanitized_model_name = model_name.replace('/', '-')
    training_args.output_dir = f"./grpo_{sanitized_model_name}_gsm8k"
    training_args.run_name = f"grpo_{sanitized_model_name}_gsm8k"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 7. WandB 初始化 (主进程) ---
    if accelerator.is_main_process:
        wandb.init(project=f"GRPO-gsm8k-finetuning", name=training_args.run_name, config=training_args)

    # --- 8. 数据加载和处理 ---
    # 使用新的数据加载函数
    dataset = get_gsm8k_and_format_prompt(split="train")
    # 为了演示，可以只取一小部分数据
    # dataset = dataset.select(range(100))

    # --- 9. 创建 Trainer ---
    # GRPOTrainer现在使用来自代码B的奖励函数列表
    trainer = GRPOTrainer(
        model=model,
        # tokenizer=tokenizer, # 传入tokenizer以处理对话模板
        args=training_args,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        train_dataset=dataset,
        # 指定数据集中包含prompt和answer的列名
        # dataset_prompt_field="prompt",
        # dataset_answer_field="answer"
    )

    # --- 10. 开始训练 ---
    trainer.train()

    # --- 11. 保存最终模型 ---
    if accelerator.is_main_process:
        final_model_path = f"{training_args.output_dir}_final"
        trainer.save_model(final_model_path)
        print(f"Training complete! Model saved to {final_model_path}")
        wandb.finish()

if __name__ == "__main__":
    main()