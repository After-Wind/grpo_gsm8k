# import json
# import os
# import sys
# from vllm import LLM, SamplingParams
# from math_utils import extract_answer

# # Model to evaluate
# model_name = sys.argv[1] if len(sys.argv) > 1 else "/data/xiongtao/models/Qwen2.5-3B-Instruct"

# print(f"Loading model: {model_name}")
# llm = LLM(
#     model=model_name,
#     tensor_parallel_size=1,
#     dtype="auto",
#     trust_remote_code=True,
#     max_model_len=1024,
#     gpu_memory_utilization=0.95,
#     enforce_eager=False,  # Use Flash Attention 2
# )

# # Load evaluation data
# eval_data = []
# with open("data/gsm8k_500.jsonl", "r") as f:
#     for line in f:
#         eval_data.append(json.loads(line))

# # Prepare prompts
# prompts = [item["prompt"] for item in eval_data]

# # Generate completions
# sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=None)

# print(f"Generating completions for {len(prompts)} problems...")
# outputs = llm.generate(prompts, sampling_params)


# # Evaluate
# correct = 0
# incorrect_completions = []

# for i, output in enumerate(outputs):
#     generated_text = output.outputs[0].text
#     predicted_answer = extract_answer(generated_text)
#     ground_truth = eval_data[i]["numerical_answer"]

#     if predicted_answer == ground_truth:
#         correct += 1
#     else:
#         incorrect_completions.append({
#             "generated_text": generated_text,
#             "predicted_answer": predicted_answer,
#             "ground_truth": ground_truth
#         })

# accuracy = correct / len(eval_data)
# print(f"\nAccuracy: {correct}/{len(eval_data)} = {accuracy:.2%}")
# # Save incorrect completions
# os.makedirs("evals", exist_ok=True)
# with open(f"evals/{model_name.replace('/', '-')}_incorrect_completions.json", "w") as f:
#     json.dump(incorrect_completions, f, indent=2)
# print(f"Saved {len(incorrect_completions)} incorrect completions to evals/{model_name.replace('/', '-')}_incorrect_completions.json")


import json
import os
import argparse
from vllm import LLM, SamplingParams
from math_utils import extract_answer

# --- 1. 定义SYSTEM_PROMPT (与训练时保持一致) ---
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def main():
    # --- 2. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Evaluate a large language model on the GSM8K dataset.")
    parser.add_argument(
        "model_name",
        type=str,
        help="The path to the model to evaluate, e.g., /path/to/your/finetuned-model"
    )
    parser.add_argument(
        "--reason",
        action="store_true",
        help="If specified, applies the reasoning system prompt to the prompts before evaluation."
    )
    args = parser.parse_args()

    # --- 3. 加载模型 ---
    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=1,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
    )

    # +++ 关键修改：从vLLM实例中获取tokenizer +++
    tokenizer = llm.get_tokenizer()

    # --- 4. 加载评估数据 ---
    eval_data = []
    with open("data/gsm8k_500.jsonl", "r") as f:
        for line in f:
            eval_data.append(json.loads(line))

    # --- 5. 根据 --reason 标志准备 Prompts ---
    if args.reason:
        print("Applying 'reason' system prompt and chat template to evaluation data...")
        prompts_to_send = []
        for item in eval_data:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["prompt"]}
            ]
            # +++ 关键修改：使用tokenizer将对话历史转换为格式化的字符串 +++
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False, # 必须为 False，我们想要字符串而不是token IDs
                add_generation_prompt=True # 添加引导模型开始回答的提示
            )
            prompts_to_send.append(formatted_prompt)
    else:
        print("Using raw prompts for evaluation (base model evaluation)...")
        prompts_to_send = [item["prompt"] for item in eval_data]
    
    # --- 6. 生成 completions ---
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=None)

    print(f"Generating completions for {len(prompts_to_send)} problems...")
    # 现在 prompts_to_send 始终是一个字符串列表，vLLM可以正确处理
    outputs = llm.generate(prompts_to_send, sampling_params)

    # --- 7. 评估结果 ---
    correct = 0
    incorrect_completions = []

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        predicted_answer = extract_answer(generated_text)
        ground_truth = eval_data[i]["numerical_answer"]

        if predicted_answer == ground_truth:
            correct += 1
        else:
            incorrect_completions.append({
                "original_prompt": eval_data[i]["prompt"],
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth
            })

    accuracy = correct / len(eval_data)
    print(f"\nAccuracy: {correct}/{len(eval_data)} = {accuracy:.2%}")

    # --- 8. 保存评估结果 (使用动态文件名) ---
    os.makedirs("evals", exist_ok=True)
    model_name_sanitized = args.model_name.replace('/', '-')
    eval_type = "reason" if args.reason else "base"
    output_filename = f"evals/{model_name_sanitized}_{eval_type}_incorrect_completions.json"

    with open(output_filename, "w") as f:
        json.dump(incorrect_completions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(incorrect_completions)} incorrect completions to {output_filename}")

if __name__ == "__main__":
    main()