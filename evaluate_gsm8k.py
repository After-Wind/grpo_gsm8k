import json
import os
import sys
from vllm import LLM, SamplingParams
from math_utils import extract_answer

# Model to evaluate
model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading model: {model_name}")
llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    dtype="auto",
    trust_remote_code=True,
    max_model_len=1024,
    gpu_memory_utilization=0.95,
    enforce_eager=False,  # Use Flash Attention 2
)

# Load evaluation data
eval_data = []
with open("data/gsm8k_500.jsonl", "r") as f:
    for line in f:
        eval_data.append(json.loads(line))

# Prepare prompts
prompts = [item["prompt"] for item in eval_data]

# Generate completions
sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, stop=None)

print(f"Generating completions for {len(prompts)} problems...")
outputs = llm.generate(prompts, sampling_params)


# Evaluate
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
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth
        })

accuracy = correct / len(eval_data)
print(f"\nAccuracy: {correct}/{len(eval_data)} = {accuracy:.2%}")
# Save incorrect completions
os.makedirs("evals", exist_ok=True)
with open(f"evals/{model_name.replace('/', '-')}_incorrect_completions.json", "w") as f:
    json.dump(incorrect_completions, f, indent=2)
print(f"Saved {len(incorrect_completions)} incorrect completions to evals/{model_name.replace('/', '-')}_incorrect_completions.json")
