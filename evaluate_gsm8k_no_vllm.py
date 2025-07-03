import json
import os
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from math_utils import extract_answer

# Model to evaluate
model_name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/gpt-neo-2.7B"

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side to left for decoder-only models
tokenizer.padding_side = "left"

# Load evaluation data
eval_data = []
with open("data/gsm8k_500.jsonl", "r") as f:
    for line in f:
        eval_data.append(json.loads(line))

# Prepare prompts
prompts = [item["prompt"] for item in eval_data]

# Generate completions
generated_texts = []
print(f"Generating completions for {len(prompts)} problems...")

# Batch processing for efficiency
batch_size = 128  # Adjust based on GPU memory
for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
    batch_prompts = prompts[i:i+batch_size]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    for j, output in enumerate(outputs):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        # Find where the prompt ends in the generated text
        prompt = batch_prompts[j]
        if prompt in generated:
            # Extract everything after the prompt
            prompt_end = generated.find(prompt) + len(prompt)
            generated_text = generated[prompt_end:].strip()
        else:
            # Fallback if prompt not found (shouldn't happen)
            generated_text = generated.strip()
        generated_texts.append(generated_text)

# Evaluate
correct = 0
incorrect_completions = []

for i, generated_text in enumerate(generated_texts):
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
