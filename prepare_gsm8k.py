import json
import os
import re
from datasets import load_dataset

# Create data directory
os.makedirs("data", exist_ok=True)

# Load GSM8K
dataset = load_dataset("/data/xiongtao/RL_learning/data/gsm8k", "main")

PROMPT = "You are a careful math tutor. Show concise reasoning, then write your final answer on its own line, prefixed by four hash marks. For example, if the final answer is 8, your last line should be #### 8."


def extract_numerical_answer(answer_text):
    # GSM8K answers end with #### followed by the numerical answer
    match = re.search(r"#### ([-\d,]+)", answer_text)
    if match:
        # Remove commas and convert to int
        return int(match.group(1).replace(",", ""))
    return None


# Process train split
train_data = []
for item in dataset["train"]:
    processed = {
        "question": item["question"],
        "prompt": PROMPT + " " + item["question"],
        "answer": item["answer"],
        "numerical_answer": extract_numerical_answer(item["answer"]),
    }
    train_data.append(processed)

# Write train data
with open("data/gsm8k_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

# Process first 500 from test split for evaluation
eval_data = []
for i, item in enumerate(dataset["test"]):
    if i >= 500:
        break
    processed = {
        "question": item["question"],
        "prompt": PROMPT + " " + item["question"],
        "answer": item["answer"],
        "numerical_answer": extract_numerical_answer(item["answer"]),
    }
    eval_data.append(processed)

# Write eval data
with open("data/gsm8k_500.jsonl", "w") as f:
    for item in eval_data:
        f.write(json.dumps(item) + "\n")

print(f"Prepared {len(train_data)} training examples -> data/gsm8k_train.jsonl")
print(f"Prepared {len(eval_data)} evaluation examples -> data/gsm8k_500.jsonl")
