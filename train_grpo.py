import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import wandb
from math_utils import extract_answer

# Initialize accelerator
accelerator = Accelerator()

# Model setup
model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": accelerator.device},
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Initialize wandb
if accelerator.is_main_process:
    wandb.init(project=f"{model_name.replace('/', '-')}-gsm8k-grpo")

# Load training data
train_data = []
with open("data/gsm8k_train.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        train_data.append({
            "prompt": item["prompt"],
            "numerical_answer": item["numerical_answer"]
        })

# Create dataset
dataset = Dataset.from_list(train_data)

# Reward functions
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
        # Reward for having #### in the output
        if "####" in completion:
            rewards.append(0.2)
        else:
            rewards.append(-0.2)
    return rewards

def length_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Penalize excessive brevity
        if len(completion) < 30:
            rewards.append(-0.5)
        else:
            # No penalty for longer completions
            rewards.append(0.0)
    return rewards

# Training config
training_args = GRPOConfig(
    output_dir=f"./grpo_{model_name.replace('/', '-')}_gsm8k",
    learning_rate=5e-6,
    lr_scheduler_type="constant_with_warmup",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    gradient_checkpointing=True,
    warmup_steps=10,
    logging_steps=1,
    save_steps=200,
    eval_strategy="no",
    bf16=True,
    report_to="wandb",
    remove_unused_columns=False,
    num_generations=4,  # Number of completions per prompt
    temperature=0.7,
    max_completion_length=512,
    max_prompt_length=512,
    disable_dropout=True,  # Important for consistent generation
    loss_type="dr_grpo",
    log_completions=True,
)

# Create trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[format_reward, accuracy_reward, length_reward],
    train_dataset=dataset,
)

# Train
trainer.train()

# Save final model
if accelerator.is_main_process:
    trainer.save_model(f"grpo_{model_name.replace('/', '-')}_gsm8k_final")
    print(f"Training complete! Model saved to grpo_{model_name.replace('/', '-')}_gsm8k_final")
