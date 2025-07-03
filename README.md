# GRPO on GSM 8K

This repository implements reinforcement learning with verifier rewards (RLVR)
with [Dr GRPO](https://arxiv.org/abs/2503.20783) for the [GSM8K
dataset](https://huggingface.co/datasets/openai/gsm8k) using
Hugging Face's
[`GRPOTrainer`](https://huggingface.co/docs/trl/main/en/grpo_trainer).

The TL;DR is that GRPO works:
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) goes from
57.80% accuracy to 84.40%, and
[Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) goes
from a mere 7.00% to 39.00%.

Here is an intro to the files in this repo:

- `prepare_gsm8k.py`
  - Prepares two [GSM8K](https://huggingface.co/datasets/openai/gsm8k) JSONLs
    using Hugging Face [Datasets](https://huggingface.co/docs/datasets/en/index):
    the training split and a 500-question evaluation split.
- `evaluate_gsm8k.py`
  - evaluates a model on the saved 500-question JSONL using
    [vLLM](https://github.com/vllm-project/vllm) for fast inference. Its cousin
    `evaluate_gsm8k_no_vllm.py` implements evaluation for old Hugging Face
    models that are not supported by vLLM, such as
    [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-2.7B).
- `train_grpo.py`
  - Executes via Hugging Face
    [Accelerate](https://huggingface.co/docs/accelerate/en/index) for multi-GPU
    training via the Slurm script `submit_grpo.slurm`. It relies on
    `GRPOTrainer` and defines reward functions for accuracy, format and length.
- `instruct_model_repl.py`
  - Chat with models in your terminal. Edit the file directly with the Hugging
    Face models you'd like to chat with at once. Run `python
    instruct_model_repl.py`, add your prompt after the `>>> ` and hit Enter.
    Responses from all the models are shown at once. Useful for quickly
    evaluating capabilities for different models you're interested in.

Environment setup instructions:

```bash
conda create -n grpo python=3.10
conda activate grpo
pip install transformers accelerate trl wandb torch datasets tqdm vllm
pip install flash-attn
```

Usage is scrappy:

```bash
# Prepare the dataset
python prepare_gsm8k.py

# Evaluate baseline performance
python evaluate_gsm8k.py Qwen/Qwen2.5-3B-Instruct

# Train with GRPO
sbatch submit_grpo.slurm
# or for single GPU:
python train_grpo.py Qwen/Qwen2.5-3B-Instruct

# Evaluate trained model
python evaluate_gsm8k.py ./outputs/checkpoint-final
```

## Results

Before GRPO:

| Model | Accuracy |
| --- | --- |
| Qwen/Qwen2.5-7B-Instruct | 91.80 |
| Qwen/Qwen2.5-3B-Instruct | 57.80 |
| Qwen/Qwen2.5-0.5B-Instruct | 46.40 |
| Qwen/Qwen-7B-Chat | 36.20 |
| meta-llama/Llama-2-7b-chat-hf | 7.00 |
| EleutherAI/gpt-neo-2.7B | 2.40 |

I picked one already-somewhat-strong model and one weak-but-promising model.
After GRPO:

| Model | Accuracy |
| --- | --- |
| Qwen/Qwen2.5-3B-Instruct | 84.40 |
| meta-llama/Llama-2-7b-chat-hf | 39.00 |
