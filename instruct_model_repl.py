#!/usr/bin/env python
"""
Quick throwaway REPL for comparing model outputs.
Usage: Edit model names to compare directly in the script.
Exit with Ctrl-D or type "quit".
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    models = ["Qwen/Qwen2.5-7B-Instruct"]

    # Load everything up front
    loaded = []
    for repo_id in models:
        tok = AutoTokenizer.from_pretrained(
            repo_id, use_fast=True, trust_remote_code=True
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        loaded.append((repo_id, tok, mdl))

    gen_cfg = dict(max_new_tokens=1028, top_p=0.9, temperature=0.7)

    while True:
        try:
            user_in = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_in.lower() in {"quit", "exit"}:
            break

        for repo_id, tok, mdl in loaded:
            messages = [
                {
                    "role": "system",
                    "content": "You are a careful math tutor. Show concise reasoning, then write your final answer on its own line, prefixed by four hash marks. For example, if the final answer is 8, your last line should be #### 8.",
                },
                {"role": "user", "content": user_in},
            ]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tok([text], return_tensors="pt").to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**model_inputs, **gen_cfg)
            assistant_ids = out[0][len(model_inputs.input_ids[0]) :]
            decoded = tok.decode(assistant_ids, skip_special_tokens=True).lstrip()

            print(f"{repo_id}:\n")
            print(decoded, end="\n\n")


if __name__ == "__main__":
    main()
