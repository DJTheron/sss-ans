from mlx_lm import load, generate
import mlx.core as mx
import os

os.environ["HF_HUB_OFFLINE"] = "1"

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore


prompt = "Statement: The sky is blue on a clear day. Question: Is this True or False? Response format: Response should contain only True or False."

messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

prompt = mx.array(prompt)
prompt = prompt[None]

logit_table = model(prompt)

trueid = logit_table[0][41][2434]
falseid = logit_table[0][41][3913]

print(trueid, falseid)