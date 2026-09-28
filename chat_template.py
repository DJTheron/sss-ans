from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore

prompt = "Statement: The sky is blue on a clear day. Question: Is this True or False? Response format: Response should contain only True or False."

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

print(prompt)