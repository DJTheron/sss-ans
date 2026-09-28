from mlx_lm import load, generate


model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit")

prompt = "Statement: The sky is blue. Response should contain only Yes or No."

messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

