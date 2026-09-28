import os
os.environ["HF_HUB_OFFLINE"] = "1" #must be set to zero if model is not downloaded already!!

from mlx_lm import load, generate
import mlx.core as mx


TRUEID = 2434
FALSEID = 3913

#try:
model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore

#except:
#    print("Model is probably not downloaded already so edit me (the main.py program) and at the top set HF_HUB_OFFLINE=0")

prompt = "Statement: The sky is blue on a clear day. Question: Is this True or False? Response format: Response should contain only True or False."

messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

prompt = mx.array(prompt)
prompt = prompt[None]

logit_table = model(prompt)

truescore = logit_table[0, -1, TRUEID].item()
falsescore = logit_table[0, -1, FALSEID].item() # i learned u can just do this for arrays instead of have [][][] becuase this is more efficient (maybe faster idk)

probability_true = mx.sigmoid(truescore - falsescore) # if below 50% then false, if above 50% then true

print(truescore, falsescore, probability_true)
