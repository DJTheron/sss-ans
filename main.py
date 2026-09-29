import os
os.environ["HF_HUB_OFFLINE"] = "1" # must be set to zero if model is not downloaded already!!

from mlx_lm import load, generate
import mlx.core as mx
import time

TRUEID = 2434
FALSEID = 3913

#try:
print(f"[{time.strftime('%H:%M:%S')}] model_loading...")
start = time.perf_counter() # changed to this from time.time() becuase the processors clock can jump and this adjusts for it

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore

print(f"[{time.strftime('%H:%M:%S')}] model_loaded in {time.perf_counter() - start} seconds")

#except:
#    print("Model is probably not downloaded already so edit me (the main.py program) and at the top set HF_HUB_OFFLINE=0")

def prob(statement: str) -> float: #shows the thing accepts a string and returns a float
    prompt = f"Statement: {statement} Question: Is this True or False? Response format: Response should contain only True or False."

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

    prompt = mx.array(prompt) 
    prompt = prompt[None] # puts it into a "prompt folder" which is what the model expects

    logit_table = model(prompt)

    truescore = logit_table[0, -1, TRUEID].item() # 1st thing is which prompt in the "prompt folder" we are fetching the result from 2nd thing is -1 cuz that fetches the last token the model sent and trueid fetches the score for the score of the token True
    falsescore = logit_table[0, -1, FALSEID].item() # i learned u can just do this for arrays instead of have [][][] becuase this is more efficient (maybe faster idk)

    probability_true = mx.sigmoid(truescore - falsescore) # if below 50% then false, if above 50% then true

    return probability_true.item()

def prob_batch(statements: list[str]) -> list[float]: #shows the thing accepts a list of strings and returns a list of floats
    prompt = f"Statement: {statement} Question: Is this True or False? Response format: Response should contain only True or False."

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

    prompt = mx.array(prompt) 
    prompt = prompt[None] # puts it into a "prompt folder" which is what the model expects

    logit_table = model(prompt)

    truescore = logit_table[0, -1, TRUEID].item() # 1st thing is which prompt in the "prompt folder" we are fetching the result from 2nd thing is -1 cuz that fetches the last token the model sent and trueid fetches the score for the score of the token True
    falsescore = logit_table[0, -1, FALSEID].item() # i learned u can just do this for arrays instead of have [][][] becuase this is more efficient (maybe faster idk)

    probability_true = mx.sigmoid(truescore - falsescore) # if below 50% then false, if above 50% then true

    return probability_true.item()

batch_statements = ["The sky is green on a clear day.", "The sky is blue on a clear day.", "Pineapple belongs on pizza."]

start = time.perf_counter()
for statement in batch_statements:
    print(prob(statement))
print("time to beat: ", time.perf_counter() - start)