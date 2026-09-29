import os
os.environ["HF_HUB_OFFLINE"] = "1" # must be set to zero if model is not downloaded already!!

from mlx_lm import load
import mlx.core as mx
import time
from fastapi import FastAPI, HTTPException

TRUEID = 2434
FALSEID = 3913

try:
    print(f"[{time.strftime('%H:%M:%S')}] model_loading...")
    start = time.perf_counter() # changed to this from time.time() becuase the processors clock can jump and this adjusts for it

    model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore

    print(f"[{time.strftime('%H:%M:%S')}] model_loaded in {time.perf_counter() - start} seconds")

except FileNotFoundError: # this is the error load() will ommit if it cannot find the file locally (not exact error but this one is a general term that will catch the specific one)
    print("Model is probably not downloaded already so edit me (the main.py program) and at the top set HF_HUB_OFFLINE=0")
    raise SystemExit(1)

max_mem = int(mx.device_info()["max_recommended_working_set_size"]) # we do these two lines so the server doesnt get its memory compressed which makes it slow if its been sitting idle
mx.set_wired_limit(max_mem)

app = FastAPI()

@app.post("/prot/")
def prob_batch(statements: list[str]) -> list[float]: #shows the thing accepts a list of strings and returns a list of floats
    if statements == []: # exits if empty instead of erroring
        return []
    if len(statements) > 128:
        raise HTTPException(status_code=413, detail="Request too large, list exceeded 128 statements.") # error 413 means request too large
    prompts = []
    final_prompts = []
    p_lengths = []
    
    for statement in statements:
        prompt = f"Statement: {statement} Question: Is this True or False? Response format: Response should contain only True or False."

        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False) #thinking is turned off as we just want the scoreboard (logit table) of the first response token the model generates

        p_lengths.append(len(prompt)) # so we can add padding + calculate which token to fetch
        prompts.append(prompt) # puts it into a "prompt folder" which is what the model expects

    longest = max(p_lengths)
    
    for prompt in prompts:
        padding = longest - len(prompt)
        prompt = prompt + [0] * padding
        final_prompts.append(prompt)
    
    final_prompts = mx.array(final_prompts)
    logit_table = model(final_prompts)

    index_row = mx.arange(len(statements))
    index_pos = mx.array(p_lengths) - 1
    
    truescores = logit_table[index_row, index_pos, TRUEID]
    falsescores = logit_table[index_row, index_pos, FALSEID]
    
    probabilities_true = mx.sigmoid(truescores - falsescores).tolist()
    
    return probabilities_true # type: ignore

print(f"[{time.strftime('%H:%M:%S')}] model_warming_up...")
start = time.perf_counter()
prob_batch(["Warmup"])
print(f"[{time.strftime('%H:%M:%S')}] model_warmed_up in {time.perf_counter() - start} seconds")