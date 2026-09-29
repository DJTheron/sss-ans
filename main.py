import os
os.environ["HF_HUB_OFFLINE"] = "1" # must be set to zero if model is not downloaded already!!

from mlx_lm import load, generate
import mlx.core as mx
import time
from fastapi import FastAPI

TRUEID = 2434
FALSEID = 3913

#try:
print(f"[{time.strftime('%H:%M:%S')}] model_loading...")
start = time.perf_counter() # changed to this from time.time() becuase the processors clock can jump and this adjusts for it

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit") #type: ignore

print(f"[{time.strftime('%H:%M:%S')}] model_loaded in {time.perf_counter() - start} seconds")

#except:
#    print("Model is probably not downloaded already so edit me (the main.py program) and at the top set HF_HUB_OFFLINE=0")

app = FastAPI()

@app.post("/noul/")
def prob_batch(statements: list[str]) -> list[float]: #shows the thing accepts a list of strings and returns a list of floats
    if statements == []:
        return []
    prompts = []
    final_prompts = []
    p_lengths = []
    probabilities_true = []
    for statement in statements:
        prompt = f"Statement: {statement} Question: Is this True or False? Response format: Response should contain only True or False."

        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

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
prob_batch(["Warmup"])
print(f"[{time.strftime('%H:%M:%S')}] model_warmed_up")

#batch_statements = ["The sky is green on a clear day.", "The sky is blue on a clear day.", "Pineapple belongs on pizza."]


#    print(f"[{time.strftime('%H:%M:%S')}] model_running_batch...")
#    start = time.perf_counter()
#
#    print(prob_batch(batch_statements))
#
#    print(f"[{time.strftime('%H:%M:%S')}] model_finished_batch in", time.perf_counter() - start, "seconds")