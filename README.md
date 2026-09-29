
# SSS Ans
SSS-ANS scores how true or false a statement is by comparing a LLMs true and false logits.
I haven't built the CUDA/pytorch version yet. Currently MLX on Mac is supported.

## Choosing the model:  
Download correct model with correct quant  
Model:  
* around 8b Param  
* Q8 probably  
* This but quantized [https://huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)  
* Mac option: [https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit) 
## Downloading and loading the model:  
### Transformers lib  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("qwen/qwen3.5-9b", dtype="auto", device_map="auto") # this in bf16 is 18gb
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen3.5-9b")
```
### MLX lib:  
Install:  
```bash
pip install mlx-lm
```
Run:  
```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit")
```
Model is now loaded.  
## Running model to get yes and no tokens:  
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit")

prompt = "Statement: The sky is blue. Response should contain only Yes or No."

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

print(prompt)

# for later: text = generate(model, tokenizer, prompt=prompt, verbose=True)
```
I learned to read docs carefully, and that you must look for prompt-template.jinja to find the stuff for the prompt settings.  
  
### Token table  

| Word  | No space in front | Space in front |
| ----- | ----------------- | -------------- |
| Yes   | 9175              | 7179           |
| No    | 2665              | 2233           |
| yes   | 9405              | 9542           |
| no    | 2083              | 874            |
| YES   | 13602             |                |
| NO    | 8725              |                |
| True  | 2434              | 2912           |
| False | 3913              | 3439           |
| true  | 1802              | 804            |
| false | 3721              | 867            |
| A     | 32                |                |
| B     | 33                |                |
  
## Program to get the model result  
``` python
import os
os.environ["HF_HUB_OFFLINE"] = "1" #must be set to zero if model is not downloaded already!!

from mlx_lm import load, generate
import mlx.core as mx


TRUEID = 2434
FALSEID = 3913

def prob(statement: str) -> float: #shows the thing accepts a string and returns a float
    prompt = f"Statement: {statement} Question: Is this True or False? Response format: Response should contain only True or False."

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

    prompt = mx.array(prompt) 
    prompt = prompt[None] # puts it into a "prompt folder" which is what the model expects

    logit_table = model(prompt)

    truescore = logit_table[0, -1, TRUEID].item() # 1st thing is which prompt in the "prompt folder" we are fetching the result from 2nd thing is -1 cuz that fetches the last prediction the model made and trueid fetches the score for the score of the token True
    falsescore = logit_table[0, -1, FALSEID].item() # i learned u can just do this for arrays instead of have [][][] becuase this is more efficient (maybe faster idk)

    probability_true = mx.sigmoid(truescore - falsescore) # if below 50% then false, if above 50% then true

    return probability_true.item()
```

## Check out [main.py](main.py) to see the final code

## License
Copyright (c) 2026 DJTheron

This project is licensed under the [Mozilla Public License 2.0](LICENSE). You can use it in your own projects, including closed-source or commercial ones, but if you change any of these files you must share those changes under the MPL 2.0. I chose this to keep the spirit of open source without stopping progress.
