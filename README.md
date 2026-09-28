# SSS Ans
## Setup:   
- [ ] Python3.14 virtual env   
- [ ] PyTorch   
- [ ] HF transformers library   
## Choosing the model:  
Download correct model with correct quant  
Model:  
* around 8b Param  
* Q8 probably  
* This but quantized [https://huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)  
* Mac option: [https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-8bit)  
* Cuda gpu option: nah it depends idk  
## Downloading and loading the model:  
### Transformers lib  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("qwen/qwen3.5-9b", dtype="auto", device_map="auto") # this in bf16 is 18gb
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen3.5-9b")
```
### Mlx lib:  
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
I learned to read docs carefully, and that you must look for prompt-template.jinja to find the stuff for the prompt settings  
  
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
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-9B-MLX-8bit")

prompt = "Statement: The sky is blue. Response should contain only Yes or No."

messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

prompt = mx.array
prompt = prompt[None]
```
