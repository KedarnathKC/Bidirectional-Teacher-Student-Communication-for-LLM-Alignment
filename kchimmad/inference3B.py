
import re
import copy
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
import random
import json
from tqdm import tqdm
from score_preds import get_score
from utils import stop_sequences_criteria


cache_dir = '/scratch3/workspace/wenlongzhao_umass_edu-reason/dev_kedar/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Loading data
data_eval_path = "../datasets/gsm8k/test/"
data_test = load_from_disk(data_eval_path)
data_train = load_from_disk("../datasets/gsm8k/train/")

# Loading model
hf_token = os.getenv("hf_token")
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = 'meta-llama/Llama-3.1-8B-Instruct'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_name, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, config=config,cache_dir=cache_dir)
# Defaults to float32
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, config=config,cache_dir=cache_dir, device_map=device)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
add_special_tokens = {"add_special_tokens": False }

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)

model.eval()
generated_outputs=[]
# Adjust batch size according to your GPU memory capacity
# With vram=80G, 1B - 64, 3B - 64, 8B - 32
batch_size=1
temp = 0.0
top_p = 0
top_k = 0
max_prompt_len = 3072
for i in tqdm(range(0, len(data_test["question"]), batch_size), desc="Processing questions"):
    batch_questions = data_test["question"][i:i+batch_size]
    inputs = []
    for q in batch_questions:
        # chat = [
        #     {"role":"system", "content": "You are a helpful assistant."},
        #     {"role":"user", "content":userContent+f'\n\nQ: {q}\nA: '}
        # ] 
        # chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, include_metadata=False)
        # inputs.append(chat)

        inputs.append(prompt + f'<|start_header_id|>user<|end_header_id|>\n\nGiven the following problem, reason and give a final answer to the problem.\nProblem: {q}\nYour response should end with \'The final answer is [answer]\' where [answer] is the response to the problem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n') 

    # inputs = [fewShotPrompt+"Now, solve the below question following the instructions given above. \n\nQ: "+q+"\nA: <|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    # inputs = [fewShotPrompt+"Now, Follow the same format for reasoning and stating your final answer as above examples and Answer the below question\n\nQ: "+q+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    # tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', max_length=max_prompt_len)
    tokenized_inputs.to(device)
    stopping_criteria = stop_sequences_criteria(tokenizer, stop_strings, tokenized_inputs['input_ids'].shape[1], tokenized_inputs['input_ids'].shape[0])

    with torch.no_grad():
        max_length=tokenized_inputs['input_ids'].shape[1] + max_gen_toks
        output = model.generate(**tokenized_inputs, max_length = max_length, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.pad_token_id,do_sample=False)
        
        # output = model.generate(**tokenized_inputs, max_new_tokens = 1024,pad_token_id=tokenizer.pad_token_id,do_sample=False, stop_strings = stop_strings, tokenizer = tokenizer )
    
    for j, o in enumerate(output):
        generated_text = tokenizer.decode(o, skip_special_tokens=True)
        answer = generated_text.split("Your response should end with \'The final answer is [answer]\' where [answer] is the response to the problem.assistant")[-1]
        generated_outputs.append({"input": inputs[j], "output": generated_text, "question": batch_questions[j], "answer":answer})
        
        # generated_outputs.append({"input": inputs[j], "output": generated_text})

output_file_name = f"../outputs/gsm8k/LLaMA3B/generated_outputs_test_with_regex_and_stop_words_batch_size_{batch_size}_prompt_len_{max_prompt_len}.json" 
with open(output_file_name, "w") as f:
    json.dump(generated_outputs, f, indent=4)
    
score = get_score(data_eval_path,output_file_name)
print("SCORE of LLaMA 3B Model at bfloat16 precision is: ",score)