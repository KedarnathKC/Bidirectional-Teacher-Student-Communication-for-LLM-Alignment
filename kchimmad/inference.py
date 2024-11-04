import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
import random
import json
from tqdm import tqdm

def create_fs_prompt(question,n,x):
    '''
    question: The question we want the model to answer
    n: The number of examples for few shot
    x: The data from which we get the few shot examples.
    '''
    prompt=""
    for i in range(n):
        ind=random.randint(0, len(x)-1)
        prompt+="Question: "+x["question"][ind]+"\n\nAnswer: "+x["answer"][ind]+"\n\n"
    prompt+="Question: "+question
    return prompt  

os.environ['TRANSFORMERS_CACHE'] = '/scratch/workspace/wenlongzhao_umass_edu-analyze/transformers_cache'

# Loading data
data_feedback = load_from_disk("../datasets/gsm8k/feedback/")
data_train = load_from_disk("../datasets/gsm8k/train/")

# Loading model
hf_token = os.getenv("hf_token")
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
config = AutoConfig.from_pretrained(model_name, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, config=config,cache_dir='/scratch/workspace/wenlongzhao_umass_edu-analyze/transformers_cache')
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, config=config,cache_dir='/scratch/workspace/wenlongzhao_umass_edu-analyze/transformers_cache')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 8-shot prompt
fewShotPrompt=oneShotPrompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBelow are few example question and answer pairs\n\n'

for i in range(9):
    fewShotPrompt+=f'Q: {data_train["question"][i]}\n\nA: {data_train["answer"][i]}\n\n'

model.eval()
generated_outputs=[]
batch_size=32 # Adjust batch size according to your GPU memory capacity
for i in tqdm(range(0, len(data_feedback["question"]), batch_size), desc="Processing questions"):
    batch_questions = data_feedback["question"][i:i+batch_size]
    inputs = [fewShotPrompt+"Now, Follow the same format for reasoning and stating your final answer using #### as above examples and Answer the below question\n\nQ: "+q+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    # inputs = [fewShotPrompt+"Now, Follow the same format for reasoning and stating your final answer as above examples and Answer the below question\n\nQ: "+q+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    tokenized_inputs.to(device)

    with torch.no_grad():
        output = model.generate(**tokenized_inputs, max_length=2000, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.1, top_p=0.95)
        # output = model.generate(**tokenized_inputs, max_new_tokens=256, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.1, top_p=0.95)
   
    for j, o in enumerate(output):
        generated_text = tokenizer.decode(o, skip_special_tokens=True)
        generated_outputs.append({"input": inputs[j], "output": generated_text})

with open("../outputs/gsm8k/LLaMA8B/generated_outputs.json", "w") as f:
    json.dump(generated_outputs, f, indent=4)

    
    


