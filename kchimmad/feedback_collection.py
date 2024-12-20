import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm
import json

os.environ['TRANSFORMERS_CACHE'] = '../transformers_cache/llama3.1/'
cache_dir = '../transformers_cache/llama3.1/'

def get_question(row):
    return row['input'].split('Q: ')[-1]
def get_answer(row):
    return row['output'].split('A: ')[-1]


model_name = "meta-llama/Llama-3.1-70B-Instruct"
hf_token = os.getenv("hf_token")

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    offload_state_dict=True,
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

path = '../outputs/gsm8k/LLaMA1B/'
file_name = 'generated_outputs_test'
df = pd.read_json(path+file_name+'.json')
data_test = load_from_disk("/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/datasets/gsm8k/test")
test = pd.DataFrame(data_test)

index = df[df['score']==0].index

df['answer']=df.apply(get_answer,axis=1)

prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nThink like you are an expert in solving maths questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBelow are the answers generated by a student model for the respective answers. Provide feedback on where the model has made a mistake"
ending_prompt="<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

batch_size=8
for i in tqdm(range(0,index.shape[0],batch_size)):
    inputs = [prompt+"\n\nQ: "+test.iloc[index[j]]['question']+"\n\nA: "+df.iloc[index[j]]['answer']+"\n\n"+ending_prompt for j in range(i,i+batch_size)]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        output = model.generate(**tokenized_inputs, max_length=500, num_return_sequences=1,  pad_token_id=tokenizer.pad_token_id)
    for j, o in enumerate(output):
        df.loc[i,'assistant'] = tokenizer.decode(o, skip_special_tokens=True)
        # df.loc[i,'assistant'] = tokenizer.decode(output[0], skip_special_tokens=True)

df.drop(columns=['answer'], inplace=True)

records = df.to_dict(orient='records')
formatted_json = json.dumps(records, indent=4)
with open(path+file_name+'_feedback.json', "w") as f:
    f.write(formatted_json)