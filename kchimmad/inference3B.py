
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

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = 'meta-llama/Llama-3.1-8B-Instruct'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
# Defaults to float32
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir, device_map=device, torch_dtype=torch.bfloat16)
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
batch_size=64
# temp = 0.0
# top_p = 0
# top_k = 0
# max_prompt_len = 3072
max_gen_toks = 512

stop_strings =  ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']


questions =[
    'There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?',
    'If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?',
    'Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?',
    'Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?',
    'Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?',
    'There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
    'Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?',
    'Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'
]

answers =[
    'There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6',
    'There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5',
    'Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39',
    'Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8',
    'Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9',
    'There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29',
    'Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33',
    'Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8'
]


chat = []

for i in range(len(questions)):
    chat.append({'role':'user','content':f'Given the following problem, reason and give a final answer to the problem.\nProblem: {questions[i]}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'})
    chat.append({'role':'assistant','content':answers[i]})


for i in tqdm(range(0, len(data_test["question"]), batch_size), desc="Processing questions"):
    batch_questions = data_test["question"][i:i+batch_size]
    inputs = []
    for q in batch_questions:
        chat_copy = copy.deepcopy(chat)
        chat_copy.append({'role':'user','content':f'Given the following problem, reason and give a final answer to the problem.\nProblem: {q}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'})
        inputs.append(tokenizer.apply_chat_template(chat_copy,  tokenize= False, add_generation_prompt=True))
   
    # tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding='longest',  **add_special_tokens)
    
    
    tokenized_inputs.to(device)
    stopping_criteria = stop_sequences_criteria(tokenizer, stop_strings, tokenized_inputs['input_ids'].shape[1], tokenized_inputs['input_ids'].shape[0])

    with torch.no_grad():
        max_length=tokenized_inputs['input_ids'].shape[1] + max_gen_toks
        output = model.generate(**tokenized_inputs, max_length = max_length, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.pad_token_id,do_sample=False)
        
        # output = model.generate(**tokenized_inputs, max_new_tokens = 1024,pad_token_id=tokenizer.pad_token_id,do_sample=False, stop_strings = stop_strings, tokenizer = tokenizer )
    
    for j, o in enumerate(output):
        generated_text = tokenizer.decode(o, skip_special_tokens=True)
        answer = generated_text.split("Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.assistant")[-1].strip()
        generated_outputs.append({"input": inputs[j], "output": generated_text, "question": batch_questions[j], "answer":answer})
   
output_file_name = f"../outputs/gsm8k/LLaMA3B/generated_outputs_bfloat16_new_tokens_512.json" 
with open(output_file_name, "w") as f:
    json.dump(generated_outputs, f, indent=4)

score = get_score(data_eval_path,output_file_name)
print("SCORE of LLaMA 3B Model at bfloat16 precision is: ",score)