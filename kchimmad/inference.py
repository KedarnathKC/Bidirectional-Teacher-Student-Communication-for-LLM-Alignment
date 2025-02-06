import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
import random
import json
from tqdm import tqdm
from score_preds import get_score

cache_dir = '/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Loading data
data_eval_path = "../datasets/gsm8k/test/"
data_test = load_from_disk(data_eval_path)
data_train = load_from_disk("../datasets/gsm8k/train/")

# Loading model
hf_token = os.getenv("hf_token")
# model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = 'meta-llama/Llama-3.1-8B-Instruct'
model_name = 'ibm-granite/granite-3.0-2b-instruct'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(model_name, token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, config=config,cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, config=config,cache_dir=cache_dir, device_map=device)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

# 8-shot prompt
userContent='Below are few example question and answer pairs\n\n'

# 8-shot examples from:
# https://arxiv.org/pdf/2201.11903 (Mentioned in the LLaMA website that they use this prompt)
# The same is used in lm-eval-harness here: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml
userContent += "Here your job is to answer a math question. "
userContent += "Your question will appear after 8 demonstrations of similar math tasks. "
userContent += "As in those demonstrations, you must generate both a step-by-step reasoning and a final answer. "
userContent += "Perform a simple action, e.g., a single mathematical operation, at each step of your reasoning. "
userContent += "Your final answer must contain only a number and no additional text. "
userContent += "State your final answer after ####.\n"
userContent += '\nQ: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?'
userContent += '\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6. #### 6\n'
userContent += '\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?'
userContent += '\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5. #### 5\n'
userContent += '\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?'
userContent += '\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. #### 39\n'
userContent += '\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?'
userContent += '\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8. #### 8\n'
userContent += '\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'
userContent += '\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9. #### 9\n'
userContent += '\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?'
userContent += '\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29. #### 29\n'
userContent += '\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'
userContent += '\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33. #### 33\n'
userContent += '\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'
userContent += '\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. #### 8\n'
userContent += '\nNow, solve the below question following the instructions given above.'

# # 8-shot examples from train set
# for i in range(9):

#     fewShotPrompt+=f'Q: {data_train["question"][i]}\nA: {data_train["answer"][i]}\n\n'
# 8-shot examples from: https://github.com/kojima-takeshi188/zero_shot_cot/blob/main/utils.py
# fewShotPrompt+='Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n\nA: There are 15 trees originally.\nThen there were 21 trees after some more were planted.\nSo there must have been 21 - 15 = 6.\n#### 6\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n\nA: Originally, Leah had 32 chocolates.\nHer sister had 42. So in total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\n#### 39\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n\nA: Jason started with 20 lollipops.\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8.\n#### 8\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n\nA: Shawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n\nA: There were originally 9 computers.\nFor each of 4 days, 5 more computers were added.\nSo 5 * 4 = 20 computers were added.\n9 + 20 is 29.\n#### 29\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n\nA: Michael started with 58 golf balls.\nAfter losing 23 on tuesday, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n\nA: Olivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\nSo she has 23 - 15 dollars left.\n23 - 15 is 8.\n#### 8\n\n'

model.eval()
generated_outputs=[]
# Adjust batch size according to your GPU memory capacity
# With vram=80G, 1B - 64, 3B - 64, 8B - 32
batch_size=64
temp = 0.1
top_p = 0.95
for i in tqdm(range(0, len(data_test["question"]), batch_size), desc="Processing questions"):
    batch_questions = data_test["question"][i:i+batch_size]
    inputs = []
    for q in batch_questions:
        chat = [
            {"role":"system", "content": "You are a helpful assistant."},
            {"role":"user", "content":userContent+f'\n\nQ: {q}\nA: '}
        ] 
        chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, include_metadata=False)
        inputs.append(chat)
    # inputs = [fewShotPrompt+"Now, solve the below question following the instructions given above. \n\nQ: "+q+"\nA: <|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    # inputs = [fewShotPrompt+"Now, Follow the same format for reasoning and stating your final answer as above examples and Answer the below question\n\nQ: "+q+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
    tokenized_inputs.to(device)

    with torch.no_grad():
        output = model.generate(**tokenized_inputs, pad_token_id=tokenizer.pad_token_id)
        # output = model.generate(**tokenized_inputs, max_length=2000, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=temp, top_p=top_p)
        # output = model.generate(**tokenized_inputs, max_new_tokens=256, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.1, top_p=0.95)
   
    for j, o in enumerate(output):
        generated_text = tokenizer.decode(o, skip_special_tokens=True)
        answer = generated_text.split("A: assistant")[-1]
        generated_outputs.append({"input": inputs[j], "output": generated_text, "question": batch_questions[j], "answer":answer})
        # generated_outputs.append({"input": inputs[j], "output": generated_text})

output_file_name = "../outputs/gsm8k/Granite2B/generated_outputs_test_new_prompt_lm_eval_harness_without_max_len.json" 
with open(output_file_name, "w") as f:
    json.dump(generated_outputs, f, indent=4)

score = get_score(data_eval_path,output_file_name)
print("SCORE of Granite 2B Model is: ",score)