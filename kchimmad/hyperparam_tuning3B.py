import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
from datasets import load_from_disk
import random
import json
from tqdm import tqdm
from score_preds import get_score


def inference(model_name, data_train_path, data_eval_path, top_p, temp, batch_size=32):
    '''
    model_name : HF identifiable model name
    data_train_path : path to HF train dataset
    data_eval_path : path to HF evaluation dataset
    top_p : list of top_p values
    temp : list of temp values
    batch_size : 1B & 3B -> 64, 8B -> 32
    '''
    

    # Loading data
    data_test = load_from_disk(data_eval_path)
    data_train = load_from_disk(data_train_path)

    # Loading model
    hf_token = os.getenv("hf_token")
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, config=config,cache_dir='../transformers_cache')
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, config=config,cache_dir='../transformers_cache')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fewShotPrompt=f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n'
    fewShotPrompt+='\n\nBelow are few example question and answer pairs\n\n'
    fewShotPrompt += "Here your job is to answer a math question. "
    fewShotPrompt += f"Your question will appear after 8 demonstrations of similar math tasks. "
    fewShotPrompt += "As in those demonstrations, you must generate both a step-by-step reasoning and a final answer. "
    fewShotPrompt += "Perform a simple action, e.g., a single mathematical operation, at each step of your reasoning. "
    fewShotPrompt += "Your final answer must contain only a number and no additional text. "
    fewShotPrompt += "State your final answer after ####. "

    # 8-shot examples from train set
    for i in range(9):
        fewShotPrompt+=f'Q: {data_train["question"][i]}\nA: {data_train["answer"][i]}\n\n'

    model.eval()
    
    exp_runs=[]
    for p in top_p:
        for t in temp:
            generated_outputs=[]
            for i in tqdm(range(0, len(data_test["question"]), batch_size), desc="Processing questions"):
                batch_questions = data_test["question"][i:i+batch_size]
                inputs = [fewShotPrompt+"Now, solve the below question following the instructions given above. \n\nQ: "+q+"\nA: <|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
                # inputs = [fewShotPrompt+"Now, Follow the same format for reasoning and stating your final answer as above examples and Answer the below question\n\nQ: "+q+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>" for q in batch_questions]
                tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
                tokenized_inputs.to(device)

                with torch.no_grad():
                    # output = model.generate(**tokenized_inputs, max_length=2000, pad_token_id=tokenizer.pad_token_id)
                    output = model.generate(**tokenized_inputs, max_length=2000, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=t, top_p=p)
                    # output = model.generate(**tokenized_inputs, max_new_tokens=256, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.1, top_p=0.95)
            
                for j, o in enumerate(output):
                    generated_text = tokenizer.decode(o, skip_special_tokens=True)
                    answer = generated_text.split("A: assistant")[-1]
                    generated_outputs.append({"input": inputs[j], "output": generated_text, "question": batch_questions[j], "answer":answer})
                    # generated_outputs.append({"input": inputs[j], "output": generated_text})
            output_file_name = f"../outputs/gsm8k/LLaMA3B/generated_outputs_val_hyptune_{t}_{p}.json" 
            with open(output_file_name, "w") as f:
                json.dump(generated_outputs, f, indent=4)   
            
            score = get_score(data_eval_path,output_file_name)
            exp_runs.append({'model':model_name,'temp':t,'top_p':p,'score':score})
            print(f"\n\nModel Name: {model_name}\tTemp: {t}\tTop_P: {p}\tScore: {score}")
    
    output_file_name = f"../outputs/gsm8k/hyperparameter3B.json" 
    with open(output_file_name, "w") as f:
        json.dump(exp_runs, f, indent=4)   
    
    return

def main():
    os.environ['TRANSFORMERS_CACHE'] = '../transformers_cache'
    
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    batch_size = 64

    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    # batch_size = 32

    train_path = "../datasets/gsm8k/train/"
    val_path = "../datasets/gsm8k/val/"
    # top_p = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6]
    top_p = [0.65,0.6]
    temp = [0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7,1.9]

    inference(model_name,train_path,val_path,top_p,temp,batch_size)


if __name__ == '__main__':
    main()