import pandas as pd
from tqdm import tqdm
from evaluate import load
import json
from datasets import load_from_disk
import re

def get_answer_dataset(row):
    return row['answer'].split('####')[-1].strip()
def get_answer_model(row):
    regex_pattern = r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))"
    match = re.search(regex_pattern, row['answer'])
    if match:
        # Group -1 corresponds to the entire match
        result = match.group(1)
    else:
        result = None

    return result

    # return row['answer'].split('The final answer is')[-1].strip()

def get_score(data_path, output_path):
    '''
        data_path = Path to HF dataset which has GT
        output_path = path to json file containing model outputs
    '''
    test = pd.DataFrame(load_from_disk(data_path))    
    df = pd.read_json(output_path)
    # df.drop(columns=['score','model_answer'],inplace=True)
    df['model_answer'] = df.apply(get_answer_model,axis=1)
    test['GT_Answer'] = test.apply(get_answer_dataset, axis=1)
    # print(f'DF Shape: {df.shape}, Test Shape: {test.shape}')
    exact_match_metric = load("exact_match")
    
    # df['score'] = (df['model_answer'] == test['GT_Answer']).astype(int)
    # for i in tqdm(range(df.shape[0])):
    for i in range(df.shape[0]):
        df.loc[i,'score'] = exact_match_metric.compute(predictions = [df.iloc[i]['model_answer']], references = [test.iloc[i]['GT_Answer']])['exact_match']
    
    # df.drop(columns=['model_answer'], inplace=True)
    score = df['score'].sum()/df.shape[0]
    # Write back the exact scores to output_path
    records = df.to_dict(orient = 'records')
    formatted_json = json.dumps(records, indent=4)
    
    with open(output_path,'w') as f:
        f.write(formatted_json)
    
    return score

def main():
    # data_test = load_from_disk("../datasets/gsm8k/test")
    data_path ="../datasets/gsm8k/test"
    model_output_path = f"../outputs/gsm8k/LLaMA1B/generated_outputs_full_precision.json" 

    score = get_score(data_path, model_output_path)
    print("The score of the model is: ",score)

if __name__=='__main__':
    main()
