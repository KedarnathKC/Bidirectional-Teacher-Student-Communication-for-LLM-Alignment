import pandas as pd
from tqdm import tqdm
from evaluate import load
import json
from datasets import load_from_disk

def get_answer_dataset(row):
    return row['answer'].split('####')[-1]
def get_answer_model(row):
    return row['output'].split('####')[-1]

data_test = load_from_disk("../datasets/gsm8k/test")

model_output_path =  '../outputs/gsm8k/LLaMA3B/generated_outputs_test_new_prompt_greedy.json'
df = pd.read_json(model_output_path)
test = pd.DataFrame(data_test)
df['model_answer']=df.apply(get_answer_model,axis=1)
test['GT_Answer']=test.apply(get_answer_dataset,axis=1)
exact_match_metric = load("exact_match")

for i in tqdm(range(df.shape[0])):
    df.loc[i,'score']=exact_match_metric.compute(predictions=[df.iloc[i]['model_answer']], references=[test.iloc[i]["GT_Answer"]])['exact_match']

# df['score'] = (df['model_answer'] == feedback['GT_Answer']).astype(int)

df.drop(columns=['model_answer'],inplace=True)
print("SCORE: ", df['score'].sum()/df.shape[0])
records = df.to_dict(orient='records')
formatted_json = json.dumps(records, indent=4)
with open(model_output_path, "w") as f:
    f.write(formatted_json)
