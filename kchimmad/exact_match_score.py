import pandas as pd
from tqdm import tqdm
from evaluate import load

def get_answer_dataset(row):
    return row['answer'].split('####')[-1]
def get_answer_model(row):
    return row['output'].split('####')[-1]

data_feedback = load_from_disk("/scratch/workspace/wenlongzhao_umass_edu-analyze/datasets/gsm8k/feedback")
model_output_path =  '/scratch/workspace/wenlongzhao_umass_edu-analyze/outputs/gsm8k/LLaMA1B/generated_outputs.json'
df_model = pd.read_json(model_output_path)
feedback = pd.DataFrame(data_feedback)
df['model_answer']=df.apply(get_answer_model,axis=1)
feedback['GT_Answer']=feedback.apply(get_answer_dataset,axis=1)
exact_match_metric = load("exact_match")

for i in tqdm(range(df.shape[0])):
    df['score']=exact_match_metric.compute(predictions=[df.iloc[i]['model_answer']], references=[data_feedback["answer"][i].split("####")[-1]])