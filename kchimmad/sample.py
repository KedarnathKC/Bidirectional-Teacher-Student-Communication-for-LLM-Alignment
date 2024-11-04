import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/workspace/wenlongzhao_umass_edu-analyze/transformers_cache'

hf_token = "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"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
config = AutoConfig.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token, config=config)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, config=config)

model.eval()

# Input prompt
input_text = "What are the benefits of using large language models?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
with torch.no_grad():
    output = model.generate(**inputs, max_length=100, num_return_sequences=1)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)