#!/bin/bash
export NO_PROGRESS_BAR=true
export HF_HOME='/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache'
export HF_TOKEN=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
hostname
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,token=${HF_TOKEN},cache_dir='/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache',dtype='bfloat16' \
    --tasks gsm8k_cot_llama \
    --device cuda:0 \
    --batch_size 64 \
    --seed 42 \
    --output_path /scratch/workspace/wenlongzhao_umass_edu-analyze/lm-evaluation-harness/Results/LLaMA1B  \
    --log_samples \
    --apply_chat_template  \
    --fewshot_as_multiturn \
    --num_fewshot 8 
    