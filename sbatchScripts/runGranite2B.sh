#!/bin/bash
export NO_PROGRESS_BAR=true
export HF_HOME='/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache'
export HF_TOKEN=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
hostname
lm_eval --model hf \
    --model_args pretrained=ibm-granite/granite-3.0-2b-instruct,token=${HF_TOKEN},cache_dir='/scratch/workspace/wenlongzhao_umass_edu-analyze/Bidirectional-Teacher-Student-Communication-for-LLM-Alignment/transformers_cache' \
    --tasks gsm8k_cot \
    --device cuda:0 \
    --batch_size 64