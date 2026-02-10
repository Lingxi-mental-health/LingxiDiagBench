#!/bin/bash

# 运行 LLM benchmark（跳过 TF-IDF）
bash ./evaluation/run_benchmark_static.sh \
    --train-file ./raw_data/LingxiDiag-16K_train_data.json \
    --test-file ./raw_data/LingxiDiag-16K_validation_data.json \
    --output-dir ./evaluation_results/static_doctor_eval_lingxi \
    --deploy-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --gpu-devices 0,1,2,3,4,5,6,7 \
    --startup-timeout 1200 \
    --next-utterance-limit 2000 
    # --skip-tfidf \
    # --skip-llm-diagnosis \
    # --skip-next-utterance