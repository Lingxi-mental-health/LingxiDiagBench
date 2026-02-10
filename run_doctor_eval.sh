#!/bin/bash

# ============================================================
# Doctor Agent 评估脚本
# 包含 LLM-as-Judge 评估 + Dynamic 诊断验证（2class/4class/12class）
# ============================================================

# 使用批量评估（推荐用于多模型对比实验）
# 自动部署不同doctor模型，运行评估，结果汇总到Excel
python evaluation/batch_doctor_eval.py \
    --doctor-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --doctor-version base \
    --patient-model "Qwen3-32B@10.119.29.220:9052" \
    --patient-version v3 \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/dynamic_doctor_eval \
    --output-excel ../evaluation_results/dynamic_doctor_eval/doctor_eval_summary.xlsx \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --max-turns 50 \
    --limit 200

python evaluation/batch_doctor_eval.py \
    --doctor-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --doctor-version mdd5k \
    --patient-model "Qwen3-32B@10.119.29.220:9052" \
    --patient-version v3 \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/dynamic_doctor_eval \
    --output-excel ../evaluation_results/dynamic_doctor_eval/doctor_eval_summary.xlsx \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --max-turns 50 \
    --limit 200

python evaluation/batch_doctor_eval.py \
    --doctor-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --doctor-version v2 \
    --patient-model "Qwen3-32B@10.119.29.220:9052" \
    --patient-version v3 \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/dynamic_doctor_eval \
    --output-excel ../evaluation_results/dynamic_doctor_eval/doctor_eval_summary.xlsx \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --max-turns 50 \
    --limit 200

python evaluation/batch_doctor_eval.py \
    --doctor-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --doctor-version v3 \
    --patient-model "Qwen3-32B@10.119.29.220:9052" \
    --patient-version v3 \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/dynamic_doctor_eval \
    --output-excel ../evaluation_results/dynamic_doctor_eval/doctor_eval_summary.xlsx \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --max-turns 50 \
    --limit 200
