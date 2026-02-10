#!/bin/bash

# ============================================================
# Patient Agent 评估脚本
# 支持单模型评估和批量多模型评估
# ============================================================


# 混合评估（本地模型 + OpenRouter模型）

python evaluation/batch_patient_eval.py \
    --patient-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --patient-version mddd5k \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/static_patient_eval \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --eval-interval 5 \
    --limit 100 \
    --skip-failed


# python evaluation/batch_patient_eval.py \
#     --patient-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
#     --patient-version cot \
#     --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
#     --output-dir ./evaluation_results/static_patient_eval \
#     --gpu-devices "0,1,2,3,4,5,6,7" \
#     --port 9060 \
#     --max-workers 32 \
#     --eval-interval 5 \
#     --limit 100 \
#     --skip-failed

python evaluation/batch_patient_eval.py \
    --patient-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Baichuan-M2-32B,Baichuan-M3-235B,openai/gpt-oss-20b,deepseek/deepseek-v3.2,google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" \
    --patient-version v3 \
    --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
    --output-dir ./evaluation_results/static_patient_eval \
    --gpu-devices "0,1,2,3,4,5,6,7" \
    --port 9060 \
    --max-workers 32 \
    --eval-interval 5 \
    --limit 100 \
    --skip-failed


# ============================================================
# 重新评估已有结果（当评估模型 IP 填错导致评估失败时使用）
# ============================================================

# 模式6: 从已有 JSON 文件重新评估（修复 IP 错误）
# 从指定目录读取所有 patient_eval_*.json 文件，按 patient_model 分组后重新评估
# 同一个 patient_model 的多个文件会自动合并去重，三个评估模型的结果会生成一个聚合报告
# python evaluation/reevaluate_patient_results.py \
#     --input-dir ./evaluation_results/static_patient_eval \
#     --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
#     --output-dir ./evaluation_results/static_patient_eval \
#     --max-workers 16

# 模式7: 重新评估指定的文件（使用 glob 模式）
# python evaluation/reevaluate_patient_results.py \
#     --input-files "./evaluation_results/static_patient_eval/patient_eval_v3_*20260115*.json" \
#     --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
#     --output-dir ./evaluation_results/static_patient_eval \
#     --max-workers 16

# 模式8: 重新评估多个指定文件（逗号分隔）
# 会自动按 patient_model 分组，每组生成一个包含三个评估模型的聚合报告
# python evaluation/reevaluate_patient_results.py \
#     --input-files "./evaluation_results/static_patient_eval/patient_eval_v3_google_gemini-3-flash-preview_*.json,./evaluation_results/static_patient_eval/patient_eval_mdd5k_*.json" \
#     --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
#     --output-dir ./evaluation_results/static_patient_eval \
#     --max-workers 16
