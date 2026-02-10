#!/bin/bash

# for evaluation used model
CUDA_VISIBLE_DEVICES=0,3 nohup python -m vllm.entrypoints.openai.api_server \
  --model /tcci_mnt/zhoutiancheng/models/LLM-Research/gemma-3-27b-it/ \
  --served-model-name gemma-3-27b-it \
  --port 9051 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --dtype bfloat16 > /tcci_mnt/shihao/logs/gemma_27b.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
  --model /tcci_mnt/zhoutiancheng/models/qwen/qwen3-30b-a3b-instruct-2507/ \
  --served-model-name qwen3-30b \
  --port 9041 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser deepseek_r1 \
  --max-model-len 16384 \
  --dtype bfloat16 > /tcci_mnt/shihao/logs/qwen_30b.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
  --model /tcci_mnt/shihao/models/gpt-oss-20b \
  --served-model-name gpt-oss-20b \
  --port 9042 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 > /tcci_mnt/shihao/logs/gpt_oss_20b.log 2>&1 &


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server \
  --model /tcci_mnt/shihao/models/Qwen3-32B \
  --served-model-name Qwen3-32B \
  --port 9052 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser deepseek_r1 \
  --max-model-len 16384 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 > /tcci_mnt/shihao/logs/qwen3-32b.log 2>&1 &


# # for patient agent used model
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server \
#   --model /tcci_mnt/shihao/models/Baichuan-M3-235B \
#   --served-model-name Baichuan-M3-235B \
#   --port 9052 \
#   --host 0.0.0.0 \
#   --trust-remote-code \
#   --gpu-memory-utilization 0.95 \
#   --reasoning-parser deepseek_r1 \
#   --max-model-len 16384 \
#   --tensor-parallel-size 8 \
#   --dtype bfloat16 > /tcci_mnt/shihao/logs/Baichuan-M3-235B.log 2>&1 &
