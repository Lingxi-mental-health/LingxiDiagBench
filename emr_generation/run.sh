## Example command:

## 分析数据分布和关键词
python scripts/analyze_data.py 

# 完整参数
python scripts/extract_symptoms.py \
    --data-file real_emrs/input_real_emrs.json \
    --output mapping/symptoms_mapping.json \
    --host 10.119.28.xxx \
    --port 9040 \
    --model qwen/Qwen3-32B \
    --workers 16

## 测试10个病例生成
python scripts/generate_emr.py --num 10

## 生成16000条病例并行
python scripts/generate_emr.py --num 16000 --use-llm --host 10.119.28.xxx --port 9040 --workers 32

## 修复错误数据
