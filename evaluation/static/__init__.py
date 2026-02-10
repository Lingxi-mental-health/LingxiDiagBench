"""
静态精神疾病诊断Benchmark模块

该模块包含以下功能:
1. 辅助诊断: 通过医患对话预测疾病类别 (12个ICD-code大类别)
   - 评测: 2分类、4分类、12分类
   - 指标: Macro/Weighted Precision, Recall, F1, Exact Match, Top-1, Top-3 Acc
   - 方法: TF-IDF, BERT微调, LLM Zero-shot

2. 医生提问下一句预测: 给定上文预测医生的下一句提问
   - 指标: BLEU, RougeL, BertScore (cos similarity)
"""

__version__ = "1.0.0"
__author__ = "Lingxi Team"

