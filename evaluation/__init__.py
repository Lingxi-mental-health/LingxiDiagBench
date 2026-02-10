"""
Agent 统一评估模块

包含：
- llm_client: LLM 客户端模块，支持 vLLM 和 OpenRouter
- unified_patient_eval: Patient Agent 统一评估程序，支持6个指标评估
- unified_doctor_eval: Doctor Agent 统一评估程序，支持7个指标评估
- aggregate_results: 多模型评估结果聚合程序

Patient Agent 评估的6个指标：
1. 准确性 (Accuracy) - 评估患者回复是否与已知病例信息一致
2. 诚实性 (Honesty) - 评估患者在信息不存在时是否诚实回答
3. 回复简洁度 (Response_Brevity) - 评估回复长度是否像真人
4. 信息主动性 (Information_Proactivity) - 评估是否过度主动提供信息
5. 情感表达度 (Emotional_Restraint) - 评估情感表达是否克制
6. 语言修饰度 (Language_Polish) - 评估语言是否过度修饰

Doctor Agent 评估的7个指标：
1. 信息完整性 (Information_Completeness) - 是否系统性收集主诉、现病史、既往史等
2. 症状探索深度 (Symptom_Exploration_Depth) - 是否对症状进行追问、澄清、量化
3. 鉴别诊断意识 (Differential_Diagnosis_Awareness) - 是否考虑排除标准、共病筛查
4. 风险评估 (Risk_Assessment) - 是否筛查自杀/自伤意念、暴力风险
5. 沟通质量 (Communication_Quality) - 开放式提问、语言清晰、节奏适当
6. 共情与关系建立 (Empathy_and_Rapport) - 情绪验证、支持性回应
7. 问诊效率 (Consultation_Efficiency) - 信息获取效率、逻辑顺序
"""

from .llm_client import (
    create_llm_client,
    VLLMClient,
    OpenRouterClient,
    PromptLoader,
    parse_evaluation_result,
    parse_realness_multi_result,
    RealnessMultiEvaluation,
    Evaluation,
)

from .unified_patient_eval import (
    UnifiedPatientEvaluator,
    TurnEvaluationResult,
    SampleEvaluationResult,
    EvalTask,
    extract_dialogue_turns,
    extract_patient_info,
    format_dialogue_history,
    load_data,
    compute_statistics,
    generate_patient_reply,
    get_patient_agent,
    prepare_eval_tasks,
    evaluate_tasks_parallel,
    PATIENT_VERSION_CHOICES,
)

from .unified_doctor_eval import (
    UnifiedDoctorEvaluator,
    DoctorEvaluationResult,
    DoctorEvaluation,
    DoctorEvalMetrics,
    DoctorMetricScore,
    DOCTOR_VERSION_CHOICES,
)

__all__ = [
    # LLM 客户端
    "create_llm_client",
    "VLLMClient",
    "OpenRouterClient",
    "PromptLoader",
    "parse_evaluation_result",
    "parse_realness_multi_result",
    "RealnessMultiEvaluation",
    "Evaluation",
    # 统一评估
    "UnifiedPatientEvaluator",
    "TurnEvaluationResult",
    "SampleEvaluationResult",
    "EvalTask",
    "extract_dialogue_turns",
    "extract_patient_info",
    "format_dialogue_history",
    "load_data",
    "compute_statistics",
    "generate_patient_reply",
    "get_patient_agent",
    "prepare_eval_tasks",
    "evaluate_tasks_parallel",
    "PATIENT_VERSION_CHOICES",
    # 结果聚合
    "compute_single_model_stats",
    "aggregate_multi_model_results",
    "load_result_file",
    # Doctor Agent 评估
    "UnifiedDoctorEvaluator",
    "DoctorEvaluationResult",
    "DoctorEvaluation",
    "DoctorEvalMetrics",
    "DoctorMetricScore",
    "DOCTOR_VERSION_CHOICES",
]
