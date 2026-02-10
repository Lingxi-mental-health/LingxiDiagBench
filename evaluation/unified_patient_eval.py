#!/usr/bin/env python3
"""
统一的 Patient Agent 评估程序（优化版 - 两阶段并行评估）

评估指标（6个维度）：
1. 准确性 (Accuracy) - 评估患者回复是否与已知病例信息一致
2. 诚实性 (Honesty) - 评估患者在信息不存在时是否诚实回答
3. 回复简洁度 (Response_Brevity) - 评估回复长度是否像真人
4. 信息主动性 (Information_Proactivity) - 评估是否过度主动提供信息
5. 情感表达度 (Emotional_Restraint) - 评估情感表达是否克制
6. 语言修饰度 (Language_Polish) - 评估语言是否过度修饰

优化的评估流程（两阶段并行）：
- 阶段1：并行准备所有评估任务（包括并行生成 Patient Agent 回复）
- 阶段2：多模型并行评估所有任务

支持两种评估模式：
1. 原始回复模式 (--use-original-reply): 评估数据集中真实的患者回复
2. Patient Agent 模式: 使用 Patient Agent 生成回复并评估

使用方法:
    # 模式1: 使用原始数据集中的患者回复（评估真实对话数据）
    python unified_patient_eval.py \\
        --data-file /path/to/data.json \\
        --use-original-reply \\
        --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041" \\
        --max-workers 16 \\
        --limit 10

    # 模式2: 使用 Patient Agent 生成回复（评估 Patient Agent）
    python unified_patient_eval.py \\
        --data-file /path/to/data.json \\
        --patient-model "qwen3-30b@10.119.28.185:9041" \\
        --patient-version v1 \\
        --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041" \\
        --max-workers 16 \\
        --limit 10
        
    # 支持的 Patient 版本:
    # - v1: 基础版本 (src/patient/patient_v1.py)
    # - cot: Chain-of-Thought 版本 (src/patient/patient_cot.py)
    # - mdd5k: MDD-5K 版本 (src/patient/patient_mdd5k.py)
    # - v3: 优化版本 (src/patient/patient_v3.py)
"""

import argparse
import json
import math
import os
import sys
import re
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到路径（必须在导入 evaluation 模块之前）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.llm_client import extract_reasoning_content, extract_message_text

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from evaluation.llm_client import (
    create_llm_client,
    PromptLoader,
    parse_evaluation_result,
    parse_realness_multi_result,
    RealnessMultiEvaluation,
    extract_message_text,
)


# ============================================================
# Patient Agent 导入 (延迟加载，带线程锁)
# ============================================================

_patient_classes_cache: Dict[str, Any] = {}
_patient_classes_lock = threading.Lock()
_patient_classes_loaded = False

def _load_patient_class(version: str):
    """
    按需加载指定版本的 Patient 类（线程安全）
    
    Args:
        version: 患者版本 ("v1", "cot", "mdd5k", "v3")
    
    Returns:
        Patient 类，如果加载失败则返回 None
    """
    global _patient_classes_cache
    
    with _patient_classes_lock:
        # 已经加载过，直接返回
        if version in _patient_classes_cache:
            return _patient_classes_cache[version]
        
        # 按需加载指定版本
        patient_cls = None
        try:
            if version == "v1":
                from src.patient.patient_v1 import Patient as _PatientV1
                patient_cls = _PatientV1
            elif version == "cot":
                from src.patient.patient_cot import Patient as _PatientCoT
                patient_cls = _PatientCoT
            elif version == "mdd5k":
                from src.patient.patient_mdd5k import Patient as _PatientMDD5K
                patient_cls = _PatientMDD5K
            elif version == "v3":
                from src.patient.patient_v3 import Patient as _PatientV3
                patient_cls = _PatientV3
            else:
                print(f"[WARNING] 不支持的 Patient 版本: {version}")
                return None
            
            _patient_classes_cache[version] = patient_cls
            print(f"[PatientLoader] 成功加载 Patient{version.upper()}")
            return patient_cls
            
        except Exception as e:
            print(f"[WARNING] 无法加载 Patient{version.upper()}: {e}")
            _patient_classes_cache[version] = None
            return None


def get_patient_class(version: str):
    """获取指定版本的 Patient 类"""
    return _load_patient_class(version)


PATIENT_VERSION_CHOICES = ["v1", "cot", "mdd5k", "v3"]


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class TurnEvaluationResult:
    """单轮对话的评估结果"""
    turn_index: int
    doctor_question: str
    patient_reply: str
    # 准确性/诚实性评估（互斥）
    information_exists: bool = True
    accuracy_score: Optional[int] = None
    accuracy_reason: str = ""
    honesty_score: Optional[int] = None
    honesty_reason: str = ""
    # 真实性多维度评估
    realness_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    realness_overall_impression: str = "Unknown"
    realness_average_score: Optional[float] = None
    # Token 使用量
    tokens: Tuple[int, int] = (0, 0)
    # 错误信息
    error: Optional[str] = None


@dataclass
class SampleEvaluationResult:
    """单个样本的评估结果"""
    patient_id: str
    total_turns: int
    evaluated_turns: int
    turn_results: List[TurnEvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 工具函数
# ============================================================

def calc_average(scores: List[Optional[float]]) -> float:
    """计算平均值"""
    valid = [s for s in scores if isinstance(s, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0


def calc_std(scores: List[Optional[float]]) -> float:
    """计算标准差"""
    valid = [float(s) for s in scores if isinstance(s, (int, float))]
    if len(valid) < 2:
        return 0.0
    mean = sum(valid) / len(valid)
    variance = sum((x - mean) ** 2 for x in valid) / len(valid)
    return math.sqrt(variance)


def sanitize_model_name(model: str) -> str:
    """将模型名转换为适合作为文件名的安全字符串"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return safe or "model"


def extract_dialogue_turns(cleaned_text: str) -> List[Dict[str, str]]:
    """
    从清洗后的对话文本中提取对话轮次
    
    支持两种格式：
    1. 明确标记格式：以 "医生：" 和 "患者：" 开头
    2. 未知发言人格式：以 "未知发言人：" 开头
    """
    turns = []
    current_turn = {"doctor": "", "patient": ""}
    
    if not cleaned_text:
        return turns
    
    lines = cleaned_text.strip().split('\n')
    
    # 检查是否全部是 "未知发言人：" 格式
    unknown_speaker_lines = []
    has_explicit_speaker = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("医生：") or line.startswith("医生:") or \
           line.startswith("患者：") or line.startswith("患者:"):
            has_explicit_speaker = True
            break
        if line.startswith("未知发言人：") or line.startswith("未知发言人:"):
            unknown_speaker_lines.append(line)
    
    # 如果全部是 "未知发言人：" 格式
    if not has_explicit_speaker and unknown_speaker_lines:
        for idx, line in enumerate(unknown_speaker_lines):
            content = line[6:].strip()
            
            if idx % 2 == 0:
                if current_turn["doctor"] or current_turn["patient"]:
                    turns.append(current_turn.copy())
                current_turn = {"doctor": content, "patient": ""}
            else:
                current_turn["patient"] = content
        
        if current_turn["doctor"] or current_turn["patient"]:
            turns.append(current_turn)
        
        return turns
    
    # 明确标记处理逻辑
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("医生：") or line.startswith("医生:"):
            if current_turn["doctor"] or current_turn["patient"]:
                turns.append(current_turn.copy())
                current_turn = {"doctor": "", "patient": ""}
            current_turn["doctor"] = line[3:].strip()
        elif line.startswith("患者：") or line.startswith("患者:"):
            current_turn["patient"] = line[3:].strip()
        elif line.startswith("未知发言人：") or line.startswith("未知发言人:"):
            content = line[6:].strip()
            if not current_turn["doctor"]:
                current_turn["doctor"] = content
            else:
                current_turn["patient"] = content
        elif line.startswith("家属"):
            continue
        else:
            if current_turn["patient"]:
                current_turn["patient"] += " " + line
            elif current_turn["doctor"]:
                current_turn["doctor"] += " " + line
    
    if current_turn["doctor"] or current_turn["patient"]:
        turns.append(current_turn)
    
    return turns


def format_dialogue_history(turns: List[Dict[str, str]], exclude_last_n: int = 1) -> str:
    """格式化对话历史"""
    if not turns:
        return "（无对话历史）"
    
    history_turns = turns[:-exclude_last_n] if exclude_last_n > 0 else turns
    
    lines = []
    for idx, turn in enumerate(history_turns, start=1):
        doctor = turn.get("doctor", "").strip()
        patient = turn.get("patient", "").strip()
        if doctor:
            lines.append(f"医生: {doctor}")
        if patient:
            lines.append(f"患者: {patient}")
        lines.append("")
    
    return "\n".join(lines).strip() or "（无对话历史）"


def extract_patient_info(sample: Dict[str, Any]) -> str:
    """从样本中提取患者信息摘要"""
    parts = []
    
    age = sample.get("Age") or sample.get("age")
    gender = sample.get("Gender") or sample.get("gender")
    dept = sample.get("Department") or sample.get("department")
    
    if age is not None:
        parts.append(f"年龄:{age}")
    if gender:
        parts.append(f"性别:{gender}")
    if dept:
        parts.append(f"科室:{dept}")
    
    cc = sample.get("ChiefComplaint") or sample.get("主诉", "")
    if cc:
        parts.append(f"主诉:{cc}")
    
    pi = sample.get("PresentIllnessHistory") or sample.get("现病史", "")
    if pi:
        parts.append(f"现病史:{pi}")
    
    ph = sample.get("PersonalHistory") or sample.get("个人史", "")
    if ph:
        parts.append(f"个人史:{ph}")
    
    fh = sample.get("FamilyHistory") or sample.get("家族史", "")
    if fh:
        parts.append(f"家族史:{fh}")
    
    diag = sample.get("Diagnosis") or sample.get("诊断", "")
    if diag:
        parts.append(f"诊断:{diag}")
    
    patient_info = sample.get("Patient info") or sample.get("patient_info", "")
    if patient_info and not parts:
        parts.append(f"病例:{patient_info}")
    
    return " | ".join(str(p).strip() for p in parts if str(p).strip())


def strip_reasoning_from_text(text: str) -> str:
    """去除文本中的 <think>...</think> 标记"""
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_patient_agent(
    patient_version: str,
    case_info: Dict[str, Any],
    model_name: str,
    use_api: bool = True,
):
    """
    获取 Patient Agent 实例
    
    Args:
        patient_version: 患者版本 ("v1", "cot", "mdd5k", "v3")
        case_info: 病例信息
        model_name: 模型名称
        use_api: 是否使用 API
    
    Returns:
        Patient Agent 实例
    """
    patient_cls = get_patient_class(patient_version)
    
    if not patient_cls:
        raise ValueError(f"不支持的 patient_version: {patient_version}")
    
    return patient_cls(patient_template=case_info, model_path=model_name, use_api=use_api)


def generate_patient_reply(
    sample: Dict[str, Any],
    dialogue_history: str,
    current_doctor_question: str,
    patient_model: str,
    patient_version: str,
    use_api: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    使用 Patient Agent 生成患者回复
    
    Args:
        sample: 样本数据（包含病例信息）
        dialogue_history: 对话历史
        current_doctor_question: 当前医生问题
        patient_model: 患者模型名称
        patient_version: 患者版本
        use_api: 是否使用 API
    
    Returns:
        Tuple[str, Optional[str]]: (患者回复, reasoning)
    """
    try:
        agent = get_patient_agent(patient_version, sample, patient_model, use_api=use_api)
        
        # 调用患者回复生成
        response = agent.patient_response_gen(
            current_topic=None,
            dialogue_history=dialogue_history,
            current_doctor_question=current_doctor_question,
        )
        
        # 处理返回值（可能是 tuple）
        if isinstance(response, tuple):
            reply = response[0]
            # 尝试获取 reasoning（第4个元素）
            reasoning = response[3] if len(response) > 3 else None
        else:
            reply = response
            reasoning = None
        
        # 去掉 <think>...</think> 部分
        if reply:
            reply = strip_reasoning_from_text(reply)
        
        return (reply or "").strip(), reasoning
        
    except Exception as exc:
        print(f"[PatientAgent] 生成回复失败: {exc}")
        return "", None


# ============================================================
# 评估器类
# ============================================================

class UnifiedPatientEvaluator:
    """统一的患者回复评估器"""
    
    # 真实性评估的子维度
    REALNESS_METRICS = [
        "Response_Brevity",
        "Information_Proactivity",
        "Emotional_Restraint",
        "Language_Polish",
        "Conversational_Cooperation",
    ]
    
    # 输出所需的6个维度
    OUTPUT_METRICS = [
        "accuracy",                   # 准确性
        "honesty",                    # 诚实性
        "Response_Brevity",           # 回复简洁度
        "Information_Proactivity",    # 信息主动性
        "Emotional_Restraint",        # 情感表达度
        "Language_Polish",            # 语言修饰度
    ]
    
    def __init__(
        self,
        eval_model: str,
        prompts_dir: str = None,
        api_key: str = None,
    ):
        """
        初始化评估器
        
        Args:
            eval_model: 评估模型名称
            prompts_dir: prompts 目录路径
            api_key: API Key（仅用于 OpenRouter）
        """
        self.client = create_llm_client(eval_model, api_key=api_key)
        if prompts_dir is None:
            prompts_dir = PROJECT_ROOT / "prompts" / "evaluation"
        self.prompt_loader = PromptLoader(str(prompts_dir))
        self.eval_model = eval_model
    
    def evaluate_information_existence(
        self,
        doctor_question: str,
        patient_info: str,
        dialogue_history: str,
    ) -> Tuple[bool, str, Tuple[int, int]]:
        """
        评估医生询问的信息是否存在于已知资料中
        """
        prompt = f"""你是一个专业的医疗信息分析助手，请判断医生询问的信息是否存在于已知的病例信息和医患对话记录中。

【病例】：{patient_info}
【既往对话】：{dialogue_history}
【当前医生问题】：{doctor_question}

评估标准：
根据目前的患者信息判断是否可以回答医生关于病情的问题。

评判原则：
- 如果医生询问的信息在病例信息或医患对话记录中有明确记录或相关提及或可以回答，只输出"存在"
- 如果医生询问的信息在病例信息和医患对话记录中都没有涉及或记录，无法回答，输出"不存在"

只输出"存在"或"不存在"，不要其他内容。"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的医疗信息分析助手"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, tokens = self.client.chat_completion(
                messages,
                temperature=0.1,
                use_json_format=False,
            )
            
            exists = response.startswith("存在") or ("存在" in response and "不存在" not in response)
            return exists, response, tokens
            
        except Exception as e:
            print(f"警告：信息存在性评估失败: {e}")
            return True, str(e), (0, 0)
    
    def evaluate_accuracy(
        self,
        patient_reply: str,
        patient_info: str,
        dialogue_history: str,
        doctor_question: str,
    ) -> Tuple[Optional[int], str, Tuple[int, int]]:
        """
        评估患者回复的准确性
        """
        prompt = self.prompt_loader.format_prompt(
            "factual_accuracy",
            patient_info=patient_info,
            dialogue_history=dialogue_history,
            doctor_question=doctor_question,
            patient_reply=patient_reply,
        )
        
        messages = [
            {"role": "system", "content": "你是一个专业的医疗信息评估助手"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, tokens = self.client.chat_completion(
                messages,
                temperature=0.1,
                use_json_format=True,
            )
            
            parsed = parse_evaluation_result(response)
            return parsed.get("score"), parsed.get("reason", ""), tokens
            
        except Exception as e:
            print(f"警告：准确性评估失败: {e}")
            return None, str(e), (0, 0)
    
    def evaluate_honesty(
        self,
        patient_reply: str,
        patient_info: str,
        dialogue_history: str,
        doctor_question: str,
    ) -> Tuple[Optional[int], str, Tuple[int, int]]:
        """
        评估患者回复的诚实性
        """
        prompt = self.prompt_loader.format_prompt(
            "honesty",
            patient_info=patient_info,
            dialogue_history=dialogue_history,
            doctor_question=doctor_question,
            patient_reply=patient_reply,
        )
        
        messages = [
            {"role": "system", "content": "你是一个专业的医疗信息评估助手"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, tokens = self.client.chat_completion(
                messages,
                temperature=0.1,
                use_json_format=True,
            )
            
            parsed = parse_evaluation_result(response)
            return parsed.get("score"), parsed.get("reason", ""), tokens
            
        except Exception as e:
            print(f"警告：诚实性评估失败: {e}")
            return None, str(e), (0, 0)
    
    def evaluate_realness_multi(
        self,
        doctor_question: str,
        patient_reply: str,
    ) -> Tuple[Dict[str, Any], str, Optional[float], Tuple[int, int]]:
        """
        使用多维度真实性评估（一次调用评估多个维度）
        """
        prompt = self.prompt_loader.format_prompt(
            "realness_multi",
            doctor_question=doctor_question,
            patient_reply=patient_reply,
        )
        
        messages = [
            {"role": "system", "content": "你是一位精通语言学和临床心理学的专家。请严格按照 JSON 格式输出。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, tokens, raw_response = self.client.chat_completion(
                messages,
                temperature=0.6,
                response_model=RealnessMultiEvaluation,
                return_raw_response=True,
            )
            
            parsed = parse_realness_multi_result(response)
            if self._is_realness_parse_failed(parsed):
                self._log_realness_parse_failure(
                    doctor_question=doctor_question,
                    patient_reply=patient_reply,
                    parsed=parsed,
                    response=response,
                    raw_response=raw_response,
                    tokens=tokens,
                )
            
            return (
                parsed.get("metrics", {}),
                parsed.get("overall_impression", "Unknown"),
                parsed.get("average_score"),
                tokens,
            )
            
        except Exception as e:
            print(f"警告：真实性评估失败: {e}")
            return {}, "Unknown", None, (0, 0)

    def _is_realness_parse_failed(self, parsed: Dict[str, Any]) -> bool:
        """判断真实性评估解析是否失败"""
        if not isinstance(parsed, dict):
            return True
        if parsed.get("overall_impression") == "Unknown":
            return True
        if parsed.get("average_score") is None:
            return True
        metrics = parsed.get("metrics") or {}
        for metric in self.REALNESS_METRICS:
            score = (metrics.get(metric) or {}).get("score")
            if score is None:
                return True
        return False

    def _log_realness_parse_failure(
        self,
        doctor_question: str,
        patient_reply: str,
        parsed: Dict[str, Any],
        response: Any,
        raw_response: Any,
        tokens: Tuple[int, int],
    ) -> None:
        """记录真实性评估解析失败日志"""
        log_dir = PROJECT_ROOT / "evaluation_results" / "parse_failures"
        log_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"realness_multi_{sanitize_model_name(self.eval_model)}_{date_tag}.jsonl"

        response_text = ""
        if response is not None:
            if hasattr(response, "model_dump_json"):
                response_text = response.model_dump_json()
            else:
                response_text = str(response)

        raw_response_text = ""
        if raw_response is not None:
            try:
                message = raw_response.choices[0].message
                raw_response_text = extract_message_text(message)
            except Exception:
                raw_response_text = ""

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.eval_model,
            "doctor_question": doctor_question,
            "patient_reply": patient_reply,
            "parsed": parsed,
            "response_text": response_text,
            "raw_response_text": raw_response_text,
            "tokens": {"prompt": tokens[0], "completion": tokens[1]},
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def evaluate_turn(
        self,
        turn_index: int,
        doctor_question: str,
        patient_reply: str,
        patient_info: str,
        dialogue_history: str,
    ) -> TurnEvaluationResult:
        """
        评估单轮对话
        """
        result = TurnEvaluationResult(
            turn_index=turn_index,
            doctor_question=doctor_question,
            patient_reply=patient_reply,
        )
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        try:
            # 1. 判断信息存在性
            exists, _, exist_tokens = self.evaluate_information_existence(
                doctor_question, patient_info, dialogue_history
            )
            result.information_exists = exists
            total_prompt_tokens += exist_tokens[0]
            total_completion_tokens += exist_tokens[1]
            
            # 2. 根据信息存在性评估准确性或诚实性
            if exists:
                score, reason, acc_tokens = self.evaluate_accuracy(
                    patient_reply, patient_info, dialogue_history, doctor_question
                )
                result.accuracy_score = score
                result.accuracy_reason = reason
                result.honesty_reason = "信息存在于已知资料中，无需评估诚实性"
                total_prompt_tokens += acc_tokens[0]
                total_completion_tokens += acc_tokens[1]
            else:
                score, reason, hon_tokens = self.evaluate_honesty(
                    patient_reply, patient_info, dialogue_history, doctor_question
                )
                result.honesty_score = score
                result.honesty_reason = reason
                result.accuracy_reason = "信息不存在于已知资料中，无需评估准确性"
                total_prompt_tokens += hon_tokens[0]
                total_completion_tokens += hon_tokens[1]
            
            # 3. 评估真实性多维度
            metrics, impression, avg_score, real_tokens = self.evaluate_realness_multi(
                doctor_question, patient_reply
            )
            result.realness_metrics = metrics
            result.realness_overall_impression = impression
            result.realness_average_score = avg_score
            total_prompt_tokens += real_tokens[0]
            total_completion_tokens += real_tokens[1]
            
            result.tokens = (total_prompt_tokens, total_completion_tokens)
            
        except Exception as e:
            result.error = str(e)
        
        return result
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        eval_interval: int = 5,
        use_original_reply: bool = True,
        patient_model: Optional[str] = None,
        patient_version: str = "v1",
    ) -> SampleEvaluationResult:
        """
        评估单个样本（间隔采样）
        
        Args:
            sample: 样本数据
            eval_interval: 采样间隔
            use_original_reply: 是否使用原始回复（True），否则使用 Patient Agent 生成
            patient_model: Patient Agent 使用的模型（仅当 use_original_reply=False 时有效）
            patient_version: Patient Agent 版本（仅当 use_original_reply=False 时有效）
        """
        patient_id = str(sample.get("patient_id", "unknown"))
        cleaned_text = sample.get("cleaned_text", "")
        
        # 提取对话轮次
        turns = extract_dialogue_turns(cleaned_text)
        
        if not turns:
            return SampleEvaluationResult(
                patient_id=patient_id,
                total_turns=0,
                evaluated_turns=0,
                metadata={"error": "无有效对话轮次"}
            )
        
        # 提取患者信息
        patient_info = extract_patient_info(sample)
        
        # 选择要评估的轮次索引
        sampled_indices = list(range(eval_interval - 1, len(turns), eval_interval))
        if not sampled_indices and turns:
            sampled_indices = [len(turns) - 1]
        
        result = SampleEvaluationResult(
            patient_id=patient_id,
            total_turns=len(turns),
            evaluated_turns=len(sampled_indices),
            turn_results=[],
            metadata={
                "age": sample.get("Age"),
                "gender": sample.get("Gender"),
                "diagnosis": sample.get("Diagnosis"),
                "sampled_indices": sampled_indices,
                "eval_interval": eval_interval,
                "use_original_reply": use_original_reply,
                "patient_model": patient_model,
                "patient_version": patient_version,
            }
        )
        
        # 评估每个采样的轮次
        for turn_idx in sampled_indices:
            turn = turns[turn_idx]
            doctor_question = turn.get("doctor", "")
            
            # 构建该轮次之前的对话历史
            dialogue_history = format_dialogue_history(turns[:turn_idx + 1], exclude_last_n=1)
            
            # 获取患者回复
            if use_original_reply:
                # 使用原始回复
                patient_reply = turn.get("patient", "")
                generated_reasoning = None
            else:
                # 使用 Patient Agent 生成回复
                if not patient_model:
                    print(f"[警告] patient_model 未指定，跳过轮次 {turn_idx}")
                    continue
                
                patient_reply, generated_reasoning = generate_patient_reply(
                    sample=sample,
                    dialogue_history=dialogue_history,
                    current_doctor_question=doctor_question,
                    patient_model=patient_model,
                    patient_version=patient_version,
                    use_api=True,
                )
                
                if not patient_reply:
                    raise ValueError(f"Patient Agent 生成回复失败: {patient_model}, {patient_version}")
            
            if not patient_reply:
                continue
            
            turn_result = self.evaluate_turn(
                turn_index=turn_idx,
                doctor_question=doctor_question,
                patient_reply=patient_reply,
                patient_info=patient_info,
                dialogue_history=dialogue_history,
            )
            
            result.turn_results.append(turn_result)
        
        return result


# ============================================================
# 批量评估函数
# ============================================================

@dataclass
class EvalTask:
    """评估任务数据结构"""
    patient_id: str
    sample_idx: int
    turn_idx: int
    doctor_question: str
    patient_reply: str
    patient_info: str
    dialogue_history: str
    sample: Dict[str, Any]


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "data" in data:
            return data["data"]
        elif "samples" in data:
            return data["samples"]
        elif "cases" in data:
            return data["cases"]
    
    return []


# ============================================================
# 阶段1：并行生成 Patient Agent 回复
# ============================================================

def prepare_eval_tasks(
    samples: List[Dict[str, Any]],
    eval_interval: int = 5,
    use_original_reply: bool = True,
    patient_model: Optional[str] = None,
    patient_version: str = "v1",
    max_workers: int = 8,
) -> List[EvalTask]:
    """
    准备所有评估任务，包括并行生成 Patient Agent 回复
    
    Args:
        samples: 样本列表
        eval_interval: 采样间隔
        use_original_reply: 是否使用原始回复
        patient_model: Patient Agent 模型
        patient_version: Patient Agent 版本
        max_workers: 并行生成回复的最大线程数
    
    Returns:
        评估任务列表
    """
    # 第一步：收集所有需要处理的任务
    pending_tasks = []  # (sample_idx, sample, turn_idx, doctor_question, dialogue_history, patient_info)
    
    print(f"\n[阶段1] 准备评估任务...")
    
    for sample_idx, sample in enumerate(samples):
        patient_id = str(sample.get("patient_id", "unknown"))
        cleaned_text = sample.get("cleaned_text", "")
        
        # 提取对话轮次
        turns = extract_dialogue_turns(cleaned_text)
        if not turns:
            continue
        
        # 提取患者信息
        patient_info = extract_patient_info(sample)
        
        # 选择要评估的轮次索引
        sampled_indices = list(range(eval_interval - 1, len(turns), eval_interval))
        if not sampled_indices and turns:
            sampled_indices = [len(turns) - 1]
        
        # 收集每个轮次的任务
        for turn_idx in sampled_indices:
            turn = turns[turn_idx]
            doctor_question = turn.get("doctor", "")
            dialogue_history = format_dialogue_history(turns[:turn_idx + 1], exclude_last_n=1)
            
            if use_original_reply:
                # 使用原始回复，直接创建任务
                patient_reply = turn.get("patient", "")
                if patient_reply:
                    pending_tasks.append(EvalTask(
                        patient_id=patient_id,
                        sample_idx=sample_idx,
                        turn_idx=turn_idx,
                        doctor_question=doctor_question,
                        patient_reply=patient_reply,
                        patient_info=patient_info,
                        dialogue_history=dialogue_history,
                        sample=sample,
                    ))
            else:
                # 需要生成回复，先收集任务信息
                pending_tasks.append({
                    "patient_id": patient_id,
                    "sample_idx": sample_idx,
                    "turn_idx": turn_idx,
                    "doctor_question": doctor_question,
                    "patient_info": patient_info,
                    "dialogue_history": dialogue_history,
                    "sample": sample,
                })
    
    if use_original_reply:
        print(f"[阶段1] 使用原始回复，共 {len(pending_tasks)} 个任务")
        return pending_tasks
    
    # 第二步：预加载指定版本的 Patient 类（在主线程中完成，避免多线程重复加载）
    print(f"[阶段1] 需要生成 {len(pending_tasks)} 个 Patient Agent 回复")
    print(f"[阶段1] 使用模型: {patient_model}, 版本: {patient_version}")
    
    # 预加载 Patient 类
    patient_cls = get_patient_class(patient_version)
    if not patient_cls:
        print(f"[阶段1] 错误: 无法加载 Patient 版本 {patient_version}")
        return []
    
    print(f"[阶段1] 并行线程数: {max_workers}")
    
    eval_tasks = []
    lock = threading.Lock()
    progress = {"count": 0}
    
    def generate_reply_task(task_info: Dict[str, Any]) -> Optional[EvalTask]:
        """生成单个回复的任务"""
        try:
            patient_reply, _ = generate_patient_reply(
                sample=task_info["sample"],
                dialogue_history=task_info["dialogue_history"],
                current_doctor_question=task_info["doctor_question"],
                patient_model=patient_model,
                patient_version=patient_version,
                use_api=True,
            )
            
            if not patient_reply:
                raise ValueError(f"Patient Agent 生成回复失败: {patient_model}, {patient_version}")
            
            with lock:
                progress["count"] += 1
                if progress["count"] % 10 == 0 or progress["count"] == len(pending_tasks):
                    print(f"[阶段1] 生成进度: {progress['count']}/{len(pending_tasks)}")
            
            return EvalTask(
                patient_id=task_info["patient_id"],
                sample_idx=task_info["sample_idx"],
                turn_idx=task_info["turn_idx"],
                doctor_question=task_info["doctor_question"],
                patient_reply=patient_reply,
                patient_info=task_info["patient_info"],
                dialogue_history=task_info["dialogue_history"],
                sample=task_info["sample"],
            )
        except Exception as e:
            print(f"[阶段1] 生成回复失败: {e}")
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_reply_task, task) for task in pending_tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    eval_tasks.append(result)
            except Exception as e:
                print(f"[阶段1] 任务执行失败: {e}")
    
    print(f"[阶段1] 完成! 成功生成 {len(eval_tasks)} 个回复")
    return eval_tasks


# ============================================================
# 阶段2：并行评估所有任务
# ============================================================

def evaluate_tasks_parallel(
    eval_tasks: List[EvalTask],
    eval_models: List[str],
    max_workers: int = 8,
    api_key: Optional[str] = None,
) -> Dict[str, Dict[str, SampleEvaluationResult]]:
    """
    并行评估所有任务
    
    Args:
        eval_tasks: 评估任务列表
        eval_models: 评估模型列表
        max_workers: 每个模型的并行线程数
        api_key: API Key
    
    Returns:
        模型名称 -> (patient_id -> 评估结果) 的映射
    """
    print(f"\n[阶段2] 开始并行评估...")
    print(f"[阶段2] 评估任务数: {len(eval_tasks)}")
    print(f"[阶段2] 评估模型: {', '.join(eval_models)}")
    print(f"[阶段2] 每模型并行线程数: {max_workers}")
    
    # 初始化所有评估器
    evaluators = {}
    for model_name in eval_models:
        evaluators[model_name] = UnifiedPatientEvaluator(
            eval_model=model_name,
            api_key=api_key,
        )
    
    # 结果存储
    all_model_results: Dict[str, Dict[str, SampleEvaluationResult]] = {
        model: {} for model in eval_models
    }
    
    # 为每个模型并行评估
    def evaluate_single_model(model_name: str) -> Dict[str, SampleEvaluationResult]:
        """评估单个模型"""
        evaluator = evaluators[model_name]
        model_results: Dict[str, List[TurnEvaluationResult]] = {}
        model_metadata: Dict[str, Dict[str, Any]] = {}
        
        lock = threading.Lock()
        progress = {"count": 0}
        
        def evaluate_single_task(task: EvalTask) -> Tuple[str, int, TurnEvaluationResult]:
            """评估单个任务"""
            turn_result = evaluator.evaluate_turn(
                turn_index=task.turn_idx,
                doctor_question=task.doctor_question,
                patient_reply=task.patient_reply,
                patient_info=task.patient_info,
                dialogue_history=task.dialogue_history,
            )
            
            with lock:
                progress["count"] += 1
                if progress["count"] % 20 == 0 or progress["count"] == len(eval_tasks):
                    print(f"  [{model_name}] 评估进度: {progress['count']}/{len(eval_tasks)}")
            
            return task.patient_id, task.turn_idx, turn_result
        
        print(f"\n  [{model_name}] 开始评估 {len(eval_tasks)} 个任务...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_task, task) for task in eval_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    patient_id, turn_idx, turn_result = future.result()
                    
                    with lock:
                        if patient_id not in model_results:
                            model_results[patient_id] = []
                            # 获取 sample 的 metadata
                            task = next(t for t in eval_tasks if t.patient_id == patient_id)
                            model_metadata[patient_id] = {
                                "age": task.sample.get("Age"),
                                "gender": task.sample.get("Gender"),
                                "diagnosis": task.sample.get("Diagnosis"),
                            }
                        model_results[patient_id].append((turn_idx, turn_result))
                except Exception as e:
                    print(f"  [{model_name}] 评估失败: {e}")
        
        # 整理结果
        final_results: Dict[str, SampleEvaluationResult] = {}
        for patient_id, turn_list in model_results.items():
            # 按 turn_idx 排序
            turn_list.sort(key=lambda x: x[0])
            turn_results = [tr for _, tr in turn_list]
            
            # 获取总轮次数
            task = next(t for t in eval_tasks if t.patient_id == patient_id)
            turns = extract_dialogue_turns(task.sample.get("cleaned_text", ""))
            
            final_results[patient_id] = SampleEvaluationResult(
                patient_id=patient_id,
                total_turns=len(turns),
                evaluated_turns=len(turn_results),
                turn_results=turn_results,
                metadata=model_metadata.get(patient_id, {}),
            )
        
        print(f"  [{model_name}] 完成! 评估 {len(final_results)} 个样本")
        return final_results
    
    # 并行评估所有模型
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eval_models)) as executor:
        model_futures = {
            executor.submit(evaluate_single_model, model): model 
            for model in eval_models
        }
        
        for future in concurrent.futures.as_completed(model_futures):
            model_name = model_futures[future]
            try:
                all_model_results[model_name] = future.result()
            except Exception as e:
                print(f"[阶段2] 模型 {model_name} 评估失败: {e}")
    
    print(f"\n[阶段2] 所有模型评估完成!")
    return all_model_results


def evaluate_with_single_model(
    evaluator: UnifiedPatientEvaluator,
    samples: List[Dict[str, Any]],
    eval_interval: int = 5,
    max_workers: int = 4,
    model_name: str = "",
    use_original_reply: bool = True,
    patient_model: Optional[str] = None,
    patient_version: str = "v1",
) -> Dict[str, SampleEvaluationResult]:
    """
    使用单个模型进行评估（旧版接口，保留兼容性）
    
    Args:
        evaluator: 评估器实例
        samples: 样本列表
        eval_interval: 采样间隔
        max_workers: 最大并行线程数
        model_name: 评估模型名称（用于日志）
        use_original_reply: 是否使用原始回复
        patient_model: Patient Agent 模型
        patient_version: Patient Agent 版本
    """
    all_results = {}
    lock = threading.Lock()
    progress = {"count": 0, "turns": 0}
    model_label = f"[{model_name}] " if model_name else ""
    
    def process_sample(sample: Dict[str, Any]):
        patient_id = str(sample.get('patient_id', 'unknown'))
        result = evaluator.evaluate_sample(
            sample,
            eval_interval=eval_interval,
            use_original_reply=use_original_reply,
            patient_model=patient_model,
            patient_version=patient_version,
        )
        
        with lock:
            all_results[patient_id] = result
            progress["count"] += 1
            progress["turns"] += len(result.turn_results)
            print(f"{model_label}评估进度: {progress['count']}/{len(samples)} - patient_id: {patient_id} ({len(result.turn_results)} 轮)")
        
        return result
    
    mode_desc = "原始回复" if use_original_reply else f"Patient Agent ({patient_version}@{patient_model})"
    print(f"\n{model_label}开始并行评估")
    print(f"  模式: {mode_desc}")
    print(f"  最大线程数: {max_workers}，总样本数: {len(samples)}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in samples]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"{model_label}评估失败: {e}")
    
    print(f"\n{model_label}完成! 共评估 {progress['turns']} 轮对话")
    return all_results


def compute_statistics(
    all_results: Dict[str, SampleEvaluationResult],
) -> Dict[str, Any]:
    """计算统计信息"""
    # 收集各维度分数
    accuracy_scores = []
    honesty_scores = []
    realness_scores = {
        "Response_Brevity": [],
        "Information_Proactivity": [],
        "Emotional_Restraint": [],
        "Language_Polish": [],
    }
    
    total_turns = 0
    total_tokens = [0, 0]
    impressions = {"Real Patient": 0, "AI Agent": 0, "Unknown": 0}
    
    for patient_id, sample_result in all_results.items():
        for turn_result in sample_result.turn_results:
            total_turns += 1
            total_tokens[0] += turn_result.tokens[0]
            total_tokens[1] += turn_result.tokens[1]
            
            # 准确性/诚实性
            if turn_result.accuracy_score is not None:
                accuracy_scores.append(turn_result.accuracy_score)
            if turn_result.honesty_score is not None:
                honesty_scores.append(turn_result.honesty_score)
            
            # 真实性维度
            for metric_name in realness_scores:
                if metric_name in turn_result.realness_metrics:
                    score = turn_result.realness_metrics[metric_name].get("score")
                    if score is not None:
                        realness_scores[metric_name].append(score)
            
            # 整体印象
            impression = turn_result.realness_overall_impression
            if "Real" in impression:
                impressions["Real Patient"] += 1
            elif "AI" in impression:
                impressions["AI Agent"] += 1
            else:
                impressions["Unknown"] += 1
    
    # 计算统计
    stats = {
        "total_samples": len(all_results),
        "total_turns": total_turns,
        "accuracy": {
            "count": len(accuracy_scores),
            "average": calc_average(accuracy_scores),
            "std": calc_std(accuracy_scores),
        },
        "honesty": {
            "count": len(honesty_scores),
            "average": calc_average(honesty_scores),
            "std": calc_std(honesty_scores),
        },
        "realness_metrics": {},
        "impressions": impressions,
        "total_prompt_tokens": total_tokens[0],
        "total_completion_tokens": total_tokens[1],
    }
    
    for metric_name, scores in realness_scores.items():
        stats["realness_metrics"][metric_name] = {
            "count": len(scores),
            "average": calc_average(scores),
            "std": calc_std(scores),
        }
    
    return stats


def print_summary(stats: Dict[str, Any], model_name: str, data_file: str):
    """打印评估摘要"""
    print("\n" + "=" * 80)
    print("统一 Patient Agent 评估结果摘要")
    print("=" * 80)
    print(f"数据文件: {data_file}")
    print(f"评估模型: {model_name}")
    print(f"样本总数: {stats['total_samples']}")
    print(f"评估轮次: {stats['total_turns']}")
    print()
    
    print("【准确性与诚实性评估】(0-5分)")
    print("-" * 60)
    acc = stats["accuracy"]
    hon = stats["honesty"]
    print(f"  准确性: {acc['average']:.2f} ± {acc['std']:.2f} / 5 (样本数: {acc['count']})")
    print(f"  诚实性: {hon['average']:.2f} ± {hon['std']:.2f} / 5 (样本数: {hon['count']})")
    print()
    
    print("【真实性多维度评估】(1-5分，5分=极度像真人)")
    print("-" * 60)
    
    metric_names = {
        "Response_Brevity": "回复简洁度",
        "Information_Proactivity": "信息主动性",
        "Emotional_Restraint": "情感克制度",
        "Language_Polish": "语言修饰度",
    }
    
    for metric, display_name in metric_names.items():
        if metric in stats["realness_metrics"]:
            data = stats["realness_metrics"][metric]
            print(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f} / 5 (样本数: {data['count']})")
    
    print()
    print("【整体判断】")
    print("-" * 60)
    impressions = stats["impressions"]
    total = sum(impressions.values())
    if total > 0:
        print(f"  判断为真人: {impressions['Real Patient']} ({impressions['Real Patient']/total*100:.1f}%)")
        print(f"  判断为 AI: {impressions['AI Agent']} ({impressions['AI Agent']/total*100:.1f}%)")
    
    print()
    print("【Token 使用量】")
    print(f"  Prompt tokens: {stats['total_prompt_tokens']}")
    print(f"  Completion tokens: {stats['total_completion_tokens']}")
    print("=" * 80)


def result_to_dict(sample_result: SampleEvaluationResult) -> Dict[str, Any]:
    """将评估结果转换为字典"""
    return {
        "patient_id": sample_result.patient_id,
        "total_turns": sample_result.total_turns,
        "evaluated_turns": sample_result.evaluated_turns,
        "metadata": sample_result.metadata,
        "turns": [
            {
                "turn_index": r.turn_index,
                "doctor_question": r.doctor_question,
                "patient_reply": r.patient_reply,
                "information_exists": r.information_exists,
                "accuracy_score": r.accuracy_score,
                "accuracy_reason": r.accuracy_reason,
                "honesty_score": r.honesty_score,
                "honesty_reason": r.honesty_reason,
                "realness_metrics": r.realness_metrics,
                "realness_overall_impression": r.realness_overall_impression,
                "realness_average_score": r.realness_average_score,
                "tokens": r.tokens,
                "error": r.error,
            }
            for r in sample_result.turn_results
        ]
    }


def save_results(
    output_dir: Path,
    model_name: str,
    all_results: Dict[str, SampleEvaluationResult],
    stats: Dict[str, Any],
    data_file: Path,
    timestamp: str,
    use_original_reply: bool = True,
    patient_model: Optional[str] = None,
    patient_version: str = "v1",
):
    """保存评估结果"""
    model_safe_name = sanitize_model_name(model_name)
    
    # 根据模式生成不同的文件名前缀
    if use_original_reply:
        prefix = "patient_eval_original"
    else:
        patient_safe_name = sanitize_model_name(patient_model or "unknown")
        prefix = f"patient_eval_{patient_version}_{patient_safe_name}"
    
    # 保存详细结果
    output_file = output_dir / f"{prefix}_{data_file.stem}_{model_safe_name}_{timestamp}.json"
    output_data = {
        "metadata": {
            "data_file": str(data_file),
            "eval_model": model_name,
            "timestamp": timestamp,
            "use_original_reply": use_original_reply,
            "patient_model": patient_model,
            "patient_version": patient_version,
        },
        "statistics": stats,
        "results": [result_to_dict(r) for r in all_results.values()],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[{model_name}] 评估结果已保存到: {output_file}")
    
    return output_file


# ============================================================
# 聚合结果函数
# ============================================================

def load_result_file(file_path: str) -> Dict[str, Any]:
    """加载单个结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_single_model_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    从单个模型的结果中计算统计信息
    
    Args:
        results: 单个模型的评估结果字典
    
    Returns:
        统计信息字典
    """
    # 如果已经有统计信息，直接返回
    if "statistics" in results:
        return results["statistics"]
    
    # 否则从 results 中重新计算
    accuracy_scores = []
    honesty_scores = []
    realness_scores = {
        "Response_Brevity": [],
        "Information_Proactivity": [],
        "Emotional_Restraint": [],
        "Language_Polish": [],
    }
    
    impressions = {"Real Patient": 0, "AI Agent": 0, "Unknown": 0}
    total_turns = 0
    
    for sample_result in results.get("results", []):
        for turn in sample_result.get("turns", []):
            total_turns += 1
            
            # 准确性/诚实性
            if turn.get("accuracy_score") is not None:
                accuracy_scores.append(turn["accuracy_score"])
            if turn.get("honesty_score") is not None:
                honesty_scores.append(turn["honesty_score"])
            
            # 真实性维度
            realness_metrics = turn.get("realness_metrics", {})
            for metric_name in realness_scores:
                if metric_name in realness_metrics:
                    score = realness_metrics[metric_name].get("score")
                    if score is not None:
                        realness_scores[metric_name].append(score)
            
            # 整体印象
            impression = turn.get("realness_overall_impression", "Unknown")
            if "Real" in impression:
                impressions["Real Patient"] += 1
            elif "AI" in impression:
                impressions["AI Agent"] += 1
            else:
                impressions["Unknown"] += 1
    
    # 计算统计
    stats = {
        "total_samples": len(results.get("results", [])),
        "total_turns": total_turns,
        "accuracy": {
            "count": len(accuracy_scores),
            "average": calc_average(accuracy_scores),
            "std": calc_std(accuracy_scores),
        },
        "honesty": {
            "count": len(honesty_scores),
            "average": calc_average(honesty_scores),
            "std": calc_std(honesty_scores),
        },
        "realness_metrics": {},
        "impressions": impressions,
    }
    
    for metric_name, scores in realness_scores.items():
        stats["realness_metrics"][metric_name] = {
            "count": len(scores),
            "average": calc_average(scores),
            "std": calc_std(scores),
        }
    
    return stats


def aggregate_multi_model_results(
    model_results: Dict[str, Dict[str, Any]],
    fusion_method: str = "average",
) -> Dict[str, Any]:
    """
    聚合多个模型的评估结果
    
    Args:
        model_results: 模型名称 -> 评估结果的映射
        fusion_method: 融合方法 ("average" 或 "majority_vote")
    
    Returns:
        聚合后的统计信息
    """
    if not model_results:
        return {}
    
    # 收集所有模型的统计信息
    model_stats = {}
    for model_name, results in model_results.items():
        model_stats[model_name] = compute_single_model_stats(results)
    
    # 从第一个模型的结果中提取 patient 信息
    patient_model = None
    patient_version = None
    for results in model_results.values():
        metadata = results.get("metadata", {})
        patient_model = metadata.get("patient_model")
        patient_version = metadata.get("patient_version")
        if patient_model:
            break
    
    # 聚合各维度
    aggregated = {
        "models": list(model_results.keys()),
        "fusion_method": fusion_method,
        "patient_model": patient_model,
        "patient_version": patient_version,
        "per_model_stats": model_stats,
        "aggregated_stats": {},
    }
    
    # 计算聚合后的各维度分数
    metrics_to_aggregate = [
        ("accuracy", "average"),
        ("honesty", "average"),
    ]
    
    realness_metrics = [
        "Response_Brevity",
        "Information_Proactivity",
        "Emotional_Restraint",
        "Language_Polish",
    ]
    
    # 聚合准确性和诚实性
    for metric_key, sub_key in metrics_to_aggregate:
        values = []
        for stats in model_stats.values():
            if metric_key in stats and sub_key in stats[metric_key]:
                values.append(stats[metric_key][sub_key])
        
        aggregated["aggregated_stats"][metric_key] = {
            "average": calc_average(values),
            "std": calc_std(values),
            "per_model": {name: stats.get(metric_key, {}).get(sub_key) for name, stats in model_stats.items()},
        }
    
    # 聚合真实性维度
    aggregated["aggregated_stats"]["realness_metrics"] = {}
    for metric_name in realness_metrics:
        values = []
        per_model = {}
        for model_name, stats in model_stats.items():
            realness = stats.get("realness_metrics", {})
            if metric_name in realness:
                avg = realness[metric_name].get("average")
                if avg is not None:
                    values.append(avg)
                    per_model[model_name] = avg
        
        aggregated["aggregated_stats"]["realness_metrics"][metric_name] = {
            "average": calc_average(values),
            "std": calc_std(values),
            "per_model": per_model,
        }
    
    # 聚合整体印象
    total_real = 0
    total_ai = 0
    total_unknown = 0
    for stats in model_stats.values():
        impressions = stats.get("impressions", {})
        total_real += impressions.get("Real Patient", 0)
        total_ai += impressions.get("AI Agent", 0)
        total_unknown += impressions.get("Unknown", 0)
    
    total_impressions = total_real + total_ai + total_unknown
    aggregated["aggregated_stats"]["impressions"] = {
        "Real Patient": total_real,
        "AI Agent": total_ai,
        "Unknown": total_unknown,
        "real_rate": total_real / total_impressions if total_impressions > 0 else 0,
        "ai_rate": total_ai / total_impressions if total_impressions > 0 else 0,
    }
    
    return aggregated


def print_patient_aggregated_summary(aggregated: Dict[str, Any]):
    """打印 Patient Agent 聚合结果摘要"""
    print("\n" + "=" * 80)
    print("多模型融合 Patient Agent 评估结果摘要")
    print("=" * 80)
    print(f"评估模型: {', '.join(aggregated.get('models', []))}")
    print(f"融合方法: {aggregated.get('fusion_method', 'average')}")
    print()
    
    agg_stats = aggregated.get("aggregated_stats", {})
    
    print("【准确性与诚实性评估】(0-5分)")
    print("-" * 60)
    for metric in ["accuracy", "honesty"]:
        if metric in agg_stats:
            data = agg_stats[metric]
            print(f"  {metric}: {data['average']:.2f} ± {data['std']:.2f}")
            for model, score in data.get("per_model", {}).items():
                if score is not None:
                    print(f"    - {model}: {score:.2f}")
    print()
    
    print("【真实性多维度评估】(1-5分)")
    print("-" * 60)
    metric_names = {
        "Response_Brevity": "回复简洁度",
        "Information_Proactivity": "信息主动性",
        "Emotional_Restraint": "情感克制度",
        "Language_Polish": "语言修饰度",
    }
    
    realness_metrics = agg_stats.get("realness_metrics", {})
    for metric, display_name in metric_names.items():
        if metric in realness_metrics:
            data = realness_metrics[metric]
            print(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f}")
            for model, score in data.get("per_model", {}).items():
                if score is not None:
                    print(f"    - {model}: {score:.2f}")
    print()
    
    print("【整体判断】")
    print("-" * 60)
    impressions = agg_stats.get("impressions", {})
    print(f"  判断为真人: {impressions.get('Real Patient', 0)} ({impressions.get('real_rate', 0)*100:.1f}%)")
    print(f"  判断为 AI: {impressions.get('AI Agent', 0)} ({impressions.get('ai_rate', 0)*100:.1f}%)")
    print("=" * 80)


def save_patient_aggregated_results(aggregated: Dict[str, Any], output_file: Path):
    """保存 Patient Agent 聚合结果（JSON格式）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"聚合结果已保存到: {output_file}")


def save_patient_aggregated_summary_txt(aggregated: Dict[str, Any], output_file: Path):
    """保存 Patient Agent 聚合结果的文本摘要"""
    lines = []
    
    # 标题
    lines.append("=" * 70)
    fusion_method = aggregated.get('fusion_method', 'average')
    fusion_display = "Average Vote" if fusion_method == "average" else "Majority Vote"
    lines.append(f"多模型融合 ({fusion_display}) Patient Agent 评估结果")
    lines.append("=" * 70)
    
    # 模型信息
    models = aggregated.get('models', [])
    lines.append(f"评估模型: {', '.join(models)}")
    
    # Patient 模型信息
    patient_model = aggregated.get('patient_model', 'Unknown')
    patient_version = aggregated.get('patient_version', 'Unknown')
    lines.append(f"测试 Patient 模型: {patient_model}")
    lines.append(f"Patient 版本: {patient_version}")
    
    # 计算总轮次
    total_turns = 0
    for stats in aggregated.get("per_model_stats", {}).values():
        total_turns = max(total_turns, stats.get("total_turns", 0))
    lines.append(f"评估轮次: {total_turns}")
    lines.append("")
    
    agg_stats = aggregated.get("aggregated_stats", {})
    
    # 准确性与诚实性
    lines.append("【准确性与诚实性】(多模型平均)")
    lines.append("-" * 50)
    
    metric_display = {"accuracy": "准确性", "honesty": "诚实性"}
    for metric in ["accuracy", "honesty"]:
        if metric in agg_stats:
            data = agg_stats[metric]
            # 获取样本数量
            count = 0
            for stats in aggregated.get("per_model_stats", {}).values():
                if metric in stats:
                    count = stats[metric].get("count", 0)
                    break
            lines.append(f"  {metric_display[metric]}: {data['average']:.2f} ± {data['std']:.2f} / 5 (n={count})")
    lines.append("")
    
    # 真实性维度
    lines.append("【真实性维度】(多模型平均)")
    lines.append("-" * 50)
    
    metric_names = {
        "Response_Brevity": "回复简洁度",
        "Information_Proactivity": "信息主动性",
        "Emotional_Restraint": "情感克制度",
        "Language_Polish": "语言修饰度",
    }
    
    realness_metrics = agg_stats.get("realness_metrics", {})
    for metric, display_name in metric_names.items():
        if metric in realness_metrics:
            data = realness_metrics[metric]
            # 获取样本数量
            count = 0
            for stats in aggregated.get("per_model_stats", {}).values():
                if "realness_metrics" in stats and metric in stats["realness_metrics"]:
                    count = stats["realness_metrics"][metric].get("count", 0)
                    break
            lines.append(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f} (n={count})")
    lines.append("")
    
    # 整体判断
    lines.append("【整体判断】(投票聚合)")
    lines.append("-" * 50)
    impressions = agg_stats.get("impressions", {})
    real_count = impressions.get('Real Patient', 0)
    ai_count = impressions.get('AI Agent', 0)
    real_rate = impressions.get('real_rate', 0) * 100
    ai_rate = impressions.get('ai_rate', 0) * 100
    lines.append(f"  判断为真人: {real_count} ({real_rate:.1f}%)")
    lines.append(f"  判断为 AI: {ai_count} ({ai_rate:.1f}%)")
    lines.append("=" * 70)
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"聚合摘要已保存到: {output_file}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一的 Patient Agent 评估程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 模式1: 使用原始数据集中的患者回复（评估真实对话数据）
  python unified_patient_eval.py \\
      --data-file /path/to/data.json \\
      --use-original-reply \\
      --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041"

  # 模式2: 使用 Patient Agent 生成回复（评估 Patient Agent）
  python unified_patient_eval.py \\
      --data-file /path/to/data.json \\
      --patient-model "qwen3-30b@10.119.28.185:9041" \\
      --patient-version v1 \\
      --eval-models "gemma-3-27b-it@10.119.28.185:9051"
""",
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="数据文件路径（JSON格式）",
    )
    
    parser.add_argument(
        "--eval-models",
        type=str,
        default="gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041,gpt-oss-20b@10.119.28.185:9042",
        help="评估模型列表，逗号分隔",
    )
    
    parser.add_argument(
        "--patient-model",
        type=str,
        default=None,
        help="Patient Agent 使用的模型（当不使用 --use-original-reply 时必须指定）",
    )
    
    parser.add_argument(
        "--patient-version",
        type=str,
        choices=PATIENT_VERSION_CHOICES,
        default="v1",
        help="Patient Agent 版本（默认: v1，可选: v1, cot, mdd5k, v3）",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API Key（仅用于 OpenRouter 模型）",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="输出目录（默认: evaluation_results）",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评估的样本数量",
    )
    
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="采样间隔（默认: 5）",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行线程数（默认: 8）",
    )
    
    parser.add_argument(
        "--use-original-reply",
        action="store_true",
        help="使用数据集中原始的患者回复（不指定则使用 Patient Agent 生成）",
    )
    
    args = parser.parse_args()
    
    # 检查参数一致性
    use_original_reply = args.use_original_reply
    patient_model = args.patient_model
    patient_version = args.patient_version
    
    if not use_original_reply and not patient_model:
        print("错误: 当不使用 --use-original-reply 时，必须通过 --patient-model 指定 Patient Agent 模型")
        sys.exit(1)
    
    # 解析模型列表
    model_list = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    
    # 解析数据文件路径
    data_file = Path(args.data_file)
    if not data_file.is_absolute():
        data_file = PROJECT_ROOT / data_file
    
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        sys.exit(1)
    
    # 解析输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {data_file}")
    samples = load_data(str(data_file))
    print(f"加载了 {len(samples)} 个样本")
    
    # 限制样本数量（随机采样）
    if args.limit and args.limit < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, args.limit)
        print(f"随机采样 {args.limit} 个样本（seed=42）")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 打印配置信息
    print("\n" + "=" * 60)
    print("评估配置")
    print("=" * 60)
    print(f"数据文件: {data_file}")
    print(f"采样间隔: 每 {args.eval_interval} 轮采样一次")
    print(f"评估模型: {', '.join(model_list)}")
    if use_original_reply:
        print("回复来源: 使用数据集中的原始患者回复")
    else:
        print(f"回复来源: Patient Agent 生成")
        print(f"  - Patient 模型: {patient_model}")
        print(f"  - Patient 版本: {patient_version}")
    print("=" * 60)
    
    # ================================================================
    # 使用两阶段并行评估流程
    # ================================================================
    
    # 阶段1：准备评估任务（并行生成 Patient Agent 回复）
    eval_tasks = prepare_eval_tasks(
        samples=samples,
        eval_interval=args.eval_interval,
        use_original_reply=use_original_reply,
        patient_model=patient_model,
        patient_version=patient_version,
        max_workers=args.max_workers,
    )
    
    if not eval_tasks:
        print("错误: 没有可评估的任务")
        sys.exit(1)
    
    # 阶段2：并行评估所有任务（多模型并行）
    all_model_results = evaluate_tasks_parallel(
        eval_tasks=eval_tasks,
        eval_models=model_list,
        max_workers=args.max_workers,
        api_key=args.api_key,
    )
    
    # 计算统计并保存结果
    all_model_stats = {}
    
    for model_name in model_list:
        model_results = all_model_results.get(model_name, {})
        
        if not model_results:
            print(f"[警告] 模型 {model_name} 没有评估结果")
            continue
        
        # 计算统计信息
        stats = compute_statistics(model_results)
        all_model_stats[model_name] = stats
        
        # 打印摘要
        print_summary(stats, model_name, str(data_file))
        
        # 保存结果
        save_results(
            output_dir, model_name, model_results, stats, data_file, timestamp,
            use_original_reply=use_original_reply,
            patient_model=patient_model,
            patient_version=patient_version,
        )
    
    print(f"\n{'='*60}")
    print("所有模型评估完成!")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*60}")
    
    # 自动聚合多模型评估结果
    print(f"\n正在自动聚合多模型评估结果...")

    try:
        # 查找本次评估生成的结果文件（以 patient_ 开头）
        result_files = sorted(output_dir.glob(f"patient_*{timestamp}*.json"))
        if result_files:
            # 加载所有结果
            model_results = {}
            for file_path in result_files:
                try:
                    data = load_result_file(str(file_path))
                    model_name = data.get("metadata", {}).get("eval_model", file_path.stem)
                    model_results[model_name] = data
                except Exception as e:
                    print(f"警告: 无法加载 {file_path}: {e}")
            
            if model_results:
                # 聚合结果
                aggregated = aggregate_multi_model_results(model_results, "average")
                
                # 打印摘要
                print_patient_aggregated_summary(aggregated)
                
                # 保存 JSON 格式结果
                json_output = output_dir / f"patient_aggregated_results_{timestamp}.json"
                save_patient_aggregated_results(aggregated, json_output)
                
                # 保存文本格式摘要报告
                summary_output = output_dir / f"patient_aggregated_summary_{timestamp}.txt"
                save_patient_aggregated_summary_txt(aggregated, summary_output)
            else:
                print("警告: 没有成功加载任何结果文件，跳过聚合")
        else:
            print(f"警告: 未找到匹配的结果文件，跳过聚合")
    except Exception as e:
        print(f"警告: 自动聚合失败: {e}")


if __name__ == "__main__":
    main()


