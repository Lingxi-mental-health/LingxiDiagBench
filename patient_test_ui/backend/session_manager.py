"""Session management and orchestration for patient conversations."""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents import AutoDiagnosisVerifier, DoctorQuestionRecommender, PatientAnswerRecommender
from config import CONFIG
from interaction_log import SessionInteractionLog
from utils import safe_get_text


def extract_think_tags(text: str) -> Tuple[str, str]:
    """
    从文本中提取并移除 <think> 标签中的思考过程。
    
    Args:
        text: 包含可能的 <think> 标签的原始文本
        
    Returns:
        (cleaned_text, thinking_process): 
            - cleaned_text: 移除 <think> 标签及其内容后的文本
            - thinking_process: 提取的思考过程内容
    """
    if not text:
        return text, ""
    
    # 使用正则表达式匹配 <think>...</think> 标签（包括可能的空白字符）
    # 这个正则会匹配整个标签及其内容
    think_pattern = re.compile(
        r'<\s*think\s*>(.*?)<\s*/\s*think\s*>', 
        re.DOTALL | re.IGNORECASE
    )
    
    # 提取思考过程内容（只提取标签之间的内容，不包括标签本身）
    think_matches = think_pattern.findall(text)
    thinking_process = '\n'.join(match.strip() for match in think_matches) if think_matches else ""
    
    # 移除整个 <think>...</think> 标签及其内容
    cleaned_text = think_pattern.sub('', text).strip()
    
    # 调试输出
    if think_matches:
        print(f"[DEBUG] 检测到 <think> 标签，提取了 {len(think_matches)} 个思考过程")
        print(f"[DEBUG] 原始文本长度: {len(text)}, 清理后长度: {len(cleaned_text)}")
    
    return cleaned_text, thinking_process


class DoctorSession:
    """
    医生问诊会话管理类
    用于管理基于doctor_v2.py的医生问诊对话
    """
    def __init__(self, session_id: str, doctor_data: Dict[str, Any], user_name: str, doctor_version: str = "base"):
        self.session_id = session_id
        self.doctor_data = doctor_data
        self.user_name = user_name
        self.doctor_version = doctor_version  # 添加版本参数
        self.conversation_log = []
        self.dialogue_history = []
        self.events = []
        self.created_at = time.time()
        self.last_activity = time.time()
        
        # 初始化对话保存
        self._initialize_conversation_log()
        
        # 初始化医生AI
        self.doctor_agent = None
        self._initialize_doctor()
        
        # 初始化患者回答推荐器
        self._patient_answer_recommender: Optional[PatientAnswerRecommender] = None
        self._patient_answer_recommender_disabled = False
    
    def _initialize_conversation_log(self):
        """初始化对话记录保存"""
        import os
        
        # 创建保存目录
        self.diagnoses_dir = os.path.join(os.path.dirname(__file__), '../diagnoses')
        os.makedirs(self.diagnoses_dir, exist_ok=True)
        
        # 生成文件名
        self.log_filename = f"doctor_interactions_{self.session_id}_{int(self.created_at)}.json"
        self.log_filepath = os.path.join(self.diagnoses_dir, self.log_filename)
        
        # 初始化日志结构
        self.log_data = {
            "session_id": self.session_id,
            "doctor_id": self.doctor_data.get('doctor_id'),
            "doctor_name": self.doctor_data.get('name'),
            "user": self.user_name,
            "created_at": self.created_at,
            "events": []
        }
        
        # 记录会话创建事件
        self.record_event("session_created", {
            "doctor_info": {
                "doctor_id": self.doctor_data.get('doctor_id'),
                "name": self.doctor_data.get('name'),
                "age": self.doctor_data.get('age'),
                "gender": self.doctor_data.get('gender'),
                "special": self.doctor_data.get('special'),
                "commu": self.doctor_data.get('commu'),
                "empathy": self.doctor_data.get('empathy')
            },
            "user": self.user_name
        })
        
        # 保存初始状态
        self._save_conversation_log()
    
    def _save_conversation_log(self):
        """保存对话记录到JSON文件"""
        try:
            import json
            with open(self.log_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话记录失败: {e}")
    
    def _initialize_doctor(self):
        """初始化医生AI，根据版本动态导入"""
        try:
            import sys
            import os
            
            # 添加项目根目录到Python路径
            project_root = os.path.join(os.path.dirname(__file__), '../..')
            sys.path.insert(0, project_root)
            
            # 根据版本导入不同的Doctor类
            if self.doctor_version == 'v1':
                from src.doctor.doctor_v1 import Doctor
                from src.doctor.diagtree_v1 import DiagTree
                print(f"[DoctorSession] 使用Doctor V1版本：传统诊断树")
            elif self.doctor_version == 'v2':
                from src.doctor.doctor_v2 import Doctor
                from src.doctor.diagtree_v2 import DiagTree
                print(f"[DoctorSession] 使用Doctor V2版本：阶段式诊断树")
            else:  # base
                from src.doctor.doctor_base import DoctorBase as Doctor
                DiagTree = None
                print(f"[DoctorSession] 使用Doctor基础版本：无诊断树问诊")
            
            from config import CONFIG
            
            # 创建虚拟患者模板（用于医生初始化）
            patient_template = {
                'Age': 25,
                'Gender': '女',
                'patient_id': 'virtual_patient'
            }
            
            # 医生配置路径
            doctor_prompt_path = os.path.join(project_root, 'prompts/doctor/doctor_persona.json')
            diagtree_path = os.path.join(project_root, 'prompts/diagtree/female_adult.json')  # 占位符
            
            # 从配置中获取模型设置
            doctor_model_config = CONFIG["models"]["doctor"]
            
            if doctor_model_config["use_openrouter"]:
                # 使用OpenRouter API
                model_path = doctor_model_config["openrouter_model"]
                use_api = True
            else:
                # 使用本地VLLM模型
                # 检查 local_model_name 是否已经包含 @host:port 格式
                local_model_name = doctor_model_config['local_model_name']
                if '@' in local_model_name and ':' in local_model_name:
                    # 已经是 model@host:port 格式，直接使用
                    model_path = local_model_name
                else:
                    # 需要添加端口号
                    model_path = f"{local_model_name}:{doctor_model_config['local_model_port']}"
                use_api = True  # doctor_v2.py中的API模式也支持本地VLLM
            
            print(f"[DoctorSession] 初始化医生代理，版本: {self.doctor_version}, 模型: {model_path}, API模式: {use_api}")
            
            # 初始化医生
            self.doctor_agent = Doctor(
                patient_template=patient_template,
                doctor_prompt_path=doctor_prompt_path,
                diagtree_path=diagtree_path,
                model_path=model_path,
                use_api=use_api
            )
            
            return True
            
        except Exception as e:
            print(f"初始化医生AI失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_doctor_question(self) -> Dict[str, Any]:
        """生成医生问题"""
        try:
            if not self.doctor_agent:
                raise ValueError("医生AI未初始化")
            
            # 如果是第一次对话，生成初始问题
            if not self.conversation_log:
                # 使用医生AI生成第一个问题
                response = self.doctor_agent.doctor_response_gen("", self.dialogue_history)
                if isinstance(response, tuple):
                    question_raw = response[0]
                else:
                    question_raw = response
            else:
                # 基于对话历史生成后续问题
                last_patient_reply = ""
                if self.conversation_log:
                    for msg in reversed(self.conversation_log):
                        if msg.get("role") == "patient":
                            last_patient_reply = msg.get("content", "")
                            break
                
                response = self.doctor_agent.doctor_response_gen(last_patient_reply, self.dialogue_history)
                if isinstance(response, tuple):
                    question_raw = response[0]
                    # 检查是否是诊断结果
                    if response[1] is None and "诊断结束" in question_raw:
                        # 这是最终诊断结果，也需要清理 <think> 标签
                        question, _ = extract_think_tags(question_raw)
                        self.record_event("diagnosis_completed", {"diagnosis": question})
                        return {"question": question, "is_diagnosis": True}
                else:
                    question_raw = response
            
            # 提取并移除 <think> 标签
            question, _ = extract_think_tags(question_raw)
            
            # 记录医生问题
            timestamp = time.time()
            self.record_event("doctor_question_generated", {
                "question": question, 
                "timestamp": timestamp,
                "doctor_name": self.doctor_data.get('name', '未知医生'),
                "conversation_turn": len([msg for msg in self.conversation_log if msg.get("role") == "doctor"]) + 1
            })
            self.dialogue_history.append(f"医生: {question}")
            self.conversation_log.append({
                "role": "doctor",
                "content": question,
                "timestamp": timestamp
            })
            
            return {"question": question, "is_diagnosis": False}
            
        except Exception as e:
            print(f"生成医生问题失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def record_patient_reply(self, reply: str) -> Dict[str, Any]:
        """记录患者回复"""
        try:
            timestamp = time.time()
            self.record_event("patient_reply", {
                "reply": reply, 
                "timestamp": timestamp,
                "conversation_turn": len([msg for msg in self.conversation_log if msg.get("role") == "patient"]) + 1,
                "total_exchanges": len(self.conversation_log) // 2 + 1
            })
            self.dialogue_history.append(f"患者: {reply}")
            self.conversation_log.append({
                "role": "patient", 
                "content": reply,
                "timestamp": timestamp
            })
            
            self.last_activity = timestamp
            
            return {"success": True}
            
        except Exception as e:
            print(f"记录患者回复失败: {e}")
            raise
    
    def record_event(self, event_type: str, data: Dict[str, Any]):
        """记录事件并实时保存"""
        timestamp = time.time()
        
        # 添加到内存中的事件列表
        event = {
            "type": event_type,
            "data": data,
            "timestamp": timestamp
        }
        self.events.append(event)
        
        # 添加到日志数据中
        log_event = {
            "event_type": event_type,
            "payload": {
                **data,
                "timestamp": timestamp,
                "user": self.user_name
            },
            "timestamp": timestamp
        }
        self.log_data["events"].append(log_event)
        
        # 实时保存到文件
        self._save_conversation_log()
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            "session_id": self.session_id,
            "session_type": "doctor",
            "doctor_info": self.doctor_data,
            "user_name": self.user_name,
            "conversation_count": len([msg for msg in self.conversation_log if msg.get("role") == "doctor"]),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "conversation_log": self.conversation_log,
            "log_file": self.log_filename,
            "log_path": self.log_filepath
        }
    
    def _get_patient_answer_recommender(self) -> Optional[PatientAnswerRecommender]:
        """获取患者回答推荐器实例"""
        if self._patient_answer_recommender_disabled:
            return None
        
        if self._patient_answer_recommender is None:
            try:
                model_config = CONFIG["models"]["patient"]  # 使用患者模型配置
                openrouter_config = CONFIG["openrouter"]
                self._patient_answer_recommender = PatientAnswerRecommender(
                    model_config, openrouter_config, max_answers=3
                )
            except Exception as e:
                print(f"初始化患者回答推荐器失败: {e}")
                self._patient_answer_recommender_disabled = True
                return None
        
        return self._patient_answer_recommender
    
    def _generate_patient_answer_suggestions(self) -> List[str]:
        """生成患者回答建议"""
        recommender = self._get_patient_answer_recommender()
        if not recommender:
            return []
        
        # 构造患者信息（基于医生信息创建虚拟患者档案）
        patient_info = {
            "age": 30,  # 默认年龄
            "gender": "未知",
            "department": "精神科",
            "chief_complaint": "情绪问题",
            "symptoms": "情绪低落、焦虑、睡眠问题",
            "background": "工作压力大，人际关系紧张"
        }
        
        suggestions = recommender.suggest_answers(self.conversation_log, patient_info)
        return suggestions or []
    
    def recommend_patient_answers(self, limit: int = 3) -> List[str]:
        """推荐患者回答"""
        suggestions = self._generate_patient_answer_suggestions()
        limit = max(1, limit)
        selected = suggestions[:limit] if suggestions else []
        self.record_event(
            "patient_answer_suggestions_generated",
            {
                "requested": limit,
                "available": len(suggestions),
                "answers": selected,
            },
        )
        return selected
    
    def run_auto_diagnosis(self) -> Dict[str, Any]:
        """Generate auto diagnosis for doctor session."""
        # 检查是否有有效的对话记录
        if not any(log.get("role") == "patient" for log in self.conversation_log):
            raise ValueError("需要至少一轮有效的问诊记录后才能自动诊断")
        
        # 获取诊断代理，使用与EverDiagnosis相同的配置
        try:
            verifier = AutoDiagnosisVerifier(
                CONFIG["models"]["verifier"],
                CONFIG["openrouter"],
            )
        except Exception as exc:
            print(f"初始化诊断代理失败: {exc}")
            traceback.print_exc()
            raise RuntimeError("自动诊断模型初始化失败") from exc
        
        try:
            result = verifier.generate_diagnosis(self.conversation_log)
        except Exception as exc:
            print(f"生成诊断失败: {exc}")
            traceback.print_exc()
            raise RuntimeError("自动诊断生成失败") from exc
        
        result["generated_at"] = time.time()
        self.record_event("auto_diagnosis_generated", result)
        return result


class PatientSession:
    """Encapsulate a single doctor-patient conversation session."""

    def __init__(self, patient_template: Dict[str, Any], session_id: str, user_name: Optional[str] = None, patient_version: str = "v1"):
        self.session_id = session_id
        self.patient_template = patient_template
        self.patient_version = patient_version  # 添加版本参数
        self.patient_agent = None  # 不再类型限定为Patient，允许动态类型
        self.dialogue_history: List[str] = []
        self.conversation_log: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.auto_diagnosis_cache: Optional[Dict[str, Any]] = None
        self.user_name = user_name
        self._interactions_dir = Path(CONFIG["project_root"]) / "patient_test_ui" / "diagnoses"
        self.interaction_log = SessionInteractionLog(
            session_id=session_id,
            patient_id=patient_template.get("患者", patient_template.get("patient_id", "unknown")),
            user=user_name,
        )

        self._doctor_recommender: Optional[DoctorQuestionRecommender] = None
        self._doctor_recommender_disabled = False
        self._verifier_agent: Optional[AutoDiagnosisVerifier] = None
        self._verifier_disabled = False

    # ------------------------------------------------------------------ #
    # Patient agent lifecycle
    # ------------------------------------------------------------------ #
    def initialize_patient(self) -> bool:
        """Initialise the patient agent according to configuration and version."""
        patient_model_cfg = CONFIG["models"]["patient"]
        openrouter_cfg = CONFIG["openrouter"]

        try:
            # 根据版本导入不同的Patient类
            if self.patient_version == 'cot':
                from src.patient.patient_cot import Patient
                print(f"[PatientSession] 使用Patient CoT版本：Chain of Thought")
            else:  # v1
                from src.patient.patient_v1 import Patient
                print(f"[PatientSession] 使用Patient V1版本：基础版本")
            
            if patient_model_cfg.get("use_openrouter"):
                # Patient类会自动检测OpenRouter模型并从环境变量读取配置
                # 确保环境变量已设置: OPENROUTER_SITE_URL, OPENROUTER_SITE_NAME
                import os
                if openrouter_cfg.get("api_key"):
                    os.environ["OPENROUTER_API_KEY"] = openrouter_cfg.get("api_key", "")
                if openrouter_cfg.get("site_url"):
                    os.environ["OPENROUTER_SITE_URL"] = openrouter_cfg.get("site_url", "")
                if openrouter_cfg.get("site_name"):
                    os.environ["OPENROUTER_SITE_NAME"] = openrouter_cfg.get("site_name", "")
                
                # 根据版本传入不同的参数
                if self.patient_version == 'cot':
                    # COT版本支持enable_chief_complaint参数
                    self.patient_agent = Patient(
                        patient_template=self.patient_template,
                        model_path=patient_model_cfg.get("openrouter_model"),
                        use_api=True,
                        enable_chief_complaint=True,
                    )
                else:
                    # V1版本不支持enable_chief_complaint参数
                    self.patient_agent = Patient(
                        patient_template=self.patient_template,
                        model_path=patient_model_cfg.get("openrouter_model"),
                        use_api=True,
                    )
            else:
                # 检查 local_model_name 是否已经包含 @host:port 格式
                # 如果已经包含，就不要再添加端口
                local_model_name = patient_model_cfg.get('local_model_name')
                if '@' in local_model_name and ':' in local_model_name:
                    # 已经是 model@host:port 格式，直接使用
                    model_with_port = local_model_name
                else:
                    # 需要添加端口号
                    model_with_port = (
                        f"{local_model_name}:"  # type: ignore[str-bytes-safe]
                        f"{patient_model_cfg.get('local_model_port')}"
                    )
                # 根据版本传入不同的参数
                if self.patient_version == 'cot':
                    # COT版本支持enable_chief_complaint参数
                    self.patient_agent = Patient(
                        patient_template=self.patient_template,
                        model_path=model_with_port,
                        use_api=True,
                        enable_chief_complaint=True,
                    )
                else:
                    # V1版本不支持enable_chief_complaint参数
                    self.patient_agent = Patient(
                        patient_template=self.patient_template,
                        model_path=model_with_port,
                        use_api=True,
                    )
            
            print(f"[PatientSession] 初始化患者代理成功，版本: {self.patient_version}")
            return True
        except Exception as exc:  # pragma: no cover - defensive path
            print(f"初始化Patient Agent失败: {exc}")
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------ #
    # Helper accessors
    # ------------------------------------------------------------------ #
    def _get_doctor_recommender(self) -> Optional[DoctorQuestionRecommender]:
        if self._doctor_recommender_disabled:
            return None
        if self._doctor_recommender is None:
            try:
                self._doctor_recommender = DoctorQuestionRecommender(
                    CONFIG["models"]["doctor"],
                    CONFIG["openrouter"],
                )
            except Exception as exc:
                print(f"Doctor recommender 初始化失败: {exc}")
                traceback.print_exc()
                self._doctor_recommender_disabled = True
                return None
        return self._doctor_recommender

    def _get_verifier_agent(self) -> Optional[AutoDiagnosisVerifier]:
        if self._verifier_disabled:
            raise RuntimeError("Verifier agent unavailable")
        if self._verifier_agent is None:
            try:
                self._verifier_agent = AutoDiagnosisVerifier(
                    CONFIG["models"]["verifier"],
                    CONFIG["openrouter"],
                )
            except Exception as exc:
                print(f"Verifier agent 初始化失败: {exc}")
                traceback.print_exc()
                self._verifier_disabled = True
                raise
        return self._verifier_agent

    # ------------------------------------------------------------------ #
    # Dialogue handling
    # ------------------------------------------------------------------ #
    def record_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Record interaction event and persist to disk."""
        payload_with_user = dict(payload)
        if "user" not in payload_with_user and self.user_name:
            payload_with_user["user"] = self.user_name
        self.interaction_log.add_event(event_type, payload_with_user)
        self.interaction_log.save(self._interactions_dir)

    # ------------------------------------------------------------------ #
    # Doctor-led interview mode (patient role-play)
    # ------------------------------------------------------------------ #
    def can_generate_doctor_question(self) -> bool:
        doctor_questions = len([log for log in self.conversation_log if log.get("role") == "doctor"])
        patient_replies = len([log for log in self.conversation_log if log.get("role") == "patient"])
        return doctor_questions == patient_replies

    def _build_initial_doctor_question(self) -> str:
        patient_info = self.get_session_info()["patient_info"]
        chief_complaint = patient_info.get("chief_complaint")
        if chief_complaint:
            return (
                f"您好，我是精神科的主治医生。能和我详细聊聊关于“{chief_complaint}”的情况吗？"
            )
        return "您好，我是精神科的主治医生。请告诉我最近最困扰你的症状或感受。"

    def generate_doctor_question(self) -> Dict[str, Any]:
        """Generate a doctor question for patient role-play mode."""
        if not self.can_generate_doctor_question():
            raise ValueError("请先完成上一轮的回答，再生成新的医生问题。")

        suggestions: List[str] = []
        if self.conversation_log:
            suggestions = self._generate_doctor_suggestions()

        if not self.conversation_log:
            question = self._build_initial_doctor_question()
        elif suggestions:
            question = suggestions[0]
        else:
            question = "谢谢你的回答。我还想了解一下最近有没有其他值得关注的症状或经历？"

        timestamp = time.time()
        self.record_event(
            "doctor_question_generated",
            {"question": question, "timestamp": timestamp},
        )
        self.dialogue_history.append(f"医生: {question}")
        self.conversation_log.append(
            {
                "role": "doctor",
                "content": question,
                "timestamp": timestamp,
            }
        )

        return {"question": question}

    def record_patient_reply_manual(self, reply_text: str) -> Dict[str, Any]:
        """Record a patient reply provided manually by the user."""
        reply = reply_text.strip()
        if not reply:
            raise ValueError("患者回答不能为空")

        doctor_questions = len([log for log in self.conversation_log if log.get("role") == "doctor"])
        patient_replies = len([log for log in self.conversation_log if log.get("role") == "patient"])
        if doctor_questions == patient_replies:
            raise ValueError("请先生成医生问题后再回答。")

        timestamp = time.time()
        self.dialogue_history.append(f"患者本人: {reply}")
        self.conversation_log.append(
            {
                "role": "patient",
                "content": reply,
                "timestamp": timestamp,
            }
        )
        self.record_event(
            "patient_manual_reply",
            {
                "response": reply,
                "timestamp": timestamp,
            },
        )

        self.auto_diagnosis_cache = None
        return {"session_info": self.get_session_info()}

    def patient_response(self, doctor_question: str) -> Dict[str, Any]:
        """Process doctor's question and return patient's reply."""
        if not self.patient_agent:
            return {
                "response": "抱歉，出现了技术问题，请重新开始对话。",
                "classification": None,
                "suggested_questions": [],
                "error": "Patient agent not initialized",
            }

        try:
            timestamp = time.time()
            self.record_event("doctor_message", {"message": doctor_question, "timestamp": timestamp})
            self.dialogue_history.append(f"医生: {doctor_question}")
            self.conversation_log.append(
                {
                    "role": "doctor",
                    "content": doctor_question,
                    "timestamp": timestamp,
                }
            )

            patient_response_raw, _, classification_info, patient_reasoning = self.patient_agent.patient_response_gen(
                current_topic="患者的精神状况",
                dialogue_history=self.dialogue_history,
                current_doctor_question=doctor_question,
            )

            # 提取并移除 <think> 标签中的思考过程
            patient_response, thinking_process = extract_think_tags(patient_response_raw)
            
            # 如果有思考过程，将其合并到 reasoning 中
            if thinking_process:
                if patient_reasoning:
                    patient_reasoning = f"{thinking_process}\n\n{patient_reasoning}"
                else:
                    patient_reasoning = thinking_process

            self.dialogue_history.append(f"患者本人: {patient_response}")
            self.conversation_log.append(
                {
                    "role": "patient",
                    "content": patient_response,
                    "timestamp": time.time(),
                    "classification": classification_info,
                    "reasoning": patient_reasoning,
                }
            )

            self.record_event(
                "patient_reply",
                {
                    "response": patient_response,
                    "classification": classification_info,
                    "reasoning": patient_reasoning,
                },
            )

            # 任何新的对话轮次都会使缓存的自动诊断失效
            self.auto_diagnosis_cache = None

            return {
                "response": patient_response,
                "classification": classification_info,
                "suggested_questions": [],
                "error": None,
            }

        except Exception as exc:  # pragma: no cover - surface error to caller
            print(f"处理患者回复失败: {exc}")
            traceback.print_exc()
            return {
                "response": "抱歉，我现在有些不舒服，无法回答您的问题。",
                "classification": None,
                "suggested_questions": [],
                "error": str(exc),
            }

    def _generate_doctor_suggestions(self) -> List[str]:
        recommender = self._get_doctor_recommender()
        if not recommender:
            return []
        patient_info = self.get_session_info()["patient_info"]
        suggestions = recommender.suggest_questions(self.conversation_log, patient_info)
        return suggestions or []

    def recommend_questions(self, limit: int = 1) -> List[str]:
        suggestions = self._generate_doctor_suggestions()
        limit = max(1, limit)
        selected = suggestions[:limit] if suggestions else []
        self.record_event(
            "suggestions_generated",
            {
                "requested": limit,
                "available": len(suggestions),
                "questions": selected,
            },
        )
        return selected

    # ------------------------------------------------------------------ #
    # Auto diagnosis
    # ------------------------------------------------------------------ #
    def run_auto_diagnosis(self) -> Dict[str, Any]:
        """Generate auto diagnosis once per session."""
        if self.auto_diagnosis_cache:
            return self.auto_diagnosis_cache

        if not any(log.get("role") == "patient" for log in self.conversation_log):
            raise ValueError("需要至少一轮有效的问诊记录后才能自动诊断")

        verifier = self._get_verifier_agent()

        try:
            result = verifier.generate_diagnosis(self.conversation_log)
        except Exception:
            self._verifier_disabled = True
            raise

        result["generated_at"] = time.time()
        self.auto_diagnosis_cache = result
        self.record_event("auto_diagnosis_generated", result)
        return result

    # ------------------------------------------------------------------ #
    # Suggestion selections
    # ------------------------------------------------------------------ #
    def record_suggestion_selection(self, question: str) -> None:
        self.record_event("suggestion_selected", {"question": question})

    # ------------------------------------------------------------------ #
    # Session metadata
    # ------------------------------------------------------------------ #
    def get_session_info(self) -> Dict[str, Any]:
        patient_info = {
            "age": self.patient_template.get("Age", self.patient_template.get("年龄", "未知")),
            "gender": safe_get_text(self.patient_template, "Gender", "性别", "未知"),
            "department": safe_get_text(self.patient_template, "Department", "科室", "精神科"),
            "chief_complaint": safe_get_text(
                self.patient_template, "ChiefComplaint", "主诉"
            ),
        }

        return {
            "session_id": self.session_id,
            "patient_id": self.patient_template.get("患者", self.patient_template.get("patient_id", "unknown")),
            "patient_info": patient_info,
            "user_name": self.user_name,
            "conversation_count": len(
                [log for log in self.conversation_log if log.get("role") == "patient"]
            ),
            "duration": time.time() - self.start_time,
            "conversation_log": self.conversation_log,
        }
