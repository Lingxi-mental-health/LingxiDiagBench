"""LLM-powered helper agents for the Patient Test UI backend."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

import src.llm.llm_tools_api as llm_tools_api

LOGGER = logging.getLogger(__name__)


def _format_dialogue(dialogue: Sequence[Dict[str, Any]], limit: int = 12) -> str:
    """Format the most recent dialogue turns into a compact text block."""
    if not dialogue:
        return ""

    recent = dialogue[-limit:]
    lines: List[str] = []
    role_map = {
        "doctor": "医生",
        "patient": "患者",
        "system": "系统",
        "unknown": "未知发言人",
        "family": "家属",
        "others": "未知发言人"
    }

    for entry in recent:
        role = role_map.get(entry.get("role", ""), entry.get("role", ""))
        content = (entry.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract the content inside a custom XML-like tag."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _coerce_message_text(message: Any) -> str:
    """
    Normalize OpenAI-style message content to plain text.

    Handles both legacy string responses and the new array-of-blocks format:
    [
      {"type": "output_text", "text": "..."},
      ...
    ]
    """
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(message, dict):
        content = message.get("content", content)

    if isinstance(content, str):
        return content

    texts: List[str] = []
    if isinstance(content, (list, tuple)):
        for part in content:
            if part is None:
                continue

            part_type = getattr(part, "type", None)
            part_text = getattr(part, "text", None)

            if isinstance(part, dict):
                part_type = part.get("type", part_type)
                part_text = part.get("text", part_text) or part.get("content")

            if isinstance(part_text, str) and part_text.strip():
                texts.append(part_text)
            elif part_type in {"output_text", "text"}:
                literal = getattr(part, "content", None)
                if isinstance(part, dict):
                    literal = part.get("content", literal)
                if isinstance(literal, str) and literal.strip():
                    texts.append(literal)

    return "\n".join(t.strip() for t in texts if isinstance(t, str) and t.strip())


class LLMChatAgent:
    """Shared helper for OpenAI-compatible chat completions."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        openrouter_config: Dict[str, Any],
        agent_name: str,
    ) -> None:
        self.model_config = model_config
        self.openrouter_config = openrouter_config
        self.agent_name = agent_name
        self.client: Optional[OpenAI] = None

    @property
    def model_identifier(self) -> str:
        """Return the model identifier used in API requests."""
        if self.model_config.get("use_openrouter"):
            return self.model_config.get("openrouter_model", "")
        
        ## 处理本地模型名称，去掉@host:port部分
        if '@' in self.model_config.get("local_model_name", ""):
            return "@".join(self.model_config.get("local_model_name", "").split('@')[:-1])
        elif ':' in self.model_config.get("local_model_name", ""):
            return ":".join(self.model_config.get("local_model_name", "").split(':')[:-1])
        else:
            return self.model_config.get("local_model_name", "")
        
    def _ensure_client(self) -> None:
        """Lazily create the OpenAI-compatible client."""
        if self.client is not None:
            return

        # 使用统一的客户端初始化函数
        try:
            self.client = llm_tools_api.create_client_for_diagnosis(
                model_name=None,
                model_config=self.model_config,
                openrouter_config=self.openrouter_config,
                use_openrouter=self.model_config.get("use_openrouter", False)
            )
        except Exception as e:
            LOGGER.error(f"Failed to create client for {self.agent_name} agent: {e}")
            raise RuntimeError(
                f"Failed to create client for {self.agent_name} agent: {e}"
            )

    def _chat(self, messages: List[Dict[str, Any]], **kwargs: Any):
        """Call the chat completion endpoint with shared defaults."""
        self._ensure_client()
        params = {
            "model": self.model_identifier,
            "messages": messages,
        }
        print("模型请求msg参数: ", params)
        params.update(kwargs)

        try:
            if self.model_config.get("use_openrouter"):
                extra_headers = {}
                site_url = self.openrouter_config.get("site_url")
                site_name = self.openrouter_config.get("site_name")
                if site_url:
                    extra_headers["HTTP-Referer"] = site_url
                if site_name:
                    extra_headers["X-Title"] = site_name

                return self.client.chat.completions.create(
                    extra_headers=extra_headers,
                    extra_body={},
                    **params,
                )
            return self.client.chat.completions.create(**params)
        except Exception as exc:  # pragma: no cover - surface upstream
            LOGGER.exception("LLM call failed for agent %s: %s", self.agent_name, exc)
            raise


class DoctorQuestionRecommender(LLMChatAgent):
    """Recommend follow-up doctor questions based on recent dialogue."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        openrouter_config: Dict[str, Any],
        max_questions: int = 3,
    ) -> None:
        super().__init__(model_config, openrouter_config, agent_name="doctor_recommender")
        self.max_questions = max_questions

    def suggest_questions(
        self,
        conversation_log: Sequence[Dict[str, Any]],
        patient_profile: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[str]:
        """Generate recommended follow-up questions."""
        if not conversation_log:
            return []

        dialogue_text = _format_dialogue(conversation_log)
        if not dialogue_text:
            return []

        patient_summary = (
            f"{patient_profile.get('age', '未知')}岁"
            f"{patient_profile.get('gender', '未知')}性，科室："
            f"{patient_profile.get('department', '精神科')}，主诉："
            f"{patient_profile.get('chief_complaint', '未提供')}"
        )

        instructions = (
            f"以下是精神科初诊的问诊记录摘要：\n{dialogue_text}\n\n"
            f"患者概况：{patient_summary}\n\n"
            f"请站在资深精神科医生的角度，为下一轮问诊准备不超过"
            f"{self.max_questions}个可选问题。问题需要紧扣患者最近的回答，避免重复。"
            "请直接输出问题列表，每行一个问题，以“- ”作为前缀。"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名经验丰富的精神科主治医生，擅长根据患者的最新回答"
                    "提出有针对性的追问，语言要简洁自然。"
                ),
            },
            {"role": "user", "content": instructions},
        ]

        try:
            completion = self._chat(
                messages,
                temperature=0.1,
                top_p=0.9,
                max_tokens=512,
            )
        except Exception:
            return []

        text = completion.choices[0].message.content or ""
        questions = self._parse_questions(text)
        max_items = limit if limit is not None and limit > 0 else self.max_questions
        return questions[:max_items]

    @staticmethod
    def _parse_questions(raw_text: str) -> List[str]:
        """Extract question list from model output."""
        raw_text = raw_text.strip()
        if not raw_text:
            return []

        # First try JSON parsing to support structured outputs.
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict) and "questions" in data:
                items = data["questions"]
            elif isinstance(data, list):
                items = data
            else:
                items = []
            return [str(item).strip() for item in items if str(item).strip()]
        except json.JSONDecodeError:
            pass

        questions: List[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            elif stripped[0].isdigit() and "." in stripped[:3]:
                stripped = stripped.split(".", 1)[-1].strip()
            if stripped:
                questions.append(stripped)
        return questions


class PatientAnswerRecommender(LLMChatAgent):
    """Recommend patient answers based on doctor's questions."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        openrouter_config: Dict[str, Any],
        max_answers: int = 3,
    ) -> None:
        super().__init__(model_config, openrouter_config, agent_name="patient_answer_recommender")
        self.max_answers = max_answers

    def suggest_answers(
        self,
        conversation_log: Sequence[Dict[str, Any]],
        patient_profile: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> List[str]:
        """Generate recommended patient answers."""
        if not conversation_log:
            return []

        # 获取最新的医生问题
        latest_doctor_question = ""
        for msg in reversed(conversation_log):
            if msg.get("role") == "doctor":
                latest_doctor_question = msg.get("content", "")
                break
        
        if not latest_doctor_question:
            return []

        dialogue_text = _format_dialogue(conversation_log)
        
        patient_summary = (
            f"{patient_profile.get('age', '未知')}岁"
            f"{patient_profile.get('gender', '未知')}性，科室："
            f"{patient_profile.get('department', '精神科')}，主诉："
            f"{patient_profile.get('chief_complaint', '未提供')}"
        )

        # 获取患者的症状和背景信息
        symptoms = patient_profile.get('symptoms', '')
        background = patient_profile.get('background', '')
        
        instructions = (
            f"以下是精神科问诊记录摘要：\n{dialogue_text}\n\n"
            f"患者概况：{patient_summary}\n"
            f"患者症状：{symptoms}\n"
            f"患者背景：{background}\n\n"
            f"医生最新问题：{latest_doctor_question}\n\n"
            f"请站在患者的角度，根据患者的症状和背景，为医生的问题提供不超过"
            f"{self.max_answers}个可能的回答选项。回答应该符合患者的病情特点，"
            "语言要自然真实，体现患者的心理状态。"
            "请直接输出回答列表，每行一个回答，以"- "作为前缀。"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一名精神科患者，需要根据自己的症状和背景如实回答医生的问题。"
                    "回答要真实自然，体现患者的心理状态和病情特点。"
                ),
            },
            {"role": "user", "content": instructions},
        ]

        try:
            completion = self._chat(
                messages,
                temperature=0.1,
                top_p=0.9,
                max_tokens=512,
            )
        except Exception:
            return []

        text = completion.choices[0].message.content or ""
        answers = self._parse_answers(text)
        max_items = limit if limit is not None and limit > 0 else self.max_answers
        return answers[:max_items]

    @staticmethod
    def _parse_answers(raw_text: str) -> List[str]:
        """Extract answer list from model output."""
        raw_text = raw_text.strip()
        if not raw_text:
            return []

        # First try JSON parsing to support structured outputs.
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict) and "answers" in data:
                items = data["answers"]
            elif isinstance(data, list):
                items = data
            else:
                items = []
            return [str(item).strip() for item in items if str(item).strip()]
        except json.JSONDecodeError:
            pass

        answers: List[str] = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:]
            elif stripped[0].isdigit() and "." in stripped[:3]:
                stripped = stripped.split(".", 1)[-1].strip()
            if stripped:
                answers.append(stripped)
        return answers


class AutoDiagnosisVerifier(LLMChatAgent):
    """Generate automatic ICD-10 diagnosis with chain-of-thought."""

    SYSTEM_PROMPT = """你是一位经验丰富的精神科医生。请阅读以下初次精神科门诊的问诊对话记录，并根据ICD-10国际疾病分类标准，仔细分析后输出患者诊断结束后的ICD-10诊断代码。

## 疾病分类说明
请仅从以下ICD-10标准中的10种疾病中选择最符合的诊断大类以及进一步细分的小类：
    - F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。
        F32.0 轻度抑郁发作：症状轻，社会功能影响有限。
        F32.1 中度抑郁发作：症状更明显，日常活动受限。
        F32.2 重度抑郁发作，无精神病性症状：症状显著，丧失功能，但无妄想/幻觉。
        F32.3 重度抑郁发作，有精神病性症状：伴有抑郁性妄想、幻觉或木僵。
        F32.8 其他抑郁发作；F32.9 抑郁发作，未特指。
    - F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。
        F41.0 惊恐障碍：突发的强烈恐慌发作，常伴濒死感。
        F41.1 广泛性焦虑障碍：长期持续的过度担忧和紧张不安。
        F41.2 混合性焦虑与抑郁障碍：焦虑与抑郁并存但均不足以单独诊断。
        F41.3 其他混合性焦虑障碍：混合焦虑表现但未完全符合特定标准。
        F41.9 焦虑障碍，未特指：存在焦虑症状但资料不足以分类。
    - F39.x00 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。
    - F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。
        F51.0 非器质性失眠：入睡困难、易醒或睡眠不恢复精力。
        F51.1 非器质性嗜睡：过度睡眠或难以保持清醒。
        F51.2 非器质性睡眠-觉醒节律障碍：昼夜节律紊乱导致睡眠异常。
        F51.3 梦魇障碍：频繁恶梦导致醒后强烈不安。
        F51.4 睡眠惊恐（夜惊）：夜间突然惊恐醒来伴强烈焦虑反应。
        F51.5 梦游症：睡眠中出现起床或行走等复杂行为。
        F51.9 非器质性睡眠障碍，未特指：睡眠异常但无具体分类。
    - F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。
        F98.0 非器质性遗尿症：儿童在不适当年龄仍有排尿失控。
        F98.1 非器质性遗粪症：儿童在不适当情境排便。
        F98.2 婴儿期或儿童期进食障碍：儿童进食行为异常影响营养或发育。
        F98.3 异食癖：持续摄入非食物性物质。
        F98.4 刻板性运动障碍：重复、无目的的运动习惯。
        F98.5 口吃：言语流利性障碍，表现为言语阻塞或重复。
        F98.6 习惯性动作障碍：如咬甲、吮指等持续存在的习惯。
        F98.8 其他特指的儿童行为和情绪障碍：符合儿童期特异但不归入上述类。
        F98.9 未特指的儿童行为和情绪障碍：症状存在但缺乏分类依据。
    - F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。
        F42.0 以强迫观念为主：反复出现难以摆脱的思想或冲动。
        F42.1 以强迫行为为主：反复、仪式化的动作难以控制。
        F42.2 强迫观念与强迫行为混合：思想和动作同时反复困扰。
        F42.9 强迫障碍，未特指：存在强迫症状但分类不详。
    - F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。
        F31.0 躁狂期，无精神病性症状：躁狂明显但无妄想或幻觉。
        F31.1 躁狂期，有精神病性症状：躁狂发作伴妄想或幻觉。
        F31.2 抑郁期，无精神病性症状：抑郁发作但无精神病性特征。
        F31.3 抑郁期，有精神病性症状：抑郁伴妄想或幻觉。
        F31.4 混合状态：躁狂与抑郁症状同时或快速交替出现。
        F31.5 缓解期：既往双相障碍，当前症状缓解。
        F31.6 其他状态：不符合典型躁狂/抑郁/混合的表现。
        F31.9 未特指：双相障碍，但无法进一步分类。
    - F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。
        F43.0 急性应激反应：暴露于重大应激后立即出现短暂严重反应。
        F43.1 创伤后应激障碍：经历创伤事件后持续出现再体验、回避和警觉性增高。
        F43.2 适应障碍：对生活变故反应过度，伴情绪或行为异常。
        F43.8 其他反应性障碍：与应激相关但不符合特定诊断。
        F43.9 未特指：应激反应存在，但资料不足以分类。
    - F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。
        F45.0 躯体化障碍：反复多样的身体症状无器质性解释。
        F45.1 未分化的躯体形式障碍：躯体症状存在但未达到躯体化标准。
        F45.2 疑病障碍：持续担忧患严重疾病。
        F45.3 自主神经功能紊乱型：以心悸、胸闷等自主神经症状为主。
        F45.4 持续性躯体疼痛障碍：慢性疼痛为主要表现。
        F45.8 其他躯体形式障碍：特殊类型躯体症状但不归入上述类。
        F45.9 未特指：存在躯体症状但无法分类。
    - F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。
        F20.0 偏执型：以妄想和幻听为主。
        F20.1 紊乱型：思维、情感和行为紊乱显著。
        F20.2 紧张型：以木僵、紧张性兴奋为主要表现。
        F20.3 未分化型：符合精神分裂症但不属特定亚型。
        F20.4 残留状态：阴性症状为主，病程长期。
        F20.5 精神分裂症后抑郁：精神分裂症后出现显著抑郁。
        F20.6 单纯型：逐渐出现阴性症状，无显著阳性症状。
        F20.8 其他类型：特殊表现但不属于前述类别。
        F20.9 未特指：存在精神分裂症特征但资料不足。
    - Z71 咨询和医疗建议相关因素：包括心理咨询、健康教育、生活方式指导等，当患者主要需要咨询服务而非特定疾病治疗时使用。
        Z71.9 未特指的咨询：提供咨询，但缺乏具体分类。

## 注意：
1. 问诊对话为初次问诊，在症状严重程度和细节不可判断的时候，请推荐未特指的icd code。
2. 诊断结果可能包含1至2个icd-10诊断结果，大多只包含一个。
3. 用分号分隔不同的代码。
4. 需要严格根据icd-10标准来进行诊断的分析, 避免猜测和无根据的诊断，避免诊断错误。"""

    USER_PROMPT_TEMPLATE = """[问诊对话开始]
{dialogue}
[问诊对话结束]

## 输出格式：
请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将最后诊断的ICD-10代码必须放在<box>xxx</box>中输出，用分号分隔，格式如：<think>xxx</think><box>Fxx.x;Fxx.x;Fxx.x</box>。"""


    def __init__(
        self,
        model_config: Dict[str, Any],
        openrouter_config: Dict[str, Any],
    ) -> None:
        super().__init__(model_config, openrouter_config, agent_name="auto_diagnosis")

    def generate_diagnosis(
        self, conversation_log: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Produce automatic diagnosis output."""
        dialogue_text = _format_dialogue(conversation_log, limit=len(conversation_log))
        if not dialogue_text:
            raise ValueError("缺少有效的问诊记录，无法生成自动诊断")

        user_prompt = self.USER_PROMPT_TEMPLATE.format(dialogue=dialogue_text)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print("messages: \n%s", messages)
        
        completion = self._chat(
            messages,
            temperature=0.1,
            top_p=0.95,
            max_tokens=8096,
        )
        
        print("completion: \n%s", completion.choices[0])

        choice_message = completion.choices[0].message
        content = _coerce_message_text(choice_message).strip()
        if not content:
            raise RuntimeError("模型未返回诊断内容")

        # 提取<think>标签中的内容
        thought = _extract_tag(content, "think")
        
        # 如果没有<think>标签，尝试从reasoning_content字段获取
        reasoning_content = ""
        if hasattr(choice_message, 'reasoning_content') and choice_message.reasoning_content:
            reasoning_content = choice_message.reasoning_content.strip()
            print(f"从reasoning_content字段提取到内容: {reasoning_content}")
        elif hasattr(choice_message, 'reasoning') and choice_message.reasoning:
            reasoning_content = choice_message.reasoning.strip()
            print(f"从reasoning字段提取到内容: {reasoning_content}")
        
        # 优先使用<think>标签内容，如果没有则使用reasoning字段
        final_thought = thought if thought else reasoning_content
        
        icd_box = _extract_tag(content, "box")
        icd_codes = []
        if icd_box:
            icd_codes = [code.strip() for code in icd_box.split(";") if code.strip()]

        print(f"最终提取结果 - thought: {final_thought}, icd_box: {icd_box}, icd_codes: {icd_codes}")

        return {
            "raw": content,
            "thought": final_thought,
            "icd_box": icd_box,
            "icd_codes": icd_codes,
            "model": self.model_identifier,
            "reasoning": reasoning_content,  # 添加原始reasoning字段
            }
