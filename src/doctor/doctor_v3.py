# -*- coding: utf-8 -*-
"""
医生角色AI模块 V2版本 - RAG增强版本（支持本地模型 + CoT）

该模块在V2版本的基础上，集成了RAG（检索增强生成）功能和CoT（Chain of Thought）：
- 使用PDF文档构建的知识库
- 在问诊过程中检索相关知识辅助提问（CoT推理）
- 在生成诊断时参考诊疗指南
- 支持本地部署的大模型
"""

import sys
import os
import logging
import json
import re

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

from src.doctor.doctor_v2 import Doctor
from src.doctor.reasoning_generator import ReasoningGenerator
from src.rag import VectorStore
from src.rag.rag_config import RAGConfig
from src.llm import llm_tools_api
from src.doctor.diagtree_v2 import DiagnosticPhase

# 尝试导入 Reranker
try:
    from src.rag.reranker import Reranker
    RerankerAvailable = True
except ImportError:
    RerankerAvailable = False
    Reranker = None

# 尝试导入 DeepInfra Reranker
try:
    from src.rag.deepinfra_reranker import DeepInfraReranker
    DeepInfraRerankerAvailable = True
except ImportError:
    DeepInfraRerankerAvailable = False
    DeepInfraReranker = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================== Pydantic 模型定义 ==================
class DiseaseCandidate(BaseModel):
    """单个候选疾病模型"""
    icd_code: str = Field(description="ICD-10疾病代码，例如 F32.1")
    disease_name: str = Field(description="疾病名称，例如 中度抑郁发作")
    reason: str = Field(description="为什么可能是这个疾病的简要说明")
    possibility: str = Field(description="可能性等级：A（最高）、B（中等）、C（较低）")


class PossibleDiseases(BaseModel):
    """3个候选疾病的结构化模型"""
    disease_a: DiseaseCandidate = Field(description="可能性最高的疾病")
    disease_b: DiseaseCandidate = Field(description="可能性中等的疾病")
    disease_c: DiseaseCandidate = Field(description="可能性较低的疾病")


class DoctorV2RAGCoT(Doctor):
    """
    RAG增强的医生类 V2版本（支持本地模型 + CoT）
    
    在V2版本的基础上，添加了知识库检索功能和CoT推理：
    - 在问诊过程中检索诊疗指南（CoT推理）
    - 在生成诊断时参考诊疗指南
    - 支持本地部署的大模型
    """
    
    def __init__(self, patient_template, doctor_prompt_path, diagtree_path, model_path, use_api,
                 knowledge_base_path=None, enable_rag=True, enable_cot=True):
        """
        初始化RAG增强的医生对象 V2版本（支持本地模型 + CoT）
        
        Args:
            patient_template (dict): 患者模板信息
            doctor_prompt_path (str): 医生角色配置文件路径
            diagtree_path (str): 诊断树配置文件路径
            model_path (str): 模型路径或API模型名称（本地模型使用完整路径）
            use_api (bool): 是否使用API方式调用模型（False表示使用本地模型）
            knowledge_base_path (str): 知识库路径（索引文件路径，不包含扩展名）
            enable_rag (bool): 是否启用RAG功能
            enable_cot (bool): 是否启用CoT（Chain of Thought）推理
        """
        # 调用父类初始化（会初始化diagnosis_tree和topic_seq）
        super().__init__(patient_template, doctor_prompt_path, diagtree_path, model_path, use_api)
        
        # RAG相关属性
        self.enable_rag = enable_rag
        self.enable_cot = enable_cot
        self.vector_store = None
        self.knowledge_base_path = knowledge_base_path
        self.rag_search_history = []  # 记录检索历史
        self.reranker = None  # Reranker 模型
        
        # RAG/CoT 详细追踪信息（用于输出JSON）
        self.rag_cot_trace = []  # 记录每个步骤的详细信息
        
        # 话题结束后的共病分析结果（用于指导下一个话题的提问）
        self.topic_comorbidity_analysis = {}  # 格式: {topic_idx: {'diseases': ..., 'knowledge_context': ...}}
        
        # Reasoning生成器（可选）
        self.reasoning_generator = None                 # Reasoning生成器对象
        self.enable_reasoning = False                   # 是否启用reasoning生成
        self.turn_counter = 0                           # 对话轮次计数器
        self.current_rag_trace = None                   # 当前轮次的RAG追踪信息
        
        # 阶段到话题索引的映射（用于判断当前阶段）
        # 必须在super().__init__()之后调用，因为需要diagnosis_tree和topic_seq已经初始化
        self.phase_topic_ranges = {}  # 格式: {phase: (start_idx, end_idx)}
        self._build_phase_topic_mapping()
        
        # 初始化RAG
        if self.enable_rag:
            self._init_rag()
            self._init_reranker()
    
    def _get_device(self):
        """
        获取当前可用的计算设备
        
        如果使用本地模型且模型已加载，返回模型的设备
        否则，检查 GPU 可用性并返回相应设备
        
        Returns:
            torch.device: 可用的计算设备（cuda 或 cpu）
        """
        import torch
        
        # 如果有本地模型且模型有 device 属性，使用模型的设备
        if hasattr(self, 'doctor_model') and self.doctor_model is not None:
            if hasattr(self.doctor_model, 'device'):
                return self.doctor_model.device
        
        # 否则，检查 GPU 可用性
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _init_rag(self):
        """初始化RAG系统"""
        if not self.knowledge_base_path:
            # 使用默认知识库路径
            default_index_path = RAGConfig.KNOWLEDGE_BASE_INDEX_DIR / "faiss_index"
            if default_index_path.with_suffix('.index').exists():
                self.knowledge_base_path = str(default_index_path)
                logger.info(f"使用默认知识库索引: {self.knowledge_base_path}")
            else:
                logger.warning("未找到知识库索引，RAG功能将不可用")
                logger.info(f"请先运行 build_knowledge_base.py 构建知识库")
                self.enable_rag = False
                return
        
        try:
            # 初始化向量存储
            self.vector_store = VectorStore()
            
            # 加载知识库索引
            index_path = self.knowledge_base_path
            if os.path.exists(index_path + ".index"):
                self.vector_store.load_index(index_path)
                logger.info(f"✓ RAG知识库加载成功: {index_path}")
            else:
                logger.warning(f"索引文件不存在: {index_path}.index，RAG功能将不可用")
                logger.info(f"请先运行 build_knowledge_base.py 构建知识库")
                self.enable_rag = False
                
        except Exception as e:
            logger.error(f"初始化RAG系统失败: {e}")
            logger.warning("RAG功能将不可用，将使用基础模式")
            self.enable_rag = False
    
    def _build_phase_topic_mapping(self):
        """构建阶段到话题索引的映射关系"""
        if not hasattr(self, 'diagnosis_tree') or not self.diagnosis_tree:
            return
        
        # 获取诊断树对象
        diagtree = self.diagnosis_tree
        
        # 计算每个阶段的话题范围
        current_idx = 0
        for phase in DiagnosticPhase:
            config = diagtree.phase_configs[phase]
            
            # 计算该阶段的话题数量
            mandatory_count = len(config.mandatory_topics)
            optional_count = min(
                config.max_questions - mandatory_count,
                len(config.optional_topics)
            )
            phase_topic_count = mandatory_count + optional_count
            
            # 记录该阶段的话题范围
            start_idx = current_idx
            end_idx = current_idx + phase_topic_count
            self.phase_topic_ranges[phase] = (start_idx, end_idx)
            
            current_idx = end_idx
        
        logger.info(f"[阶段映射] 已构建阶段到话题索引的映射:")
        for phase, (start, end) in self.phase_topic_ranges.items():
            logger.info(f"  {phase.value}: 话题索引 {start}-{end-1}")
    
    def _get_current_phase(self) -> Optional[DiagnosticPhase]:
        """根据当前话题索引获取当前阶段"""
        if not self.phase_topic_ranges:
            return None
        
        current_idx = getattr(self, 'current_idx', 0)
        
        for phase, (start_idx, end_idx) in self.phase_topic_ranges.items():
            if start_idx <= current_idx < end_idx:
                return phase
        
        return None
    
    def _should_use_rag_cot(self) -> bool:
        """判断当前是否应该使用RAG和CoT（只在评估和深入阶段）"""
        if not self.enable_rag or not self.enable_cot:
            return False
        
        current_phase = self._get_current_phase()
        
        if current_phase is None:
            logger.warning(f"[RAG/CoT] 无法确定当前阶段，跳过RAG和CoT")
            return False
        
        # 只在评估阶段和深入阶段启用RAG和CoT
        should_use = current_phase in [DiagnosticPhase.ASSESSMENT, DiagnosticPhase.DEEPDIVE]
        
        if should_use:
            logger.info(f"[RAG/CoT] 当前阶段: {current_phase.value}，启用RAG和CoT")
        else:
            logger.debug(f"[RAG/CoT] 当前阶段: {current_phase.value}，跳过RAG和CoT（仅在评估和深入阶段启用）")
        
        return should_use
    
    def _analyze_comorbidity_after_topic(self, completed_topic: str, dialogue_history: list) -> dict:
        """
        在话题结束后进行共病分析（仅在评估和深入阶段）
        
        Args:
            completed_topic: 刚完成的话题
            dialogue_history: 完整的对话历史
            
        Returns:
            共病分析结果字典，包含：
            - diseases: 3个候选疾病（使用疾病名称）
            - knowledge_context: RAG检索到的知识上下文
            - analysis_summary: 分析摘要
        """
        if not self._should_use_rag_cot():
            return {}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[共病分析] 话题'{completed_topic}'结束后，开始进行共病分析")
        logger.info(f"{'='*80}")
        
        # 创建追踪记录
        trace_entry = {
            'stage': 'comorbidity_analysis_after_topic',  # 话题结束后的共病分析
            'completed_topic': completed_topic,
            'dialogue_history_length': len(dialogue_history),
            'steps': []
        }
        
        # 步骤1: RAG检索 - 基于完整对话历史
        comorbidity_query = f"基于以下完整对话历史，分析患者可能的疾病诊断：{dialogue_history[-10:]}"
        search_info = self._search_knowledge(comorbidity_query, top_k=RAGConfig.TOP_K_RESULTS)
        
        if not search_info or not search_info.get('filtered_results'):
            logger.warning(f"[共病分析] 未检索到相关知识")
            trace_entry['steps'].append({
                'step_name': 'rag_search',
                'query': comorbidity_query,
                'error': 'No results found'
            })
            self.rag_cot_trace.append(trace_entry)
            return {}
        
        # 提取检索结果
        search_results = search_info['filtered_results']
        knowledge_context = self._format_knowledge_context(search_results)
        
        # 记录检索步骤
        trace_entry['steps'].append({
            'step_name': 'rag_search',
            'query': comorbidity_query,
            'num_initial_results': search_info['num_initial_results'],
            'num_final_results': search_info['num_final_results'],
            'used_reranking': search_info['used_reranking'],
            'retrieved_knowledge': [
                {
                    'rank': i + 1,
                    'text': r.get('text', ''),
                    'score': r.get('score', 0),
                    'rerank_score': r.get('rerank_score'),
                    'source': r.get('source', 'unknown')
                }
                for i, r in enumerate(search_results)
            ]
        })
        
        # 步骤2: 提取3个候选疾病
        possible_diseases = ""
        if knowledge_context:
            logger.info(f"[共病分析] 开始从知识中提取3个候选疾病")
            possible_diseases = self._extract_possible_diseases_from_top3(knowledge_context, dialogue_history, trace_entry)
        
        if not possible_diseases:
            logger.warning(f"[共病分析] 未能提取到候选疾病")
            self.rag_cot_trace.append(trace_entry)
            return {}
        
        # 步骤3: 生成分析摘要
        analysis_summary = f"""基于话题"{completed_topic}"的对话内容，通过RAG检索诊疗指南，识别出3个最可能的候选疾病：
{possible_diseases}

这些候选疾病将用于指导后续话题的针对性提问，以帮助区分和确定最终诊断。"""
        
        # 保存分析结果
        analysis_result = {
            'diseases': possible_diseases,
            'knowledge_context': knowledge_context,
            'analysis_summary': analysis_summary,
            'completed_topic': completed_topic,
            'dialogue_length': len(dialogue_history)
        }
        
        # 记录追踪信息
        trace_entry['possible_diseases'] = possible_diseases
        trace_entry['knowledge_context'] = knowledge_context
        trace_entry['analysis_summary'] = analysis_summary
        trace_entry['analysis_result'] = analysis_result
        self.rag_cot_trace.append(trace_entry)
        
        logger.info(f"[共病分析] 共病分析完成:")
        logger.info(f"  {analysis_summary}")
        logger.info(f"{'='*80}\n")
        
        return analysis_result
    
    def _init_reranker(self):
        """初始化 Reranker 模型"""
        if not RAGConfig.ENABLE_RERANKING:
            logger.info("Reranking 功能未启用")
            return
        
        # 检查是否使用 DeepInfra Reranker
        use_deepinfra = RAGConfig.USE_DEEPINFRA_RERANKER
        
        if use_deepinfra:
            # 使用 DeepInfra Reranker
            if not DeepInfraRerankerAvailable:
                logger.warning("DeepInfra Reranker 模块不可用，请检查依赖")
                return
            
            try:
                self.reranker = DeepInfraReranker(
                    api_key=RAGConfig.DEEPINFRA_API_KEY,
                    model_name=RAGConfig.DEEPINFRA_RERANKER_MODEL
                )
                logger.info("✓ DeepInfra Reranker 模型加载成功")
            except Exception as e:
                logger.warning(f"初始化 DeepInfra Reranker 失败: {e}，将不使用 Reranking")
                logger.info("提示: 请检查 DEEPINFRA_API_KEY 是否正确设置")
                self.reranker = None
        else:
            # 使用本地 Reranker
            if not RerankerAvailable:
                logger.warning("Reranker 模块不可用，请安装 transformers 库")
                return
            
            try:
                self.reranker = Reranker()
                logger.info("✓ Reranker 模型加载成功")
            except Exception as e:
                logger.warning(f"初始化 Reranker 失败: {e}，将不使用 Reranking")
                logger.info("提示: 如果不需要 Reranking，可以在 .env 文件中设置 ENABLE_RERANKING=false")
                self.reranker = None
    
    def _search_knowledge(self, query: str, top_k: int = 5) -> list:
        """
        检索知识库
        
        Args:
            query: 检索查询
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        if not self.enable_rag or self.vector_store is None:
            return []
        
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"[RAG检索] 查询: {query[:150]}...")
            logger.info(f"{'='*80}")
            
            # 初始检索：获取更多候选结果（如果启用reranking，需要更多候选）
            initial_top_k = RAGConfig.RERANKER_TOP_K * 2 if (self.reranker and RAGConfig.ENABLE_RERANKING) else top_k
            initial_top_k = max(initial_top_k, top_k)  # 至少返回 top_k 个结果
            
            results = self.vector_store.search(query, top_k=initial_top_k, return_scores=True)
            
            # 过滤低分结果
            filtered_results = [
                r for r in results 
                if r.get('score', 0) >= RAGConfig.SCORE_THRESHOLD
            ]
            
            # 如果启用 Reranking，对结果进行重排序，并取top3用于疾病分析
            if self.reranker and RAGConfig.ENABLE_RERANKING and filtered_results:
                logger.info(f"[Reranking] 对 {len(filtered_results)} 个候选结果进行重排序...")
                try:
                    # Reranker后取top3用于疾病分析
                    reranked_results = self.reranker.rerank_search_results(
                        query=query,
                        search_results=filtered_results,
                        top_k=3  # 固定取top3用于疾病分析
                    )
                    filtered_results = reranked_results
                    logger.info(f"[Reranking] 重排序完成，返回前 {len(filtered_results)} 个结果（用于疾病分析）")
                except Exception as e:
                    logger.warning(f"Reranking 失败: {e}，使用原始检索结果")
                    filtered_results = filtered_results[:min(3, top_k)]  # 失败时也取top3
            else:
                # 未启用 Reranking，直接截取 top_k（但用于疾病分析时取top3）
                filtered_results = filtered_results[:min(3, top_k)]
            
            # 详细输出检索结果
            logger.info(f"[RAG检索] 检索到 {len(results)} 条结果，最终返回 {len(filtered_results)} 条")
            for i, result in enumerate(filtered_results[:3], 1):  # 只显示前3条
                score = result.get('score', 0)
                rerank_score = result.get('rerank_score')
                text = result.get('text', '')[:200]  # 只显示前200字符
                if rerank_score is not None:
                    logger.info(f"  [{i}] 重排序分数: {rerank_score:.4f} (原始: {result.get('original_score', 0):.4f})")
                else:
                    logger.info(f"  [{i}] 相似度: {score:.4f}")
                logger.info(f"      内容: {text}...")
            
            # 记录检索历史
            self.rag_search_history.append({
                'query': query,
                'num_results': len(filtered_results),
                'top_score': filtered_results[0]['score'] if filtered_results else 0,
                'used_reranking': self.reranker is not None and RAGConfig.ENABLE_RERANKING
            })
            
            # 返回详细结果（包含原始检索结果用于追踪）
            return {
                'filtered_results': filtered_results,
                'all_results': results,
                'query': query,
                'num_initial_results': len(results),
                'num_final_results': len(filtered_results),
                'used_reranking': self.reranker is not None and RAGConfig.ENABLE_RERANKING
            }
        except Exception as e:
            logger.error(f"检索知识库失败: {e}")
            return {
                'filtered_results': [],
                'all_results': [],
                'query': query,
                'num_initial_results': 0,
                'num_final_results': 0,
                'used_reranking': False,
                'error': str(e)
            }
    
    def _format_knowledge_context(self, search_results: list) -> str:
        """
        格式化检索到的知识为上下文
        
        Args:
            search_results: 检索结果列表
            
        Returns:
            格式化后的知识上下文
        """
        if not search_results:
            return ""
        
        context_parts = ["【诊疗指南参考】"]
        for i, result in enumerate(search_results, 1):
            text = result['text']
            score = result.get('score', 0)
            source = result.get('source', '未知来源')
            
            # 截断过长的文本
            if len(text) > 300:
                text = text[:300] + "..."
            
            context_parts.append(f"{i}. {text} (相似度: {score:.3f}, 来源: {source})")
        
        return "\n".join(context_parts)
    
    def _extract_possible_diseases_from_top3(self, knowledge_context: str, dialogue_history: list, trace_entry: dict = None) -> str:
        """
        基于RAG检索的top3知识上下文，提取最可能的3个疾病（使用疾病名称）
        使用Pydantic模型确保返回结构化JSON格式
        
        Args:
            knowledge_context: RAG检索到的top3知识上下文
            dialogue_history: 对话历史
            trace_entry: 追踪记录字典（可选）
            
        Returns:
            可能的3个疾病列表（字符串格式，用于后续处理）
        """
        if not knowledge_context:
            return ""
        
        # 改进的疾病提取提示：更详细的指导
        # 提取关键症状
        dialogue_text = ' '.join(dialogue_history[-5:]) if len(dialogue_history) >= 5 else ' '.join(dialogue_history)
        
        # 构建提取疾病的提示（要求返回JSON格式）
        disease_prompt = f"""
基于以下诊疗指南知识和对话历史，分析患者最可能的3个疾病诊断：

{knowledge_context}

## 对话历史摘要：
{dialogue_text}

## 分析要求：

**第一步：症状识别**
请识别患者的主要症状，包括：
- 情绪症状（焦虑、抑郁、恐惧等）
- 躯体症状（心慌、胸闷、睡眠问题等）
- 认知症状（注意力、记忆力等）
- 行为症状（回避、强迫行为等）

**第二步：疾病匹配**
根据诊疗指南，分析：
1. 哪些疾病的诊断标准与患者症状最匹配？
2. 需要重点考虑哪些疾病类别（焦虑障碍、抑郁障碍、混合性障碍等）？
3. 每个疾病的匹配程度如何？

**第三步：可能性排序**
按可能性从高到低排序3个候选疾病，每个疾病必须包含：
- ICD-10代码（如F41.1、F32.2等）
- 疾病名称（如"广泛性焦虑障碍"、"重度抑郁发作"等）
- 为什么可能的简要说明（基于哪些症状和指南标准）

**输出要求**：
- 必须列出3个可能的疾病诊断（使用疾病名称，不要使用A、B、C标记）
- 每个疾病必须包含：
  * ICD-10代码（如F41.1、F32.2等）
  * 疾病名称（如"广泛性焦虑障碍"、"重度抑郁发作"等）
  * 为什么可能的简要说明（基于哪些症状和指南标准）
- 必须以JSON格式返回，严格按照提供的JSON schema结构
- 如果无法确定3个疾病，也要尽量列出，并说明原因
"""
        
        disease_system_prompt = "你是一位经验丰富的精神科医生，擅长根据诊疗指南和症状分析可能的疾病诊断。你必须严格按照JSON格式返回结果。"
        disease_messages = [
            {"role": "system", "content": disease_system_prompt},
            {"role": "user", "content": disease_prompt}
        ]
        
        try:
            if self.use_api:
                # API模式：使用结构化输出
                # 获取Pydantic模型的JSON schema
                json_schema = PossibleDiseases.model_json_schema()
                
                # 构建OpenAI兼容的response_format
                # 对于OpenAI API，使用response_format参数
                try:
                    # 尝试使用结构化输出（OpenAI格式）
                    disease_response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=disease_messages,
                        temperature=0.3,
                        max_tokens=400,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "possible_diseases",
                                "strict": True,
                                "schema": json_schema,
                                "description": "3个候选疾病的结构化数据"
                            }
                        }
                    )
                except Exception as schema_error:
                    # 如果不支持json_schema格式，回退到json_object格式
                    logger.warning(f"结构化输出格式不支持，使用json_object格式: {schema_error}")
                    # 在prompt中明确要求JSON格式
                    disease_prompt_with_json = disease_prompt + "\n\n请以以下JSON格式返回：\n" + json.dumps(json_schema, ensure_ascii=False, indent=2)
                    disease_messages_json = [
                        {"role": "system", "content": disease_system_prompt},
                        {"role": "user", "content": disease_prompt_with_json}
                    ]
                    disease_response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=disease_messages_json,
                        temperature=0.3,
                        max_tokens=400,
                        response_format={"type": "json_object"}
                    )
                
                if disease_response and disease_response.choices and len(disease_response.choices) > 0:
                    response_content = disease_response.choices[0].message.content
                    
                    if not response_content or response_content.strip() == "":
                        if trace_entry:
                            trace_entry['steps'].append({
                                'step_name': 'disease_extraction',
                                'system_prompt': disease_system_prompt,
                                'user_prompt': disease_prompt,
                                'output': '',
                                'error': 'Empty response'
                            })
                        return ""
                    
                    # 解析JSON响应并验证
                    try:
                        # 尝试解析JSON
                        if isinstance(response_content, str):
                            # 如果响应是字符串，尝试提取JSON部分
                            response_content = response_content.strip()
                            # 移除可能的markdown代码块标记
                            if response_content.startswith("```json"):
                                response_content = response_content[7:]
                            if response_content.startswith("```"):
                                response_content = response_content[3:]
                            if response_content.endswith("```"):
                                response_content = response_content[:-3]
                            response_content = response_content.strip()
                        
                        # 解析JSON
                        diseases_dict = json.loads(response_content)
                        
                        # 使用Pydantic模型验证
                        diseases_model = PossibleDiseases(**diseases_dict)
                        
                        # 转换为可读的字符串格式（使用疾病名称而不是A、B、C）
                        possible_diseases = self._format_diseases_with_names(diseases_model)
                        
                        logger.info(f"[疾病分析] 从top3知识中提取出3个候选疾病（JSON格式）:")
                        logger.info(f"  {possible_diseases}")
                        
                        # 记录追踪信息（包含原始JSON）
                        if trace_entry:
                            trace_entry['steps'].append({
                                'step_name': 'disease_extraction',
                                'system_prompt': disease_system_prompt,
                                'user_prompt': disease_prompt,
                                'temperature': 0.3,
                                'max_tokens': 400,
                                'output': possible_diseases,
                                'json_output': diseases_model.model_dump(),  # 保存结构化数据
                                'json_schema_used': True
                            })
                        
                        return possible_diseases
                        
                    except json.JSONDecodeError as json_error:
                        logger.warning(f"JSON解析失败: {json_error}，响应内容: {response_content[:200]}")
                        # 如果JSON解析失败，尝试直接使用原始响应
                        if trace_entry:
                            trace_entry['steps'].append({
                                'step_name': 'disease_extraction',
                                'system_prompt': disease_system_prompt,
                                'user_prompt': disease_prompt,
                                'raw_output': response_content,
                                'error': f'JSON解析失败: {str(json_error)}'
                            })
                        return response_content  # 回退到原始响应
                    
                    except Exception as validation_error:
                        logger.warning(f"Pydantic验证失败: {validation_error}，尝试使用原始响应")
                        if trace_entry:
                            trace_entry['steps'].append({
                                'step_name': 'disease_extraction',
                                'raw_output': response_content,
                                'error': f'Pydantic验证失败: {str(validation_error)}'
                            })
                        return response_content  # 回退到原始响应
                else:
                    if trace_entry:
                        trace_entry['steps'].append({
                            'step_name': 'disease_extraction',
                            'error': 'No response from API'
                        })
                    return ""
            else:
                # 本地模型模式：提示模型返回JSON格式
                # 在prompt中明确要求JSON格式
                json_schema = PossibleDiseases.model_json_schema()
                disease_prompt_with_json = disease_prompt + "\n\n请严格按照以下JSON schema格式返回：\n" + json.dumps(json_schema, ensure_ascii=False, indent=2)
                disease_messages_local = [
                    {"role": "system", "content": disease_system_prompt},
                    {"role": "user", "content": disease_prompt_with_json}
                ]
                
                text = self.doctor_tokenizer.apply_chat_template(
                    disease_messages_local,
                    tokenize=False,
                    add_generation_prompt=True
                )
                disease_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                disease_outputs = self.doctor_model.generate(
                    disease_inputs.input_ids,
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=True
                )
                disease_generated = disease_outputs[0][disease_inputs.input_ids.shape[1]:]
                response_content = self.doctor_tokenizer.decode(disease_generated, skip_special_tokens=True)
                
                # 尝试解析JSON
                try:
                    # 清理响应内容
                    response_content = response_content.strip()
                    if response_content.startswith("```json"):
                        response_content = response_content[7:]
                    if response_content.startswith("```"):
                        response_content = response_content[3:]
                    if response_content.endswith("```"):
                        response_content = response_content[:-3]
                    response_content = response_content.strip()
                    
                    diseases_dict = json.loads(response_content)
                    diseases_model = PossibleDiseases(**diseases_dict)
                    
                    # 转换为可读格式（使用疾病名称而不是A、B、C）
                    possible_diseases = self._format_diseases_with_names(diseases_model)
                    
                    logger.info(f"[疾病分析] 从top3知识中提取出3个候选疾病（JSON格式）:")
                    logger.info(f"  {possible_diseases}")
                    
                    if trace_entry:
                        trace_entry['steps'].append({
                            'step_name': 'disease_extraction',
                            'system_prompt': disease_system_prompt,
                            'user_prompt': disease_prompt_with_json,
                            'temperature': 0.3,
                            'max_tokens': 400,
                            'output': possible_diseases,
                            'json_output': diseases_model.model_dump(),
                            'json_schema_used': True
                        })
                    
                    return possible_diseases
                    
                except (json.JSONDecodeError, Exception) as parse_error:
                    logger.warning(f"本地模型JSON解析失败: {parse_error}，使用原始响应")
                    if trace_entry:
                        trace_entry['steps'].append({
                            'step_name': 'disease_extraction',
                            'raw_output': response_content,
                            'error': f'JSON解析失败: {str(parse_error)}'
                        })
                    return response_content  # 回退到原始响应
            
        except Exception as e:
            logger.warning(f"提取疾病失败: {e}")
            if trace_entry:
                trace_entry['steps'].append({
                    'step_name': 'disease_extraction',
                    'error': str(e)
                })
            return ""
    
    def _format_diseases_with_names(self, diseases_model: PossibleDiseases) -> str:
        """
        格式化候选疾病，使用疾病名称而不是 A、B、C 标记
        
        Args:
            diseases_model: PossibleDiseases 模型实例
            
        Returns:
            格式化后的疾病字符串
        """
        disease_a_name = diseases_model.disease_a.disease_name
        disease_b_name = diseases_model.disease_b.disease_name
        disease_c_name = diseases_model.disease_c.disease_name
        
        return f"""{disease_a_name}（{diseases_model.disease_a.icd_code}）：{diseases_model.disease_a.reason}，可能性最高
{disease_b_name}（{diseases_model.disease_b.icd_code}）：{diseases_model.disease_b.reason}
{disease_c_name}（{diseases_model.disease_c.icd_code}）：{diseases_model.disease_c.reason}"""
    
    def _get_disease_names(self, diseases_model: PossibleDiseases) -> tuple:
        """
        获取三个候选疾病的名称
        
        Args:
            diseases_model: PossibleDiseases 模型实例
            
        Returns:
            (disease_a_name, disease_b_name, disease_c_name)
        """
        return (
            diseases_model.disease_a.disease_name,
            diseases_model.disease_b.disease_name,
            diseases_model.disease_c.disease_name
        )
    
    def _analyze_disease_symptoms(self, diseases: str, current_topic: str, knowledge_context: str, trace_entry: dict = None) -> str:
        """
        分析3个候选疾病在特定话题下的不同症状特征
        
        Args:
            diseases: 3个候选疾病的描述（使用疾病名称）
            current_topic: 当前问诊话题
            knowledge_context: RAG检索到的知识上下文
            
        Returns:
            疾病症状差异分析（字符串格式）
        """
        if not diseases or not knowledge_context:
            return ""
        
        # 从diseases字符串中提取疾病名称（每行第一个疾病名称）
        disease_names = []
        for line in diseases.split('\n'):
            line = line.strip()
            if line:
                # 提取疾病名称（在括号或冒号之前的部分）
                if '（' in line:
                    disease_name = line.split('（')[0].strip()
                elif '：' in line:
                    disease_name = line.split('：')[0].strip()
                else:
                    disease_name = line.split(':')[0].strip() if ':' in line else line
                if disease_name:
                    disease_names.append(disease_name)
        
        # 如果成功提取了3个疾病名称，使用它们；否则使用通用描述
        if len(disease_names) >= 3:
            disease_a_name, disease_b_name, disease_c_name = disease_names[0], disease_names[1], disease_names[2]
        else:
            # 如果提取失败，使用通用描述
            disease_a_name, disease_b_name, disease_c_name = "第一个候选疾病", "第二个候选疾病", "第三个候选疾病"
        
        # 构建症状分析提示
        symptom_analysis_prompt = f"""
基于以下诊疗指南知识和3个候选疾病，分析它们在"{current_topic}"方面的不同症状特征：

{knowledge_context}

## 3个候选疾病：
{diseases}

请分析：
1. {disease_a_name}在{current_topic}方面的典型症状和特征是什么？
2. {disease_b_name}在{current_topic}方面的典型症状和特征是什么？
3. {disease_c_name}在{current_topic}方面的典型症状和特征是什么？
4. 这3个疾病在{current_topic}方面有哪些关键区别点？哪些症状或特征可以帮助区分它们？

请以清晰的格式列出每个疾病在{current_topic}方面的症状特征，并说明如何通过询问{current_topic}来区分这3个疾病。
"""
        
        symptom_system_prompt = "你是一位经验丰富的精神科医生，擅长分析不同疾病的症状特征和鉴别诊断。"
        symptom_messages = [
            {"role": "system", "content": symptom_system_prompt},
            {"role": "user", "content": symptom_analysis_prompt}
        ]
        
        try:
            if self.use_api:
                # API模式
                symptom_response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=symptom_messages,
                    temperature=0.3,
                    max_tokens=600
                )
                
                if symptom_response and symptom_response.choices and len(symptom_response.choices) > 0:
                    symptom_analysis = symptom_response.choices[0].message.content
                    if not symptom_analysis or symptom_analysis.strip() == "":
                        if trace_entry:
                            trace_entry['steps'].append({
                                'step_name': 'symptom_analysis',
                                'system_prompt': symptom_system_prompt,
                                'user_prompt': symptom_analysis_prompt,
                                'output': '',
                                'error': 'Empty response'
                            })
                        return ""
                else:
                    if trace_entry:
                        trace_entry['steps'].append({
                            'step_name': 'symptom_analysis',
                            'error': 'No response from API'
                        })
                    return ""
            else:
                # 本地模型模式
                text = self.doctor_tokenizer.apply_chat_template(
                    symptom_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                symptom_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                symptom_outputs = self.doctor_model.generate(
                    symptom_inputs.input_ids,
                    max_new_tokens=600,
                    temperature=0.3,
                    do_sample=True
                )
                symptom_generated = symptom_outputs[0][symptom_inputs.input_ids.shape[1]:]
                symptom_analysis = self.doctor_tokenizer.decode(symptom_generated, skip_special_tokens=True)
            
            if symptom_analysis and symptom_analysis.strip():
                logger.info(f"[症状分析] 分析3个疾病在{current_topic}方面的症状差异:")
                logger.info(f"  {symptom_analysis[:300]}...")
                # 记录追踪信息
                if trace_entry:
                    trace_entry['steps'].append({
                        'step_name': 'symptom_analysis',
                        'system_prompt': symptom_system_prompt,
                        'user_prompt': symptom_analysis_prompt,
                        'temperature': 0.3,
                        'max_tokens': 600,
                        'output': symptom_analysis
                    })
                return symptom_analysis
            else:
                logger.warning("[症状分析] 未能提取到症状分析，返回空")
                if trace_entry:
                    trace_entry['steps'].append({
                        'step_name': 'symptom_analysis',
                        'output': '',
                        'error': 'Empty output'
                    })
                return ""
                
        except Exception as e:
            logger.warning(f"症状分析失败: {e}")
            if trace_entry:
                trace_entry['steps'].append({
                    'step_name': 'symptom_analysis',
                    'error': str(e)
                })
            return ""
    
    def _extract_answer_from_content(self, content: str) -> str:
        """
        从content中提取<answer>标签内的内容，如果没有标签则返回原内容
        
        Args:
            content: 可能包含<answer>标签的内容
            
        Returns:
            提取的纯文本内容（去掉<answer>标签）
        """
        if not content:
            return content
        
        content = str(content).strip()
        
        # 检查是否包含<answer>标签（可能有闭合标签，也可能没有）
        if '<answer>' in content.lower() or '<answer>' in content:
            # 先尝试匹配有闭合标签的情况
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
            if answer_match:
                # 提取<answer>标签内的内容
                answer_content = answer_match.group(1).strip()
            else:
                # 如果没有闭合标签，提取<answer>标签后的所有内容
                answer_start = re.search(r'<answer>', content, re.IGNORECASE)
                if answer_start:
                    answer_content = content[answer_start.end():].strip()
                else:
                    # 如果找不到<answer>标签，使用原内容
                    answer_content = content
            
            # 去掉可能的"医生："前缀（如果存在）
            if answer_content.startswith('医生：'):
                answer_content = answer_content[3:].strip()
            return answer_content
        
        # 如果没有<answer>标签，检查是否包含<think>标签
        # 如果有，去掉<think>标签，保留剩余内容
        if '<think>' in content or '<think>' in content.lower():
            # 去掉所有thinking标签
            cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.strip()
            # 去掉可能的"医生："前缀（如果存在）
            if cleaned.startswith('医生：'):
                cleaned = cleaned[3:].strip()
            return cleaned
        
        # 如果都没有，直接返回原内容（去掉可能的"医生："前缀）
        if content.startswith('医生：'):
            return content[3:].strip()
        return content.strip()
    
    def _clean_patient_content_from_reasoning(self, reasoning: str, patient_indicators: list) -> Optional[str]:
        """
        清理 reasoning 中的患者部分，只保留医生的 reasoning
        
        Args:
            reasoning: 原始 reasoning 内容
            patient_indicators: 患者角色指示词列表
            
        Returns:
            清理后的 reasoning（只包含医生部分），如果整个 reasoning 都是患者的则返回 None
        """
        if not reasoning:
            return None
        
        reasoning = reasoning.strip()
        if not reasoning:
            return None
        
        # 检查是否包含患者指示词
        contains_patient_content = any(indicator in reasoning for indicator in patient_indicators)
        
        if not contains_patient_content:
            # 不包含患者内容，直接返回
            return reasoning
        
        # 包含患者内容，尝试分离医生和患者的部分
        # 策略1: 查找患者内容的分隔符（如"患者："、"患者说："等）
        patient_markers = [
            r'患者[：:]\s*',
            r'患者说[：:]\s*',
            r'作为患者[，,]\s*',
            r'我是一名.*?患者[，,。]\s*',
            r'扮演.*?患者[，,。]\s*',
            r'用户希望我扮演[，,。]\s*',
        ]
        
        # 尝试找到患者内容的开始位置
        patient_start_pos = len(reasoning)  # 默认认为患者内容在最后
        for marker in patient_markers:
            match = re.search(marker, reasoning, re.IGNORECASE)
            if match:
                patient_start_pos = min(patient_start_pos, match.start())
        
        # 如果找到了患者内容的开始位置，提取之前的部分（医生的部分）
        if patient_start_pos < len(reasoning):
            doctor_reasoning = reasoning[:patient_start_pos].strip()
            
            # 验证提取的部分是否还包含患者内容
            if doctor_reasoning and not any(indicator in doctor_reasoning for indicator in patient_indicators):
                logger.info(f"[Doctor V2 RAG CoT] ✅ 已从reasoning中移除患者部分，保留医生部分（长度: {len(doctor_reasoning)}字符）")
                return doctor_reasoning
            elif doctor_reasoning:
                # 提取的部分仍然包含患者内容，尝试更激进的清理
                # 按句子分割，只保留不包含患者指示词的句子
                sentences = re.split(r'[。！？\n]', doctor_reasoning)
                clean_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not any(indicator in sentence for indicator in patient_indicators):
                        clean_sentences.append(sentence)
                
                if clean_sentences:
                    cleaned_reasoning = '。'.join(clean_sentences)
                    logger.info(f"[Doctor V2 RAG CoT] ✅ 已从reasoning中移除患者部分（按句子过滤），保留医生部分（长度: {len(cleaned_reasoning)}字符）")
                    return cleaned_reasoning
                else:
                    # 所有句子都包含患者内容
                    logger.warning(f"[Doctor V2 RAG CoT] ⚠️ reasoning中所有内容都包含患者角色，清除整个reasoning")
                    return None
            else:
                # 患者内容在开头，整个 reasoning 都是患者的
                logger.warning(f"[Doctor V2 RAG CoT] ⚠️ reasoning完全包含患者角色内容，清除整个reasoning")
                return None
        else:
            # 没有找到明确的分隔符，但包含患者指示词
            # 尝试按句子过滤
            sentences = re.split(r'[。！？\n]', reasoning)
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not any(indicator in sentence for indicator in patient_indicators):
                    clean_sentences.append(sentence)
            
            if clean_sentences:
                cleaned_reasoning = '。'.join(clean_sentences)
                logger.info(f"[Doctor V2 RAG CoT] ✅ 已从reasoning中移除患者部分（按句子过滤），保留医生部分（长度: {len(cleaned_reasoning)}字符）")
                return cleaned_reasoning
            else:
                # 所有句子都包含患者内容
                logger.warning(f"[Doctor V2 RAG CoT] ⚠️ reasoning中所有内容都包含患者角色，清除整个reasoning")
                return None
    
    def _cot_reasoning_with_rag(self, current_topic: str, dialogue_history: list, previous_comorbidity: dict = None) -> str:
        """
        CoT推理：基于RAG检索的知识进行推理
        
        Args:
            current_topic: 当前问诊话题
            dialogue_history: 对话历史
            previous_comorbidity: 上一个话题的共病分析结果（可选）
            
        Returns:
            推理结果和检索到的知识上下文
        """
        if not self.enable_cot or not self.enable_rag:
            return ""
        
        # 创建追踪记录
        trace_entry = {
            'stage': 'consultation_cot',  # 问诊阶段CoT
            'current_topic': current_topic,
            'dialogue_history_length': len(dialogue_history),
            'has_previous_comorbidity': previous_comorbidity is not None,
            'steps': []
        }
        
        # 如果有上一个话题的共病分析结果，优先使用它
        if previous_comorbidity and previous_comorbidity.get('diseases'):
            logger.info(f"[RAG+CoT] 使用上一个话题的共病分析结果来指导当前话题")
            possible_diseases = previous_comorbidity['diseases']
            knowledge_context = previous_comorbidity.get('knowledge_context', '')
            
            # 记录使用共病分析结果
            trace_entry['steps'].append({
                'step_name': 'use_previous_comorbidity',
                'previous_topic': previous_comorbidity.get('completed_topic', ''),
                'diseases': possible_diseases,
                'knowledge_context': knowledge_context
            })
        else:
            # 构建检索查询：结合当前话题和最近对话
            recent_dialogue = dialogue_history[-4:] if len(dialogue_history) >= 4 else dialogue_history
            dialogue_text = "\n".join(recent_dialogue)
            
            # 构建查询
            query = f"关于{current_topic}，结合以下对话：{dialogue_text[:200]}"
            
            # 步骤1: RAG检索
            search_info = self._search_knowledge(query, top_k=RAGConfig.TOP_K_RESULTS)
            
            if not search_info or not search_info.get('filtered_results'):
                logger.debug(f"[RAG] 话题'{current_topic}'未检索到相关知识")
                trace_entry['steps'].append({
                    'step_name': 'rag_search',
                    'query': query,
                    'results': [],
                    'error': 'No results found'
                })
                self.rag_cot_trace.append(trace_entry)
                return ""
            
            # 提取检索结果
            search_results = search_info['filtered_results']
            all_results = search_info.get('all_results', [])
            
            # 记录检索步骤
            trace_entry['steps'].append({
                'step_name': 'rag_search',
                'query': query,
                'recent_dialogue': recent_dialogue,
                'num_initial_results': search_info['num_initial_results'],
                'num_final_results': search_info['num_final_results'],
                'used_reranking': search_info['used_reranking'],
                'retrieved_knowledge': [
                    {
                        'rank': i + 1,
                        'text': r.get('text', ''),
                        'score': r.get('score', 0),
                        'rerank_score': r.get('rerank_score'),
                        'source': r.get('source', 'unknown')
                    }
                    for i, r in enumerate(search_results)
                ]
            })
            
            # 格式化知识上下文（top3）
            knowledge_context = self._format_knowledge_context(search_results)
            logger.info(f"[RAG] 话题'{current_topic}'检索到 {len(search_results)} 条相关知识（top3）")
            logger.info(f"[RAG] 知识上下文长度: {len(knowledge_context)} 字符")
            
            # 步骤2: 基于top3知识提取3个候选疾病
            possible_diseases = ""
            if knowledge_context:
                logger.info(f"\n{'='*80}")
                logger.info(f"[疾病分析] 开始从top3知识中提取3个候选疾病")
                logger.info(f"{'='*80}")
                possible_diseases = self._extract_possible_diseases_from_top3(knowledge_context, dialogue_history, trace_entry)
                if possible_diseases:
                    logger.info(f"{'='*80}\n")
        
        # 步骤3: 分析3个疾病在current_topic方面的不同症状（需要possible_diseases和knowledge_context）
        symptom_analysis = ""
        if possible_diseases and knowledge_context:
            logger.info(f"\n{'='*80}")
            logger.info(f"[症状分析] 开始分析3个疾病在{current_topic}方面的症状差异")
            logger.info(f"{'='*80}")
            symptom_analysis = self._analyze_disease_symptoms(possible_diseases, current_topic, knowledge_context, trace_entry)
            if symptom_analysis:
                logger.info(f"{'='*80}\n")
        
        # CoT推理提示（基于3个候选疾病和症状分析）
        if possible_diseases and symptom_analysis:
            # 从possible_diseases字符串中提取疾病名称
            disease_names = []
            for line in possible_diseases.split('\n'):
                line = line.strip()
                if line:
                    # 提取疾病名称（在括号或冒号之前的部分）
                    if '（' in line:
                        disease_name = line.split('（')[0].strip()
                    elif '：' in line:
                        disease_name = line.split('：')[0].strip()
                    else:
                        disease_name = line.split(':')[0].strip() if ':' in line else line
                    if disease_name:
                        disease_names.append(disease_name)
            
            # 如果成功提取了3个疾病名称，使用它们；否则使用通用描述
            if len(disease_names) >= 3:
                disease_a_name, disease_b_name, disease_c_name = disease_names[0], disease_names[1], disease_names[2]
            else:
                disease_a_name, disease_b_name, disease_c_name = "第一个候选疾病", "第二个候选疾病", "第三个候选疾病"
            
            cot_prompt = f"""
你是一位经验丰富的精神科医生，正在对患者进行问诊。基于RAG检索的诊疗指南知识，你已经识别出3个最可能的候选疾病，现在需要通过有针对性的询问来区分它们。

## 诊疗指南参考：
{knowledge_context}

## 3个候选疾病（按可能性从高到低排序）：
{possible_diseases}

## 3个疾病在"{current_topic}"方面的症状差异分析：
{symptom_analysis}

## 当前任务：
你需要围绕"{current_topic}"进行问诊，目标是区分这3个候选疾病（{disease_a_name}、{disease_b_name}、{disease_c_name}），确定患者最可能患的是哪一个。

## 请思考并回答：
1. **当前信息评估**：结合症状差异分析，关于{current_topic}，从对话历史中已经了解了哪些信息？这些信息更倾向于支持哪个候选疾病（{disease_a_name}、{disease_b_name}或{disease_c_name}）？为什么？

2. **关键区分点识别**：根据症状差异分析，在{current_topic}方面，{disease_a_name}、{disease_b_name}、{disease_c_name}有哪些最关键的区别点？哪些症状或特征是{disease_a_name}特有的？哪些是{disease_b_name}特有的？哪些是{disease_c_name}特有的？

3. **针对性问题设计**：为了区分这3个候选疾病，关于{current_topic}你需要询问哪些关键问题？这些问题应该：
   - 能够获取{disease_a_name}、{disease_b_name}、{disease_c_name}在{current_topic}方面的特异性症状
   - 帮助排除可能性较低的疾病（例如，如果患者回答X，则排除{disease_a_name}）
   - 帮助确认可能性较高的疾病（例如，如果患者回答Y，则更可能是{disease_b_name}）
   - 设计成自然的口语化问题，让患者容易理解和回答

4. **问诊策略**：如何设计1-2个问题，使得患者的回答能够清晰地指向{disease_a_name}、{disease_b_name}或{disease_c_name}中的某一个？请具体说明：
   - 如果患者回答[某种情况]，则更可能是{disease_a_name}
   - 如果患者回答[另一种情况]，则更可能是{disease_b_name}
   - 如果患者回答[第三种情况]，则更可能是{disease_c_name}

请用2-3句话总结你的思考过程，重点说明：
- 如何通过询问{current_topic}来区分这3个候选疾病（{disease_a_name}、{disease_b_name}、{disease_c_name}）
- 应该询问哪些关键问题
- 如何根据患者的回答来判断是{disease_a_name}、{disease_b_name}还是{disease_c_name}
"""
        else:
            # 如果没有共病分析，使用原来的提示
            cot_prompt = f"""
基于以下诊疗指南知识，思考如何围绕"{current_topic}"进行问诊：

{knowledge_context}

请思考：
1. 根据诊疗指南，关于{current_topic}需要关注哪些关键点？
2. 结合当前对话历史，哪些方面已经了解，哪些还需要进一步询问？
3. 如何基于指南知识提出更专业、更有针对性的问题？

请简要总结你的思考过程（1-2句话）。
"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[CoT推理] 开始为话题'{current_topic}'进行推理（基于3个候选疾病）")
        if possible_diseases and symptom_analysis:
            logger.info(f"[CoT推理] 目标：通过询问{current_topic}来区分3个候选疾病")
        logger.info(f"[CoT推理] 推理提示长度: {len(cot_prompt)} 字符")
        logger.info(f"{'='*80}")
        
        # 步骤4: CoT推理
        cot_system_prompt = "你是一位经验丰富的精神科医生，擅长基于诊疗指南进行专业问诊，特别擅长通过有针对性的问题来区分和鉴别不同的疾病诊断。"
        cot_messages = [
            {"role": "system", "content": cot_system_prompt},
            {"role": "user", "content": cot_prompt}
        ]
        
        # 使用本地模型或API进行CoT推理
        try:
            if self.use_api:
                # API模式
                cot_response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=cot_messages,
                    temperature=0.3,
                    max_tokens=400
                )
                # 添加空值检查
                if cot_response and cot_response.choices and len(cot_response.choices) > 0:
                    cot_reasoning = cot_response.choices[0].message.content
                else:
                    logger.warning("CoT推理响应为空，跳过CoT推理")
                    raise ValueError("CoT推理响应为空")
            else:
                # 本地模型模式
                text = self.doctor_tokenizer.apply_chat_template(
                    cot_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                cot_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                cot_outputs = self.doctor_model.generate(
                    cot_inputs.input_ids,
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=True
                )
                cot_generated = cot_outputs[0][cot_inputs.input_ids.shape[1]:]
                cot_reasoning = self.doctor_tokenizer.decode(cot_generated, skip_special_tokens=True)
            
            # 记录CoT推理步骤
            trace_entry['steps'].append({
                'step_name': 'cot_reasoning',
                'system_prompt': cot_system_prompt,
                'user_prompt': cot_prompt,
                'temperature': 0.3,
                'max_tokens': 400,
                'reasoning_output': cot_reasoning
            })
            
            # 组合知识上下文、候选疾病分析、症状分析和推理结果
            if possible_diseases and symptom_analysis:
                combined_context = f"{knowledge_context}\n\n【3个候选疾病（按可能性从高到低排序）】\n{possible_diseases}\n\n【3个疾病在{current_topic}方面的症状差异分析】\n{symptom_analysis}\n\n【思考过程：如何通过询问{current_topic}来区分这3个候选疾病】\n{cot_reasoning}"
            elif possible_diseases:
                combined_context = f"{knowledge_context}\n\n【3个候选疾病（按可能性从高到低排序）】\n{possible_diseases}\n\n【思考过程】{cot_reasoning}"
            else:
                combined_context = f"{knowledge_context}\n\n【思考过程】{cot_reasoning}"
            
            # 记录最终组合的上下文
            trace_entry['final_combined_context'] = combined_context
            trace_entry['knowledge_context'] = knowledge_context
            trace_entry['possible_diseases'] = possible_diseases
            trace_entry['symptom_analysis'] = symptom_analysis
            trace_entry['cot_reasoning'] = cot_reasoning
            
            logger.info(f"[CoT推理] 推理完成")
            logger.info(f"[CoT推理] 推理结果:")
            logger.info(f"  {cot_reasoning}")
            logger.info(f"{'='*80}\n")
            
            # 保存追踪记录
            self.rag_cot_trace.append(trace_entry)
            
            # 保存当前追踪信息用于reasoning生成
            self.current_rag_trace = trace_entry
            
            return combined_context
            
        except Exception as e:
            logger.warning(f"CoT推理失败: {e}，仅返回知识上下文")
            trace_entry['steps'].append({
                'step_name': 'cot_reasoning',
                'error': str(e)
            })
            self.rag_cot_trace.append(trace_entry)
            return knowledge_context
    
    def doctor_response_gen(self, patient_response, dialogue_history):
        """
        生成医生的回复 V2版本 - RAG/CoT增强版本
        
        重写父类方法，在话题结束时进行共病分析（仅在评估和深入阶段）
        """
        # 先调用父类方法获取基础逻辑
        # 但我们需要在话题结束时插入共病分析
        
        if self.use_api:
            # === API模式处理逻辑 ===
            if self.dialbegin == True:
                # 对话开始：直接调用父类方法
                doctor_response, _, doctor_reasoning = super().doctor_response_gen(patient_response, dialogue_history)
                
                # 提取answer内容（去掉<answer>标签）
                doctor_response = self._extract_answer_from_content(doctor_response)
                
                # 记录reasoning（问候语，没有患者消息）
                if self.enable_reasoning and self.reasoning_generator:
                    # 获取token信息（从父类调用中无法直接获取，设为0）
                    # 检测是否是问候语
                    is_greeting = True
                    self.turn_counter += 1
                    self.reasoning_generator.add_turn_without_rag(
                        self.turn_counter,
                        "",  # 没有患者消息
                        doctor_response,
                        doctor_full_response=doctor_response,
                        tokens=0,
                        prompt_tokens=0,
                        is_greeting=is_greeting
                    )
                    # 从reasoning_generator中提取当前轮次的reasoning（如果父类没有返回）
                    if not doctor_reasoning:
                        all_turns = self.reasoning_generator.get_all_turns()
                        if all_turns:
                            last_turn = all_turns[-1]
                            if last_turn.get('role') == 'doctor':
                                doctor_reasoning = last_turn.get('think_content', '')
                
                return doctor_response, None, doctor_reasoning
            else:   
                # 对话进行中：检查当前话题是否结束
                is_topic_end, pt, ct = self.diagnosis_tree.is_topic_end(
                    self.topic_seq[self.current_idx], 
                    dialogue_history[self.topic_begin:]
                )
                super().money_cost(pt, ct)
                
                if is_topic_end:
                    # === 当前话题已结束，准备切换到下一个话题 ===
                    
                    # 在评估和深入阶段，话题结束后进行共病分析
                    completed_topic_idx = self.current_idx
                    completed_topic = self.topic_seq[completed_topic_idx]
                    
                    # 检查当前话题是否在评估或深入阶段
                    current_phase = self._get_current_phase()
                    if current_phase in [DiagnosticPhase.ASSESSMENT, DiagnosticPhase.DEEPDIVE] and self.enable_rag and self.enable_cot:
                        logger.info(f"[共病分析] 话题'{completed_topic}'结束，开始进行共病分析（阶段: {current_phase.value}）")
                        comorbidity_result = self._analyze_comorbidity_after_topic(
                            completed_topic, 
                            dialogue_history
                        )
                        # 保存共病分析结果，供下一个话题使用（使用当前话题索引作为key）
                        if comorbidity_result:
                            self.topic_comorbidity_analysis[completed_topic_idx] = comorbidity_result
                            logger.info(f"[共病分析] 共病分析结果已保存，将在下一个话题中使用")
                    
                    # 更新话题开始位置为当前对话历史长度
                    self.topic_begin = len(dialogue_history)
                    
                    # 检查是否整个诊断流程结束
                    is_dialogue_end = self.diagnosis_tree.is_end(self.topic_seq[self.current_idx])
                    if is_dialogue_end:
                        # 诊断流程结束，生成最终诊断结果
                        # 返回格式与 doctor_v2.py 一致：(diag_result, diag_reasoning)
                        diag_result, diag_reasoning = self._generate_final_diagnosis(dialogue_history)
                        
                        # 记录诊断结果到reasoning JSON（如果启用reasoning）
                        if self.enable_reasoning and self.reasoning_generator:
                            # 提取诊断内容（去除"诊断结束，你的诊断结果为："前缀）
                            diag_content = diag_result.replace("诊断结束，你的诊断结果为：", "").strip()
                            
                            # 构建诊断阶段的 input_prompt（使用 _generate_final_diagnosis 中保存的 messages）
                            # 如果 _generate_final_diagnosis 已经保存了诊断 messages，直接使用
                            if hasattr(self, '_last_diagnosis_messages') and self._last_diagnosis_messages:
                                diagnosis_input_prompt = self._last_diagnosis_messages
                            else:
                                # 如果没有保存，使用 _build_input_prompt 构建（可能不完整，但作为后备）
                                diagnosis_input_prompt, _ = self._build_input_prompt(dialogue_history, include_doctor_persona=True)
                                # 如果 _build_input_prompt 返回的是字符串，解析它
                                if isinstance(diagnosis_input_prompt, str):
                                    import json
                                    try:
                                        diagnosis_input_prompt = json.loads(diagnosis_input_prompt)
                                    except:
                                        # 如果解析失败，构建一个基本的诊断 prompt
                                        diagnosis_input_prompt = [
                                            {"role": "system", "content": self.doctor_persona},
                                            {"role": "user", "content": f"## 对话历史：\n{dialogue_history}\n\n请根据对话历史进行诊断。"}
                                        ]
                            
                            # 序列化为 JSON 字符串（与 _build_input_prompt 的格式一致）
                            import json
                            input_prompt = json.dumps(diagnosis_input_prompt, ensure_ascii=False, indent=2)
                            
                            # 计算 prompt_tokens
                            prompt_tokens_val = llm_tools_api.estimate_tokens(diagnosis_input_prompt)
                            
                            # 从对话历史中获取最后一轮患者回复（如果存在）
                            last_patient_response = ""
                            if dialogue_history and len(dialogue_history) > 0:
                                # 从后往前查找患者回复
                                for item in reversed(dialogue_history):
                                    if isinstance(item, str) and item.startswith('患者本人:'):
                                        last_patient_response = item.replace('患者本人:', '').strip()
                                        break
                            
                            # 获取诊断的token数、reasoning和完整响应（从_generate_final_diagnosis中保存的值）
                            diag_tokens = getattr(self, '_last_diagnosis_tokens', 0)
                            diag_reasoning = getattr(self, '_last_diagnosis_reasoning', None)
                            diag_full_response = getattr(self, '_last_diagnosis_full_response', diag_result)
                            
                            # 验证并清理 diag_full_response 中的患者部分
                            if diag_full_response:
                                patient_indicators = [
                                    "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                                    "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                                    "我是一名", "我是患者", "作为患者", "患者角色"
                                ]
                                is_patient_response = any(indicator in diag_full_response for indicator in patient_indicators)
                                if is_patient_response:
                                    logger.warning(f"[诊断生成] ⚠️ 检测到 diag_full_response 包含患者响应，尝试清理")
                                    # 如果 full_response 包含患者内容，尝试从其中提取医生的部分
                                    if "<think>" in diag_full_response and "<answer>" in diag_full_response:
                                        # 提取 reasoning 和 answer 部分
                                        import re
                                        reasoning_match = re.search(r'<think>(.*?)</think>', diag_full_response, re.DOTALL)
                                        answer_match = re.search(r'<answer>(.*?)</answer>', diag_full_response, re.DOTALL)
                                        
                                        if reasoning_match and answer_match:
                                            extracted_reasoning = reasoning_match.group(1).strip()
                                            extracted_answer = answer_match.group(1).strip()
                                            
                                            # 清理两部分
                                            cleaned_reasoning = self._clean_patient_content_from_reasoning(extracted_reasoning, patient_indicators) if extracted_reasoning else None
                                            cleaned_answer = self._clean_patient_content_from_reasoning(extracted_answer, patient_indicators) if extracted_answer else None
                                            
                                            # 重新组合
                                            if cleaned_reasoning and cleaned_answer:
                                                diag_full_response = f"<think>\n{cleaned_reasoning}\n</think>\n\n<answer>{cleaned_answer}</answer>"
                                                # 更新保存的 reasoning
                                                self._last_diagnosis_reasoning = cleaned_reasoning
                                                logger.info(f"[诊断生成] ✅ 已清理 diag_full_response 中的患者部分，保留医生 reasoning 和 answer")
                                            elif cleaned_answer:
                                                # reasoning 完全被清除，只保留 answer
                                                diag_full_response = f"<answer>{cleaned_answer}</answer>"
                                                self._last_diagnosis_reasoning = None
                                                logger.warning(f"[诊断生成] ⚠️ reasoning 完全包含患者内容，已清除，只保留 answer")
                                            elif cleaned_reasoning:
                                                # answer 完全被清除，只保留 reasoning（这种情况较少见）
                                                diag_full_response = f"<think>\n{cleaned_reasoning}\n</think>"
                                                self._last_diagnosis_reasoning = cleaned_reasoning
                                                logger.warning(f"[诊断生成] ⚠️ answer 完全包含患者内容，已清除，只保留 reasoning")
                                            else:
                                                # 两部分都完全被清除，使用 diag_result
                                                logger.warning(f"[诊断生成] ⚠️ reasoning 和 answer 都完全包含患者内容，使用 diag_result")
                                                diag_full_response = diag_result
                                                self._last_diagnosis_reasoning = None
                                        else:
                                            # 无法解析，使用清理后的 diag_result
                                            cleaned_diag_result = self._clean_patient_content_from_reasoning(diag_result, patient_indicators)
                                            if cleaned_diag_result:
                                                diag_full_response = cleaned_diag_result
                                                logger.info(f"[诊断生成] ✅ 已清理 diag_result 中的患者部分")
                                            else:
                                                logger.warning(f"[诊断生成] ⚠️ 无法清理，使用原始 diag_result")
                                                diag_full_response = diag_result
                                    else:
                                        # 没有标签，直接清理整个内容
                                        cleaned_full = self._clean_patient_content_from_reasoning(diag_full_response, patient_indicators)
                                        if cleaned_full:
                                            diag_full_response = cleaned_full
                                            logger.info(f"[诊断生成] ✅ 已清理 diag_full_response 中的患者部分")
                                        else:
                                            logger.warning(f"[诊断生成] ⚠️ diag_full_response 完全包含患者内容，使用 diag_result")
                                    diag_full_response = diag_result
                            
                            # 构建完整的诊断响应（包含前缀）
                            # 如果full_response已经包含<think>和<answer>标签，将前缀添加到<answer>标签内
                            if "<answer>" in diag_full_response:
                                # 提取answer内容，添加前缀，然后重新组合
                                import re
                                answer_match = re.search(r'<answer>(.*?)</answer>', diag_full_response, re.DOTALL)
                                if answer_match:
                                    answer_content = answer_match.group(1).strip()
                                    answer_with_prefix = "诊断结束，你的诊断结果为：" + answer_content
                                    diag_full_response_with_prefix = diag_full_response.replace(
                                        f"<answer>{answer_content}</answer>",
                                        f"<answer>{answer_with_prefix}</answer>"
                                    )
                                else:
                                    diag_full_response_with_prefix = "诊断结束，你的诊断结果为：" + diag_full_response
                            else:
                                # 如果没有标签，直接添加前缀
                                diag_full_response_with_prefix = "诊断结束，你的诊断结果为：" + diag_full_response
                            
                            # 记录诊断轮次
                            self.record_reasoning_turn(
                                last_patient_response,  # 最后一轮患者回复
                                diag_content,  # 诊断内容（已去除前缀）
                                thinking=diag_reasoning,  # 思考过程（从reasoning字段提取）
                                doctor_full_response=diag_full_response_with_prefix,  # 完整诊断结果（包含think和answer标签，以及前缀）
                                tokens=diag_tokens,  # 诊断的token数
                                prompt_tokens=prompt_tokens_val,
                                input_prompt=input_prompt,
                                is_diagnosis=True  # 标记为诊断
                            )
                        
                        # 返回格式与 doctor_v2.py 一致：(diag_result, None, diag_reasoning)
                        return diag_result, None, diag_reasoning
                    else:
                        # 切换到下一个话题
                        self.current_idx += 1
                        if self.topic_seq[self.current_idx] == 'parse':
                            # 处理特殊的'parse'话题
                            self._handle_parse_topic(dialogue_history)
                        
                        # 使用增强的提示词构建方法（会使用共病分析结果）
                        doctor_prompt = self._build_topic_prompt(dialogue_history, is_empathy_enabled=True, enable_reasoning=self.enable_reasoning)
                        
                        # 如果启用reasoning且没有RAG上下文，在system prompt中也添加thinking要求
                        if self.enable_reasoning and len(self.messages) > 0 and self.messages[0].get('role') == 'system':
                            # 检查是否有RAG上下文（通过检查prompt中是否包含"【诊疗指南参考"）
                            if "【诊疗指南参考" not in doctor_prompt:
                                # 没有RAG上下文，在system prompt中添加thinking要求
                                system_think_note = "\n\n【重要】在每次回复时，你必须先进行思考，将思考过程放在<think>...</think>标签中，然后将实际回复放在<answer>...</answer>标签中。\n\n【角色强调】你是医生，不是患者！你的思考过程应该从医生的角度分析患者的症状，而不是从患者的角度思考。思考过程要简洁明了，重点分析患者的具体症状和下一步问诊目的，不要使用'患者描述了相关情况'这样的通用表述。"
                                if system_think_note not in self.messages[0]['content']:
                                    self.messages[0]['content'] = self.messages[0]['content'] + system_think_note
                        
                        # 调用API生成回复
                        self.messages.append({"role": "user", "content": doctor_prompt})
                        logger.info(f"[话题切换] 切换到新话题: {self.topic_seq[self.current_idx]}")
                        
                        max_retries = 2
                        doctor_response = None
                        chat_response = None
                        
                        max_retries = 2
                        doctor_response = None
                        doctor_full_response = None
                        chat_response = None
                        
                        for retry in range(max_retries):
                            try:
                                chat_response = self.client.chat.completions.create(
                                    model=self.api_model_name,
                                    messages=self.messages,
                                    top_p=0.93,
                                    frequency_penalty=0.8,
                                    max_tokens=llm_tools_api.get_max_tokens(),
                                    **self._reasoning_kwargs()
                                )
                                
                                if chat_response and chat_response.choices and len(chat_response.choices) > 0:
                                    # 提取响应内容（包括reasoning）
                                    try:
                                        # 使用 extract_answer_and_reasoning 替代 extract_reasoning_from_response
                                        content, reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
                                        
                                        # 为了保持兼容性，获取 message 对象（日志可能会用到）
                                        message = chat_response.choices[0].message
                                        
                                        if reasoning:
                                            logger.info(f"[Doctor V2 RAG CoT] ✅ 找到reasoning字段（长度: {len(reasoning)}字符）")
                                        else:
                                            # 调试：打印响应结构以便排查问题
                                            logger.debug(f"[Doctor V2 RAG CoT] ❌ 未找到reasoning字段")
                                            logger.debug(f"[Doctor V2 RAG CoT] message type: {type(message)}")
                                            logger.debug(f"[Doctor V2 RAG CoT] message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                                            if hasattr(message, 'extra_info'):
                                                logger.debug(f"[Doctor V2 RAG CoT] message.extra_info: {message.extra_info}")
                                            if hasattr(message, 'content'):
                                                logger.debug(f"[Doctor V2 RAG CoT] message.content type: {type(message.content)}, length: {len(str(message.content)) if message.content else 0}")
                                        
                                        if content is None or content.strip() == "":
                                            if retry < max_retries - 1:
                                                logger.warning(f"[Doctor V2 RAG CoT] 警告: API返回空内容（话题切换），重试第 {retry + 1} 次...")
                                                continue
                                            else:
                                                self.messages.pop()
                                                raise Exception("Doctor响应内容为空")
                                        
                                        # 验证 reasoning 和 content 不是患者响应，并清理患者部分
                                        patient_indicators = [
                                            "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                                            "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                                            "我是一名", "我是患者", "作为患者", "患者角色"
                                        ]
                                        
                                        # 清理 reasoning，移除患者部分，只保留医生的部分
                                        if reasoning:
                                            reasoning = self._clean_patient_content_from_reasoning(reasoning, patient_indicators)
                                        
                                        is_patient_content = content and any(indicator in content for indicator in patient_indicators)
                                        
                                        if is_patient_content:
                                            logger.warning(f"[Doctor V2 RAG CoT] ⚠️ 检测到API返回的content包含患者角色内容，仅使用content")
                                        
                                        # 提取answer内容（去掉<answer>标签）
                                        doctor_response = self._extract_answer_from_content(content)
                                        
                                        # 如果有reasoning，组合成完整响应（格式：<think>reasoning</think><answer>content</answer>）
                                        if reasoning:
                                            doctor_full_response = f"<think>\n{reasoning}\n</think>\n\n<answer>{doctor_response}</answer>"
                                            logger.info(f"[Doctor V2 RAG CoT] 已组合reasoning和content为完整响应")
                                        else:
                                            doctor_full_response = doctor_response
                                        
                                        if doctor_response and doctor_response.strip():
                                            break
                                            
                                    except (AttributeError, IndexError) as e:
                                        if retry < max_retries - 1:
                                            logger.warning(f"[Doctor V2 RAG CoT] 警告: 无法提取响应内容（话题切换）: {e}，重试第 {retry + 1} 次...")
                                            continue
                                        else:
                                            self.messages.pop()
                                            raise Exception(f"Doctor API返回格式异常: {e}")
                                        
                            except Exception as e:
                                if retry < max_retries - 1:
                                    logger.warning(f"[Doctor V2 RAG CoT] 警告: API调用异常（话题切换）: {e}，重试第 {retry + 1} 次...")
                                    continue
                                else:
                                    logger.error(f"[Doctor V2 RAG CoT] 错误: API调用异常（话题切换）: {e}，已达最大重试次数")
                                    self.messages.pop()
                                    raise
                        
                        # 记录token使用量
                        if chat_response:
                            prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                                chat_response,
                                messages=self.messages,
                                response_text=doctor_response
                            )
                            super().money_cost(prompt_tokens, completion_tokens)
                        
                        # 记录reasoning
                        if self.enable_reasoning and patient_response:
                            # 提取tokens信息
                            tokens = completion_tokens if chat_response else 0
                            # 构建input_prompt（从messages中提取，去除think标签）
                            # 使用辅助函数，包含doctor_persona（确保prompt_tokens逐轮增加）
                            input_prompt, prompt_tokens_val = self._build_input_prompt(dialogue_history, include_doctor_persona=True)
                            
                            # 检测是否是诊断
                            is_diagnosis = "诊断结束" in doctor_response or "<Diagnosis>" in doctor_response
                            
                            # 获取完整响应（如果之前提取了reasoning，使用组合后的完整响应）
                            full_response_for_reasoning = doctor_full_response if doctor_full_response else doctor_response
                            
                            # 验证 full_response_for_reasoning 不是患者响应
                            if full_response_for_reasoning:
                                patient_indicators = [
                                    "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                                    "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                                    "我是一名", "我是患者", "作为患者", "患者角色"
                                ]
                                is_patient_response = any(indicator in full_response_for_reasoning for indicator in patient_indicators)
                                if is_patient_response:
                                    logger.warning(f"[Doctor V2 RAG CoT] ⚠️ 检测到 full_response_for_reasoning 包含患者响应，使用 doctor_response 代替")
                                    full_response_for_reasoning = doctor_response
                            
                            self.record_reasoning_turn(
                                patient_response, 
                                doctor_response,
                                doctor_full_response=full_response_for_reasoning,
                                tokens=tokens,
                                prompt_tokens=prompt_tokens_val,
                                input_prompt=input_prompt,
                                is_diagnosis=is_diagnosis
                            )
                        
                        # 从reasoning_generator中提取当前轮次的reasoning
                        current_reasoning = None
                        if self.enable_reasoning and self.reasoning_generator:
                            all_turns = self.reasoning_generator.get_all_turns()
                            if all_turns:
                                # 获取最后一个轮次（当前轮次）的think_content
                                last_turn = all_turns[-1]
                                if last_turn.get('role') == 'doctor':
                                    current_reasoning = last_turn.get('think_content', '')
                        
                        # 清理临时消息并返回结果
                        self.messages.pop()
                        return doctor_response, self.topic_seq[self.current_idx], current_reasoning
                else:
                    # === 当前话题未结束，继续围绕当前话题进行问诊 ===
                    if self.topic_seq[self.current_idx] == 'parse':
                        self._handle_parse_topic(dialogue_history)
                    
                    # 使用增强的提示词构建方法
                    doctor_prompt = self._build_continuation_prompt(dialogue_history, enable_reasoning=self.enable_reasoning)
                    
                    # 如果启用reasoning且没有RAG上下文，在system prompt中也添加thinking要求
                    if self.enable_reasoning and len(self.messages) > 0 and self.messages[0].get('role') == 'system':
                        # 检查是否有RAG上下文（通过检查prompt中是否包含"【诊疗指南参考"）
                        if "【诊疗指南参考" not in doctor_prompt:
                            # 没有RAG上下文，在system prompt中添加thinking要求
                            system_think_note = "\n\n【重要】在每次回复时，你必须先进行思考，将思考过程放在<think>...</think>标签中，然后将实际回复放在<answer>...</answer>标签中。思考过程要简洁明了，重点分析患者的具体症状和下一步问诊目的，不要使用'患者描述了相关情况'这样的通用表述。"
                            if system_think_note not in self.messages[0]['content']:
                                self.messages[0]['content'] = self.messages[0]['content'] + system_think_note
                    
                    self.messages.append({"role": "user", "content": doctor_prompt})
                    
                    max_retries = 2
                    doctor_response = None
                    doctor_full_response = None
                    chat_response = None
                    
                    for retry in range(max_retries):
                        try:
                            chat_response = self.client.chat.completions.create(
                                model=self.api_model_name,
                                messages=self.messages,
                                top_p=0.93,
                                frequency_penalty=0.8,
                                max_tokens=llm_tools_api.get_max_tokens()
                            )
                            
                            if chat_response and chat_response.choices and len(chat_response.choices) > 0:
                                # 提取响应内容（包括reasoning）
                                try:
                                    # 使用 extract_answer_and_reasoning 替代 extract_reasoning_from_response
                                    content, reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
                                    
                                    # 为了保持兼容性，获取 message 对象
                                    message = chat_response.choices[0].message
                                    
                                    if reasoning:
                                        logger.info(f"[Doctor V2 RAG CoT] ✅ 找到reasoning字段（长度: {len(reasoning)}字符）")
                                    
                                    if content is None or content.strip() == "":
                                        if retry < max_retries - 1:
                                            logger.warning(f"[Doctor V2 RAG CoT] 警告: API返回空内容（继续话题），重试第 {retry + 1} 次...")
                                            continue
                                        else:
                                            self.messages.pop()
                                            raise Exception("Doctor响应内容为空")
                                    
                                    # 验证 reasoning 和 content 不是患者响应，并清理患者部分
                                    patient_indicators = [
                                        "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                                        "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                                        "我是一名", "我是患者", "作为患者", "患者角色"
                                    ]
                                    
                                    # 清理 reasoning，移除患者部分，只保留医生的部分
                                    if reasoning:
                                        reasoning = self._clean_patient_content_from_reasoning(reasoning, patient_indicators)
                                    
                                    is_patient_content = content and any(indicator in content for indicator in patient_indicators)
                                    
                                    if is_patient_content:
                                        logger.warning(f"[Doctor V2 RAG CoT] ⚠️ 检测到API返回的content包含患者角色内容，仅使用content")
                                    
                                    # 提取answer内容（去掉<answer>标签）
                                    doctor_response = self._extract_answer_from_content(content)
                                    
                                    # 如果有reasoning，组合成完整响应（格式：<think>reasoning</think><answer>content</answer>）
                                    if reasoning:
                                        doctor_full_response = f"<think>\n{reasoning}\n</think>\n\n<answer>{doctor_response}</answer>"
                                        logger.info(f"[Doctor V2 RAG CoT] 已组合reasoning和content为完整响应")
                                    else:
                                        doctor_full_response = doctor_response
                                    
                                    if doctor_response and doctor_response.strip():
                                        break
                                        
                                except (AttributeError, IndexError) as e:
                                    if retry < max_retries - 1:
                                        logger.warning(f"[Doctor V2 RAG CoT] 警告: 无法提取响应内容（继续话题）: {e}，重试第 {retry + 1} 次...")
                                        continue
                                    else:
                                        self.messages.pop()
                                        raise Exception(f"Doctor API返回格式异常: {e}")
                                    
                        except Exception as e:
                            if retry < max_retries - 1:
                                logger.warning(f"[Doctor V2 RAG CoT] 警告: API调用异常（继续话题）: {e}，重试第 {retry + 1} 次...")
                                continue
                            else:
                                logger.error(f"[Doctor V2 RAG CoT] 错误: API调用异常（继续话题）: {e}，已达最大重试次数")
                                self.messages.pop()
                                raise
                    
                    # 记录token使用量
                    if chat_response:
                        prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                            chat_response,
                            messages=self.messages,
                            response_text=doctor_response
                        )
                        super().money_cost(prompt_tokens, completion_tokens)
                    
                    # 记录reasoning
                    if self.enable_reasoning and patient_response:
                        # 提取tokens信息
                        tokens = completion_tokens if chat_response else 0
                        # 构建input_prompt（从messages中提取，去除think标签）
                        # 使用辅助函数，包含doctor_persona（确保prompt_tokens逐轮增加）
                        input_prompt, prompt_tokens_val = self._build_input_prompt(dialogue_history, include_doctor_persona=True)
                        
                        # 检测是否是诊断
                        is_diagnosis = "诊断结束" in doctor_response or "<Diagnosis>" in doctor_response
                        
                        # 获取完整响应（如果之前提取了reasoning，使用组合后的完整响应）
                        full_response_for_reasoning = doctor_full_response if doctor_full_response else doctor_response
                        
                        # 验证 full_response_for_reasoning 不是患者响应
                        if full_response_for_reasoning:
                            patient_indicators = [
                                "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                                "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                                "我是一名", "我是患者", "作为患者", "患者角色"
                            ]
                            is_patient_response = any(indicator in full_response_for_reasoning for indicator in patient_indicators)
                            if is_patient_response:
                                logger.warning(f"[Doctor V2 RAG CoT] ⚠️ 检测到 full_response_for_reasoning 包含患者响应，使用 doctor_response 代替")
                                full_response_for_reasoning = doctor_response
                        
                        self.record_reasoning_turn(
                            patient_response, 
                            doctor_response,
                            doctor_full_response=full_response_for_reasoning,
                            tokens=tokens,
                            prompt_tokens=prompt_tokens_val,
                            input_prompt=input_prompt,
                            is_diagnosis=is_diagnosis
                        )
                    
                    # 从reasoning_generator中提取当前轮次的reasoning
                    current_reasoning = None
                    if self.enable_reasoning and self.reasoning_generator:
                        all_turns = self.reasoning_generator.get_all_turns()
                        if all_turns:
                            # 获取最后一个轮次（当前轮次）的think_content
                            last_turn = all_turns[-1]
                            if last_turn.get('role') == 'doctor':
                                current_reasoning = last_turn.get('think_content', '')
                    
                    # 清理临时消息并返回结果
                    self.messages.pop()
                    return doctor_response, self.topic_seq[self.current_idx], current_reasoning
        else:
            # 本地模型模式：类似逻辑，但使用本地模型
            if self.dialbegin == True:
                doctor_response, _, doctor_reasoning = super().doctor_response_gen(patient_response, dialogue_history)
                # 提取answer内容（去掉<answer>标签）
                doctor_response = self._extract_answer_from_content(doctor_response)
                return doctor_response, None, doctor_reasoning
            else:
                # 对话进行中
                is_topic_end, pt, ct = self.diagnosis_tree.is_topic_end(
                    self.topic_seq[self.current_idx], 
                    dialogue_history[self.topic_begin:]
                )
                super().money_cost(pt, ct)
                
                if is_topic_end:
                    # 话题结束：进行共病分析
                    completed_topic_idx = self.current_idx
                    completed_topic = self.topic_seq[completed_topic_idx]
                    
                    # 检查当前话题是否在评估或深入阶段
                    current_phase = self._get_current_phase()
                    if current_phase in [DiagnosticPhase.ASSESSMENT, DiagnosticPhase.DEEPDIVE] and self.enable_rag and self.enable_cot:
                        logger.info(f"[共病分析] 话题'{completed_topic}'结束，开始进行共病分析（阶段: {current_phase.value}）")
                        comorbidity_result = self._analyze_comorbidity_after_topic(
                            completed_topic, 
                            dialogue_history
                        )
                        if comorbidity_result:
                            self.topic_comorbidity_analysis[completed_topic_idx] = comorbidity_result
                            logger.info(f"[共病分析] 共病分析结果已保存，将在下一个话题中使用")
                    
                    self.topic_begin = len(dialogue_history)
                    is_dialogue_end = self.diagnosis_tree.is_end(self.topic_seq[self.current_idx])
                    if is_dialogue_end:
                        # 返回格式与 doctor_v2.py 一致：(diag_result, diag_reasoning)
                        diag_result, diag_reasoning = self._generate_final_diagnosis(dialogue_history)
                        return diag_result, None, diag_reasoning
                    else:
                        self.current_idx += 1
                        if self.topic_seq[self.current_idx] == 'parse':
                            self._handle_parse_topic(dialogue_history)
                        
                        doctor_prompt = self._build_topic_prompt(dialogue_history, is_empathy_enabled=True, enable_reasoning=self.enable_reasoning)
                        self.messages.append({"role": "user", "content": doctor_prompt})
                        
                        # 使用本地模型生成
                        text = self.doctor_tokenizer.apply_chat_template(
                            self.messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                        outputs = self.doctor_model.generate(
                            inputs.input_ids,
                            max_new_tokens=512,
                            top_p=0.93,
                            frequency_penalty=0.8
                        )
                        generated = outputs[0][inputs.input_ids.shape[1]:]
                        raw_response = self.doctor_tokenizer.decode(generated, skip_special_tokens=True)
                        # 提取answer内容（去掉<answer>标签）
                        doctor_response = self._extract_answer_from_content(raw_response)
                        
                        self.messages.pop()
                        return doctor_response, self.topic_seq[self.current_idx], None
                else:
                    # 话题继续
                    if self.topic_seq[self.current_idx] == 'parse':
                        self._handle_parse_topic(dialogue_history)
                    
                    doctor_prompt = self._build_continuation_prompt(dialogue_history)
                    self.messages.append({"role": "user", "content": doctor_prompt})
                    
                    text = self.doctor_tokenizer.apply_chat_template(
                        self.messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                    outputs = self.doctor_model.generate(
                        inputs.input_ids,
                        max_new_tokens=512,
                        top_p=0.93,
                        frequency_penalty=0.8
                    )
                    generated = outputs[0][inputs.input_ids.shape[1]:]
                    raw_response = self.doctor_tokenizer.decode(generated, skip_special_tokens=True)
                    # 提取answer内容（去掉<answer>标签）
                    doctor_response = self._extract_answer_from_content(raw_response)
                    
                    # 记录reasoning（本地模型模式）
                    if self.enable_reasoning and patient_response:
                        # 本地模型模式下无法直接获取tokens，设为0
                        # 构建input_prompt（从messages中提取，去除think标签）
                        input_prompt = None
                        if self.messages:
                            try:
                                # 清理messages，去除assistant消息中的think标签
                                cleaned_messages = []
                                for msg in self.messages:
                                    cleaned_msg = msg.copy()
                                    if msg.get('role') == 'assistant' and 'content' in msg:
                                        content = msg['content']
                                        # 提取answer部分（去除think标签）
                                        if '<answer>' in content:
                                            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                                            if answer_match:
                                                cleaned_msg['content'] = answer_match.group(1).strip()
                                        elif '<think>' in content:
                                            # 如果没有answer标签，去除think标签
                                            cleaned_msg['content'] = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                                    cleaned_messages.append(cleaned_msg)
                                input_prompt = json.dumps(cleaned_messages, ensure_ascii=False, indent=2)
                            except:
                                pass
                        
                        # 检测是否是诊断
                        is_diagnosis = "诊断结束" in doctor_response or "<Diagnosis>" in doctor_response
                        
                        self.record_reasoning_turn(
                            patient_response, 
                            doctor_response,
                            doctor_full_response=doctor_response,
                            tokens=0,  # 本地模型无法获取
                            prompt_tokens=0,  # 本地模型无法获取
                            input_prompt=input_prompt,
                            is_diagnosis=is_diagnosis
                        )
                    
                    # 从reasoning_generator中提取当前轮次的reasoning
                    current_reasoning = None
                    if self.enable_reasoning and self.reasoning_generator:
                        all_turns = self.reasoning_generator.get_all_turns()
                        if all_turns:
                            # 获取最后一个轮次（当前轮次）的think_content
                            last_turn = all_turns[-1]
                            if last_turn.get('role') == 'doctor':
                                current_reasoning = last_turn.get('think_content', '')
                    
                    self.messages.pop()
                    return doctor_response, self.topic_seq[self.current_idx], current_reasoning
    
    def _build_topic_prompt(self, dialogue_history, is_empathy_enabled=True, enable_reasoning=True):
        """
        构建话题切换时的提示词（RAG增强版本 + CoT + 共病分析）
        """
        # 获取当前话题
        current_topic = self.topic_seq[self.current_idx] if self.current_idx < len(self.topic_seq) else "患者的精神状况"
        
        # 获取上一个话题的共病分析结果（如果存在）
        previous_topic_idx = self.current_idx - 1
        previous_comorbidity = self.topic_comorbidity_analysis.get(previous_topic_idx, {})
        
        # 只在评估阶段和深入阶段启用RAG和CoT
        rag_context = ""
        if self._should_use_rag_cot() and self.current_idx < len(self.topic_seq):
            current_phase = self._get_current_phase()
            logger.info(f"[RAG+CoT] 阶段: {current_phase.value if current_phase else 'Unknown'}，开始为话题'{current_topic}'检索知识和推理（对话轮数: {len(dialogue_history)}）")
            
            # 如果有上一个话题的共病分析结果，先使用它
            if previous_comorbidity:
                logger.info(f"[RAG+CoT] 使用上一个话题的共病分析结果来指导当前话题")
                # 将共病分析结果作为上下文
                rag_context = self._cot_reasoning_with_rag(
                    current_topic, 
                    dialogue_history,
                    previous_comorbidity=previous_comorbidity
                )
            else:
                # 没有共病分析结果，进行常规的RAG/CoT推理
                rag_context = self._cot_reasoning_with_rag(current_topic, dialogue_history)
            
            if rag_context:
                logger.info(f"[RAG+CoT] 话题'{current_topic}'已增强，RAG上下文长度: {len(rag_context)} 字符")
        
        # 构建基础提示词
        empathy_text = ""
        if hasattr(self, 'doctor_prompt') and self.doctor_prompt and self.doctor_prompt.get('empathy') == '有':
            empathy_text = "在适当的时候提供与患者的共情"
        else:
            empathy_text = "简洁的生成"
        
        # 检查对话历史中是否有重复的问题
        recent_doctor_questions = []
        for item in dialogue_history[-10:]:  # 检查最近10轮对话
            if isinstance(item, str) and item.startswith('医生:'):
                question = item.replace('医生:', '').strip()
                if question:
                    recent_doctor_questions.append(question)
        
        # 构建防重复提示
        anti_repeat_instruction = ""
        if len(recent_doctor_questions) >= 2:
            # 如果最近有多个问题，检查是否有重复
            unique_questions = set(recent_doctor_questions)
            if len(unique_questions) < len(recent_doctor_questions):
                anti_repeat_instruction = "\n\n【重要】检查对话历史，如果患者已经回答过某个问题，绝对不要再问相同或类似的问题！必须基于患者已经回答的内容，提出新的、不同的问题。如果当前话题的信息已经足够，可以询问该话题的其他方面或切换到下一个话题。"
        
        base_prompt = self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history) + \
            "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(current_topic) + \
            f"\n3.你每次只能围绕1个话题询问。使用口语化的表达，{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头或相同的问题。{anti_repeat_instruction}\n输出有标点符号的一段文字，不要换行。"
        
        # 如果启用reasoning且没有RAG上下文（非RAG/CoT阶段），添加thinking要求
        thinking_instruction = ""
        if enable_reasoning and not rag_context:
            thinking_instruction = """\n\n【必须严格遵守】在生成回复前，你必须先进行思考，将思考过程放在<think>...</think>标签中，然后将实际回复放在<answer>...</answer>标签中。\n\n【重要】你的回复必须包含<think>和<answer>两个标签，缺一不可！\n\n【角色强调】你是医生，不是患者！你的思考过程必须从医生的角度分析患者的症状，绝对不要从患者的角度思考（例如"用户希望我扮演一位患者"这样的表述是错误的）。\n\n思考过程要求（必须详细具体，不能使用通用表述）：\n1) 对患者当前描述的具体理解：要具体说明患者提到了什么症状、情绪、行为等，不要使用'患者描述了相关情况'这样的通用表述。必须具体列出患者提到的症状关键词，如：情绪低落、失眠、注意力不集中等。\n2) 基于对话历史的分析：结合之前的对话，分析患者症状的变化、特点、可能的关联等。要说明症状的持续时间、严重程度变化等。\n3) 下一步问诊的策略和目的：明确说明为什么要问这个问题，想了解什么信息，这个问题如何帮助诊断。要说明这个问题能帮助区分哪些疾病或判断什么。\n\n【格式要求】你的回复必须严格按照以下格式：\n<think>\n[你的思考过程，必须详细具体，不能使用通用表述，必须从医生的角度思考]\n</think>\n\n<answer>\n[你的实际回复]\n</answer>\n\n示例（患者对话：最近一个月情绪特别低落，做什么都没兴趣，脑子里总是空空的。晚上经常到凌晨才能睡着。但脑子还在转那些负面的想法，觉得自己很没用，对未来也没希望。这种状态已经影响到学习了，明明很累却总觉得脑子像一团浆糊注意力完全集中不了。）：\n<think>\n患者表现出明显的抑郁症和睡眠障碍的核心症状，包括情绪低落、兴趣丧失、失眠、负面认知、注意力不集中等。这些症状已经持续一个月并严重影响到他的学习和日常生活。这是初次问诊，患者已经详细描述了当前的痛苦。根据指示，我需要询问患者的"既往求助经历"。这有助于了解患者对自身问题的认识程度、过往的应对方式、是否有接受过其他治疗或诊断，以及他是否曾向他人寻求帮助。这对于全面评估患者的情况非常重要。\n</think>\n\n<answer>之前有没有去过心理科咨询过或者吃过什么药？</answer>"""
            base_prompt = base_prompt + thinking_instruction + "\n\n【再次强调】你必须使用<think>和<answer>标签格式回复，不能直接输出答案！"
        
        # 如果有RAG上下文，作为额外的参考信息添加到提示词中（不取代原有逻辑）
        final_prompt = base_prompt
        if rag_context:
            # 从rag_context中提取疾病名称（如果包含候选疾病信息）
            disease_names_text = ""
            if "【3个候选疾病" in rag_context or "候选疾病" in rag_context:
                # 尝试从rag_context中提取疾病名称
                import re
                disease_section_match = re.search(r'【3个候选疾病[^】]*】\s*\n(.*?)(?=\n\n|$)', rag_context, re.DOTALL)
                if disease_section_match:
                    disease_section = disease_section_match.group(1).strip()
                    # 提取每行的疾病名称
                    disease_names = []
                    for line in disease_section.split('\n'):
                        line = line.strip()
                        if line:
                            if '（' in line:
                                disease_name = line.split('（')[0].strip()
                            elif '：' in line:
                                disease_name = line.split('：')[0].strip()
                            else:
                                disease_name = line.split(':')[0].strip() if ':' in line else line
                            if disease_name and len(disease_name) < 50:  # 过滤掉太长的行
                                disease_names.append(disease_name)
                    if len(disease_names) >= 3:
                        disease_names_text = f"（{disease_names[0]}、{disease_names[1]}、{disease_names[2]}）"
            
            final_prompt = f"""【诊疗指南参考和推理过程】
{rag_context}

【医生问诊要求】
{base_prompt}

【重要提示】
上述诊疗指南参考和推理过程包含了：
1. 3个候选疾病及其可能性排序
2. 这3个疾病在{current_topic}方面的症状差异分析
3. 如何通过询问{current_topic}来区分这3个疾病的思考过程

你的任务是：
- 基于上述疾病分析和症状差异，设计有针对性的问题来区分这3个候选疾病{disease_names_text}
- 你的提问应该能够帮助判断患者更可能是哪个疾病
- 提问要自然、口语化，但要有明确的区分目的
- 仍然要基于对话历史和当前话题，不要偏离主题"""
        
        # 记录最终生成的提示词（用于追踪）
        if self.rag_cot_trace and len(self.rag_cot_trace) > 0:
            last_trace = self.rag_cot_trace[-1]
            if last_trace.get('stage') == 'consultation_cot' and last_trace.get('current_topic') == current_topic:
                last_trace['final_prompt'] = final_prompt
                last_trace['base_prompt'] = base_prompt
        
        return final_prompt
    
    def _build_continuation_prompt(self, dialogue_history, enable_reasoning=False):
        """
        构建话题继续时的提示词（RAG增强版本 + CoT + 共病分析）
        """
        # 获取当前话题
        current_topic = self.topic_seq[self.current_idx] if self.current_idx < len(self.topic_seq) else "患者的精神状况"
        
        # 获取上一个话题的共病分析结果（如果存在）
        previous_topic_idx = self.current_idx - 1
        previous_comorbidity = self.topic_comorbidity_analysis.get(previous_topic_idx, {})
        
        # 只在评估阶段和深入阶段启用RAG和CoT
        rag_context = ""
        if self._should_use_rag_cot() and self.current_idx < len(self.topic_seq):
            current_phase = self._get_current_phase()
            logger.info(f"[RAG+CoT] 阶段: {current_phase.value if current_phase else 'Unknown'}，继续话题'{current_topic}'，检索知识和推理（对话轮数: {len(dialogue_history)}）")
            
            # 如果有上一个话题的共病分析结果，使用它
            if previous_comorbidity:
                logger.info(f"[RAG+CoT] 使用上一个话题的共病分析结果来指导当前话题的继续")
                rag_context = self._cot_reasoning_with_rag(
                    current_topic, 
                    dialogue_history,
                    previous_comorbidity=previous_comorbidity
                )
            else:
                rag_context = self._cot_reasoning_with_rag(current_topic, dialogue_history)
            
            if rag_context:
                logger.info(f"[RAG+CoT] 话题'{current_topic}'已增强，RAG上下文长度: {len(rag_context)} 字符")
        
        # 构建基础提示词
        empathy_text = ""
        if hasattr(self, 'doctor_prompt') and self.doctor_prompt and self.doctor_prompt.get('empathy') == '有':
            empathy_text = "在适当的时候提供与患者的共情"
        else:
            empathy_text = "简洁的，口语化的表达进行文本生成"
        
        # 检查对话历史中是否有重复的问题
        recent_doctor_questions = []
        for item in dialogue_history[-10:]:  # 检查最近10轮对话
            if isinstance(item, str) and item.startswith('医生:'):
                question = item.replace('医生:', '').strip()
                if question:
                    recent_doctor_questions.append(question)
        
        # 构建防重复提示
        anti_repeat_instruction = ""
        if len(recent_doctor_questions) >= 2:
            # 如果最近有多个问题，检查是否有重复
            unique_questions = set(recent_doctor_questions)
            if len(unique_questions) < len(recent_doctor_questions):
                anti_repeat_instruction = "\n\n【重要】检查对话历史，如果患者已经回答过某个问题，绝对不要再问相同或类似的问题！必须基于患者已经回答的内容，提出新的、不同的问题。如果当前话题的信息已经足够，可以询问该话题的其他方面或切换到下一个话题。"
        
        base_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history) + \
            "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。。".format(current_topic) + \
            f"3.\n你每次只能围绕1个话题询问。使用{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头或相同的问题。{anti_repeat_instruction}\n输出有标点符号的一段文字，不要换行。"
        
        # 如果启用reasoning且没有RAG上下文（非RAG/CoT阶段），添加thinking要求
        thinking_instruction = ""
        if enable_reasoning and not rag_context:
            thinking_instruction = """\n\n【必须严格遵守】在生成回复前，你必须先进行思考，将思考过程放在<think>...</think>标签中，然后将实际回复放在<answer>...</answer>标签中。\n\n【重要】你的回复必须包含<think>和<answer>两个标签，缺一不可！\n\n【角色强调】你是医生，不是患者！你的思考过程必须从医生的角度分析患者的症状，绝对不要从患者的角度思考（例如"用户希望我扮演一位患者"这样的表述是错误的）。\n\n思考过程要求（必须详细具体，不能使用通用表述）：\n1) 对患者当前描述的具体理解：要具体说明患者提到了什么症状、情绪、行为等，不要使用'患者描述了相关情况'这样的通用表述。必须具体列出患者提到的症状关键词，如：情绪低落、失眠、注意力不集中等。\n2) 基于对话历史的分析：结合之前的对话，分析患者症状的变化、特点、可能的关联等。要说明症状的持续时间、严重程度变化等。\n3) 下一步问诊的策略和目的：明确说明为什么要问这个问题，想了解什么信息，这个问题如何帮助诊断。要说明这个问题能帮助区分哪些疾病或判断什么。\n\n【格式要求】你的回复必须严格按照以下格式：\n<think>\n[你的思考过程，必须详细具体，不能使用通用表述，必须从医生的角度思考]\n</think>\n\n<answer>\n[你的实际回复]\n</answer>\n\n示例（患者对话：最近一个月情绪特别低落，做什么都没兴趣，脑子里总是空空的。晚上经常到凌晨才能睡着。但脑子还在转那些负面的想法，觉得自己很没用，对未来也没希望。这种状态已经影响到学习了，明明很累却总觉得脑子像一团浆糊注意力完全集中不了。）：\n<think>\n患者表现出明显的抑郁症和睡眠障碍的核心症状，包括情绪低落、兴趣丧失、失眠、负面认知、注意力不集中等。这些症状已经持续一个月并严重影响到他的学习和日常生活。这是初次问诊，患者已经详细描述了当前的痛苦。根据指示，我需要询问患者的"既往求助经历"。这有助于了解患者对自身问题的认识程度、过往的应对方式、是否有接受过其他治疗或诊断，以及他是否曾向他人寻求帮助。这对于全面评估患者的情况非常重要。\n</think>\n\n<answer>之前有没有去过心理科咨询过或者吃过什么药？</answer>"""
            base_prompt = base_prompt + thinking_instruction + "\n\n【再次强调】你必须使用<think>和<answer>标签格式回复，不能直接输出答案！"
        
        # 如果有RAG上下文，作为额外的参考信息添加到提示词中（不取代原有逻辑）
        final_prompt = base_prompt
        if rag_context:
            # 从rag_context中提取疾病名称（如果包含候选疾病信息）
            disease_names_text = ""
            if "【3个候选疾病" in rag_context or "候选疾病" in rag_context:
                # 尝试从rag_context中提取疾病名称
                import re
                disease_section_match = re.search(r'【3个候选疾病[^】]*】\s*\n(.*?)(?=\n\n|$)', rag_context, re.DOTALL)
                if disease_section_match:
                    disease_section = disease_section_match.group(1).strip()
                    # 提取每行的疾病名称
                    disease_names = []
                    for line in disease_section.split('\n'):
                        line = line.strip()
                        if line:
                            if '（' in line:
                                disease_name = line.split('（')[0].strip()
                            elif '：' in line:
                                disease_name = line.split('：')[0].strip()
                            else:
                                disease_name = line.split(':')[0].strip() if ':' in line else line
                            if disease_name and len(disease_name) < 50:  # 过滤掉太长的行
                                disease_names.append(disease_name)
                    if len(disease_names) >= 3:
                        disease_names_text = f"（{disease_names[0]}、{disease_names[1]}、{disease_names[2]}）"
            
            final_prompt = f"""【诊疗指南参考和推理过程】
{rag_context}

【医生问诊要求】
{base_prompt}

【重要提示】
上述诊疗指南参考和推理过程包含了：
1. 3个候选疾病及其可能性排序
2. 这3个疾病在{current_topic}方面的症状差异分析
3. 如何通过询问{current_topic}来区分这3个疾病的思考过程

你的任务是：
- 基于上述疾病分析和症状差异，设计有针对性的问题来区分这3个候选疾病{disease_names_text}
- 你的提问应该能够帮助判断患者更可能是哪个疾病
- 提问要自然、口语化，但要有明确的区分目的
- 仍然要基于对话历史和当前话题，不要偏离主题"""
        
        # 记录最终生成的提示词（用于追踪）
        if self.rag_cot_trace and len(self.rag_cot_trace) > 0:
            last_trace = self.rag_cot_trace[-1]
            if last_trace.get('stage') == 'consultation_cot' and last_trace.get('current_topic') == current_topic:
                last_trace['final_prompt'] = final_prompt
                last_trace['base_prompt'] = base_prompt
        
        return final_prompt
    
    
    def _generate_final_diagnosis(self, dialogue_history):
        """
        生成最终诊断结果（RAG增强版本）
        在诊断时进行RAG检索
        """
        # 创建诊断阶段追踪记录
        diagnosis_trace_entry = {
            'stage': 'diagnosis',
            'dialogue_history_length': len(dialogue_history),
            'steps': []
        }
        
        # 进行RAG检索
        rag_context = ""
        if self.enable_rag:
            try:
                # 基于对话历史进行RAG检索
                query = f"根据以下对话历史，检索相关的诊疗指南知识用于最终诊断：\n{dialogue_history}"
                search_info = self._search_knowledge(query, top_k=RAGConfig.TOP_K_RESULTS)
                
                if search_info and search_info.get('filtered_results'):
                    search_results = search_info['filtered_results']
                    knowledge_context = self._format_knowledge_context(search_results)
                    rag_context = f"{knowledge_context}\n"
                    
                    # 记录RAG检索步骤
                    diagnosis_trace_entry['steps'].append({
                        'step_name': 'rag_retrieval',
                        'query': query,
                        'num_initial_results': search_info['num_initial_results'],
                        'num_final_results': search_info['num_final_results'],
                        'used_reranking': search_info['used_reranking'],
                        'knowledge_context': knowledge_context
                    })
                    diagnosis_trace_entry['knowledge_context'] = knowledge_context
                    logger.info(f"[最终诊断] RAG检索完成，检索到 {search_info['num_final_results']} 条相关知识")
                else:
                    logger.warning(f"[最终诊断] RAG检索未找到相关知识")
                    diagnosis_trace_entry['steps'].append({
                        'step_name': 'rag_retrieval',
                        'query': query,
                        'error': 'No results found'
                    })
            except Exception as e:
                logger.warning(f"[最终诊断] RAG检索失败: {e}")
                diagnosis_trace_entry['steps'].append({
                    'step_name': 'rag_retrieval',
                    'error': str(e)
                })
        
        # 保存诊断阶段追踪记录
        self.rag_cot_trace.append(diagnosis_trace_entry)
        
        # 使用父类的诊断生成逻辑，但添加RAG上下文
        diagnosis_prompt = (
f"你是一位经验丰富的精神科医生。请阅读以下初次精神科门诊的问诊对话记录，并根据ICD-10国际疾病分类标准，仔细分析后输出患者诊断结束后的ICD-10诊断代码。\n\n"
        )
        
        # 如果有RAG上下文，添加到提示词中
        if rag_context:
            diagnosis_prompt += f"{rag_context}\n"
        
        # 清理对话历史中的患者 think 内容，只保留 <answer> 标签内的内容
        cleaned_dialogue_history_for_diagnosis = []
        for hist_item in dialogue_history:
            if isinstance(hist_item, str):
                # 清理患者回复中的think和answer标签
                if hist_item.startswith('患者本人:') or hist_item.startswith('患者:'):
                    patient_prefix = '患者本人: ' if hist_item.startswith('患者本人:') else '患者: '
                    # 先尝试提取answer内容（如果有）
                    answer_match = re.search(r'<answer>(.*?)</answer>', hist_item, re.DOTALL | re.IGNORECASE)
                    if answer_match:
                        cleaned_item = patient_prefix + answer_match.group(1).strip()
                    else:
                        # 如果没有answer标签，去除think标签，保留实际回复内容
                        cleaned_item = re.sub(r'<think>.*?</think>', '', hist_item, flags=re.DOTALL | re.IGNORECASE)
                        cleaned_item = re.sub(r'<answer>.*?</answer>', '', cleaned_item, flags=re.DOTALL | re.IGNORECASE)
                        cleaned_item = re.sub(r'</?think>', '', cleaned_item, flags=re.IGNORECASE)
                        cleaned_item = re.sub(r'</?answer>', '', cleaned_item, flags=re.IGNORECASE)
                        cleaned_item = cleaned_item.strip()
                        # 确保保留患者前缀
                        if not cleaned_item.startswith(patient_prefix):
                            content_without_prefix = cleaned_item.replace(patient_prefix, '').strip()
                            cleaned_item = patient_prefix + content_without_prefix
                    
                    # 去除患者回复中的换行符（\n 和实际的换行）
                    if cleaned_item.startswith(patient_prefix):
                        content_part = cleaned_item[len(patient_prefix):]
                        # 去除所有换行符（包括 \n 和实际的换行）
                        content_part = content_part.replace('\n', ' ').replace('\\n', ' ')
                        # 去除多余的空格
                        content_part = re.sub(r'\s+', ' ', content_part).strip()
                        cleaned_item = patient_prefix + content_part
                    
                    cleaned_dialogue_history_for_diagnosis.append(cleaned_item)
                elif hist_item.startswith('医生:'):
                    # 清理医生回复中的think和answer标签
                    answer_match = re.search(r'<answer>(.*?)</answer>', hist_item, re.DOTALL | re.IGNORECASE)
                    if answer_match:
                        cleaned_item = '医生: ' + answer_match.group(1).strip()
                    else:
                        cleaned_item = re.sub(r'<think>.*?</think>', '', hist_item, flags=re.DOTALL | re.IGNORECASE).strip()
                        cleaned_item = re.sub(r'<answer>.*?</answer>', '', cleaned_item, flags=re.DOTALL | re.IGNORECASE).strip()
                    cleaned_dialogue_history_for_diagnosis.append(cleaned_item)
                else:
                    cleaned_dialogue_history_for_diagnosis.append(hist_item)
            else:
                cleaned_dialogue_history_for_diagnosis.append(hist_item)
        
        # 继续原有的诊断提示词...
        diagnosis_prompt += (
"## 疾病分类说明\n"
"请仅从以下ICD-10标准中的10种疾病中选择最符合的诊断大类以及进一步细分的小类：\n"
"    - F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。\n"
"        F32.0 轻度抑郁发作：症状轻，社会功能影响有限。\n"
"        F32.1 中度抑郁发作：症状更明显，日常活动受限。\n"
"        F32.2 重度抑郁发作，无精神病性症状：症状显著，丧失功能，但无妄想/幻觉。\n"
"        F32.3 重度抑郁发作，有精神病性症状：伴有抑郁性妄想、幻觉或木僵。\n"
"        F32.8 其他抑郁发作；F32.9 抑郁发作，未特指。\n"
"    - F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。\n"
"        F41.0 惊恐障碍：突发的强烈恐慌发作，常伴濒死感。\n"
"        F41.1 广泛性焦虑障碍：长期持续的过度担忧和紧张不安。\n"
"        F41.2 混合性焦虑与抑郁障碍：焦虑与抑郁并存但均不足以单独诊断。\n"
"        F41.3 其他混合性焦虑障碍：混合焦虑表现但未完全符合特定标准。\n"
"        F41.9 焦虑障碍，未特指：存在焦虑症状但资料不足以分类。\n"
"    - F39.9 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。\n"
"    - F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。\n"
"        F51.0 非器质性失眠：入睡困难、易醒或睡眠不恢复精力。\n"
"        F51.1 非器质性嗜睡：过度睡眠或难以保持清醒。\n"
"        F51.2 非器质性睡眠-觉醒节律障碍：昼夜节律紊乱导致睡眠异常。\n"
"        F51.3 梦魇障碍：频繁恶梦导致醒后强烈不安。\n"
"        F51.4 睡眠惊恐（夜惊）：夜间突然惊恐醒来伴强烈焦虑反应。\n"
"        F51.5 梦游症：睡眠中出现起床或行走等复杂行为。\n"
"        F51.9 非器质性睡眠障碍，未特指：睡眠异常但无具体分类。\n"
"    - F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。\n"
"        F98.0 非器质性遗尿症：儿童在不适当年龄仍有排尿失控。\n"
"        F98.1 非器质性遗粪症：儿童在不适当情境排便。\n"
"        F98.2 婴儿期或儿童期进食障碍：儿童进食行为异常影响营养或发育。\n"
"        F98.3 异食癖：持续摄入非食物性物质。\n"
"        F98.4 刻板性运动障碍：重复、无目的的运动习惯。\n"
"        F98.5 口吃：言语流利性障碍，表现为言语阻塞或重复。\n"
"        F98.6 习惯性动作障碍：如咬甲、吮指等持续存在的习惯。\n"
"        F98.8 其他特指的儿童行为和情绪障碍：符合儿童期特异但不归入上述类。\n"
"        F98.9 未特指的儿童行为和情绪障碍：症状存在但缺乏分类依据。\n"
"    - F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。\n"
"        F42.0 以强迫观念为主：反复出现难以摆脱的思想或冲动。\n"
"        F42.1 以强迫行为为主：反复、仪式化的动作难以控制。\n"
"        F42.2 强迫观念与强迫行为混合：思想和动作同时反复困扰。\n"
"        F42.9 强迫障碍，未特指：存在强迫症状但分类不详。\n"
"    - F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。\n"
"        F31.0 躁狂期，无精神病性症状：躁狂明显但无妄想或幻觉。\n"
"        F31.1 躁狂期，有精神病性症状：躁狂发作伴妄想或幻觉。\n"
"        F31.2 抑郁期，无精神病性症状：抑郁发作但无精神病性特征。\n"
"        F31.3 抑郁期，有精神病性症状：抑郁伴妄想或幻觉。\n"
"        F31.4 混合状态：躁狂与抑郁症状同时或快速交替出现。\n"
"        F31.5 缓解期：既往双相障碍，当前症状缓解。\n"
"        F31.6 其他状态：不符合典型躁狂/抑郁/混合的表现。\n"
"        F31.9 未特指：双相障碍，但无法进一步分类。\n"
"    - F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。\n"
"        F43.0 急性应激反应：暴露于重大应激后立即出现短暂严重反应。\n"
"        F43.1 创伤后应激障碍：经历创伤事件后持续出现再体验、回避和警觉性增高。\n"
"        F43.2 适应障碍：对生活变故反应过度，伴情绪或行为异常。\n"
"        F43.8 其他反应性障碍：与应激相关但不符合特定诊断。\n"
"        F43.9 未特指：应激反应存在，但资料不足以分类。\n"
"    - F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。\n"
"        F45.0 躯体化障碍：反复多样的身体症状无器质性解释。\n"
"        F45.1 未分化的躯体形式障碍：躯体症状存在但未达到躯体化标准。\n"
"        F45.2 疑病障碍：持续担忧患严重疾病。\n"
"        F45.3 自主神经功能紊乱型：以心悸、胸闷等自主神经症状为主。\n"
"        F45.4 持续性躯体疼痛障碍：慢性疼痛为主要表现。\n"
"        F45.8 其他躯体形式障碍：特殊类型躯体症状但不归入上述类。\n"
"        F45.9 未特指：存在躯体症状但无法分类。\n"
"    - F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。\n"
"        F20.0 偏执型：以妄想和幻听为主。\n"
"        F20.1 紊乱型：思维、情感和行为紊乱显著。\n"
"        F20.2 紧张型：以木僵、紧张性兴奋为主要表现。\n"
"        F20.3 未分化型：符合精神分裂症但不属特定亚型。\n"
"        F20.4 残留状态：阴性症状为主，病程长期。\n"
"        F20.5 精神分裂症后抑郁：精神分裂症后出现显著抑郁。\n"
"        F20.6 单纯型：逐渐出现阴性症状，无显著阳性症状。\n"
"        F20.8 其他类型：特殊表现但不属于前述类别。\n"
"        F20.9 未特指：存在精神分裂症特征但资料不足。\n"
"    - Z71 咨询和医疗建议相关因素：包括心理咨询、健康教育、生活方式指导等，当患者主要需要咨询服务而非特定疾病治疗时使用。\n"
"        Z71.9 未特指的咨询：提供咨询，但缺乏具体分类。\n\n"
f"## 对话历史：\n{cleaned_dialogue_history_for_diagnosis}\n\n"
f"## 患者背景信息：\n{self.patient_template.get('cleaned_text', '')}\n\n"
"## 注意：\n"
"1. 问诊对话为初次问诊，在症状严重程度和细节不可判断的时候，请推荐未特指的icd code。\n"
"2. 诊断结果包含1-2个icd-10诊断代码，大多包含1个。但也有2个的情况，用分号分隔不同的代码\n"
"3. 需要严格根据icd-10标准来进行诊断的分析, 避免猜测和无根据的诊断，避免诊断错误。\n"
"## 输出格式：\n"
"请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将最后诊断的ICD-10代码必须放在<box>xxx</box>中输出，用分号分隔，格式如：<think>xxx</think>xxx<box>Fxx.x;Fxx.x</box>。"
        )

        diagnosis_messages = [
            {"role": "system", "content": self.doctor_persona},
            {"role": "user", "content": diagnosis_prompt}
        ]
        
        # 保存诊断 messages 供后续记录 reasoning 使用
        self._last_diagnosis_messages = diagnosis_messages
        
        # 记录诊断请求信息
        diagnosis_request_info = {
            'step_name': 'diagnosis_llm_request',
            'model': self.api_model_name if self.use_api else 'local_model',
            'use_api': self.use_api,
            'messages': diagnosis_messages,  # 记录完整的messages
            'temperature': 0.3,
            'max_new_tokens': 512 if not self.use_api else None,  # 本地模型使用max_new_tokens
            'diagnosis_prompt_length': len(diagnosis_prompt),
            'system_prompt_length': len(self.doctor_persona)
        }
        
        # 带重试机制的诊断生成
        max_retries = 2
        diag_result = None
        chat_response = None
        diagnosis_response_info = None
        
        for retry in range(max_retries):
            try:
                if self.use_api:
                    chat_response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=diagnosis_messages,
                        temperature=0.3,
                        max_tokens=llm_tools_api.get_max_tokens(),
                        **self._reasoning_kwargs()
                    )
                    # 添加空值检查
                    if chat_response and chat_response.choices and len(chat_response.choices) > 0:
                        # 使用 extract_answer_and_reasoning 替代 extract_reasoning_from_response
                        diag_result, diag_reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
                        
                        # 为了保持兼容性，获取 message 对象
                        message = chat_response.choices[0].message
                        
                        if diag_reasoning:
                            logger.info(f"[诊断生成] ✅ 找到reasoning字段（长度: {len(diag_reasoning)}字符）")
                        
                        # 验证 reasoning 和 content 不是患者响应，并清理患者部分
                        patient_indicators = [
                            "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                            "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                            "我是一名", "我是患者", "作为患者", "患者角色"
                        ]
                        
                        # 清理 reasoning，移除患者部分，只保留医生的部分
                        if diag_reasoning:
                            diag_reasoning = self._clean_patient_content_from_reasoning(diag_reasoning, patient_indicators)
                        
                        # 清理 content，移除患者部分，只保留医生的部分
                        if diag_result:
                            cleaned_diag_result = self._clean_patient_content_from_reasoning(diag_result, patient_indicators)
                            if cleaned_diag_result:
                                diag_result = cleaned_diag_result
                                logger.info(f"[诊断生成] ✅ 已清理content中的患者部分，保留医生部分（长度: {len(diag_result)}字符）")
                            else:
                                logger.warning(f"[诊断生成] ⚠️ content完全包含患者内容，保留原始content但标记为异常")
                            
                            # 提取answer内容（去掉<answer>标签）
                            diag_result = self._extract_answer_from_content(diag_result)
                        
                        # 如果有reasoning，组合成完整响应（格式：<think>reasoning</think><answer>content</answer>）
                        diag_full_response = None
                        if diag_reasoning:
                            diag_full_response = f"<think>\n{diag_reasoning}\n</think>\n\n<answer>{diag_result}</answer>"
                        else:
                            diag_full_response = diag_result
                        
                        # 再次验证并清理 diag_full_response（双重检查）
                        if diag_full_response:
                            is_patient_full_response = any(indicator in diag_full_response for indicator in patient_indicators)
                            if is_patient_full_response:
                                logger.warning(f"[诊断生成] ⚠️ 检测到 diag_full_response 仍包含患者内容，尝试清理")
                                # 如果 full_response 包含患者内容，尝试从其中提取医生的部分
                                if "<think>" in diag_full_response and "<answer>" in diag_full_response:
                                    # 提取 reasoning 和 answer 部分
                                    reasoning_match = re.search(r'<think>(.*?)</think>', diag_full_response, re.DOTALL)
                                    answer_match = re.search(r'<answer>(.*?)</answer>', diag_full_response, re.DOTALL)
                                    
                                    if reasoning_match and answer_match:
                                        extracted_reasoning = reasoning_match.group(1).strip()
                                        extracted_answer = answer_match.group(1).strip()
                                        
                                        # 清理两部分
                                        cleaned_reasoning = self._clean_patient_content_from_reasoning(extracted_reasoning, patient_indicators) if extracted_reasoning else None
                                        cleaned_answer = self._clean_patient_content_from_reasoning(extracted_answer, patient_indicators) if extracted_answer else None
                                        
                                        # 重新组合
                                        if cleaned_reasoning and cleaned_answer:
                                            diag_full_response = f"<think>\n{cleaned_reasoning}\n</think>\n\n<answer>{cleaned_answer}</answer>"
                                            diag_reasoning = cleaned_reasoning  # 更新保存的 reasoning
                                            diag_result = cleaned_answer  # 更新保存的 result
                                            logger.info(f"[诊断生成] ✅ 已清理 diag_full_response 中的患者部分，保留医生 reasoning 和 answer")
                                        elif cleaned_answer:
                                            # reasoning 完全被清除，只保留 answer
                                            diag_full_response = f"<answer>{cleaned_answer}</answer>"
                                            diag_reasoning = None
                                            diag_result = cleaned_answer
                                            logger.warning(f"[诊断生成] ⚠️ reasoning 完全包含患者内容，已清除，只保留 answer")
                                        elif cleaned_reasoning:
                                            # answer 完全被清除，只保留 reasoning（这种情况较少见）
                                            diag_full_response = f"<think>\n{cleaned_reasoning}\n</think>"
                                            diag_reasoning = cleaned_reasoning
                                            diag_result = ""
                                            logger.warning(f"[诊断生成] ⚠️ answer 完全包含患者内容，已清除，只保留 reasoning")
                                        else:
                                            # 两部分都完全被清除
                                            logger.warning(f"[诊断生成] ⚠️ reasoning 和 answer 都完全包含患者内容，使用原始 diag_result")
                                            diag_full_response = diag_result
                                            diag_reasoning = None
                                    else:
                                        # 无法解析，使用清理后的 diag_result
                                        cleaned_diag_result = self._clean_patient_content_from_reasoning(diag_result, patient_indicators)
                                        if cleaned_diag_result:
                                            diag_full_response = cleaned_diag_result
                                            diag_result = cleaned_diag_result
                                            logger.info(f"[诊断生成] ✅ 已清理 diag_result 中的患者部分")
                                        else:
                                            logger.warning(f"[诊断生成] ⚠️ 无法清理，使用原始 diag_result")
                                else:
                                    # 没有标签，直接清理整个内容
                                    cleaned_full = self._clean_patient_content_from_reasoning(diag_full_response, patient_indicators)
                                    if cleaned_full:
                                        diag_full_response = cleaned_full
                                        diag_result = cleaned_full
                                        logger.info(f"[诊断生成] ✅ 已清理 diag_full_response 中的患者部分")
                                    else:
                                        logger.warning(f"[诊断生成] ⚠️ diag_full_response 完全包含患者内容，使用原始 diag_result")
                                        diag_full_response = diag_result
                            # 注意：不要在这里覆盖 diag_full_response，因为之前已经正确构建了包含 reasoning 的完整响应
                            # 只有在检测到患者内容且清理失败的情况下，才使用 diag_result
                        
                        # 保存reasoning和完整响应供后续使用
                        self._last_diagnosis_reasoning = diag_reasoning
                        self._last_diagnosis_full_response = diag_full_response
                        
                        # 记录API响应信息
                        prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                            chat_response,
                            messages=diagnosis_messages,
                            response_text=diag_result
                        )
                        diagnosis_response_info = {
                            'retry_count': retry + 1,
                            'success': True,
                            'response_length': len(diag_result) if diag_result else 0,
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total_tokens': prompt_tokens + completion_tokens if prompt_tokens and completion_tokens else None,
                            'response_text': diag_result,  # 记录完整的响应文本
                            'reasoning': diag_reasoning,  # 记录reasoning
                            'full_response': diag_full_response  # 记录完整响应
                        }
                        super().money_cost(prompt_tokens, completion_tokens)
                    else:
                        logger.error("诊断响应为空")
                        raise ValueError("诊断响应为空")
                else:
                    # 本地模型
                    text = self.doctor_tokenizer.apply_chat_template(
                        diagnosis_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    diag_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self._get_device())
                    diag_outputs = self.doctor_model.generate(
                        diag_inputs.input_ids,
                        max_new_tokens=512,
                        temperature=0.3,
                        do_sample=True
                    )
                    diag_generated = diag_outputs[0][diag_inputs.input_ids.shape[1]:]
                    raw_diag_result = self.doctor_tokenizer.decode(diag_generated, skip_special_tokens=True)
                    
                    # 提取answer内容（去掉<answer>标签）
                    diag_result = self._extract_answer_from_content(raw_diag_result)
                    
                    # 从本地模型的原始响应中提取reasoning（如果有<think>标签）
                    diag_reasoning = None
                    diag_full_response = None
                    if '<think>' in raw_diag_result or '<Think>' in raw_diag_result:
                        # 提取think内容
                        think_match = re.search(r'<think>(.*?)</think>', raw_diag_result, re.DOTALL | re.IGNORECASE)
                        if think_match:
                            diag_reasoning = think_match.group(1).strip()
                            diag_full_response = raw_diag_result  # 本地模型已经包含完整格式
                        else:
                            diag_full_response = diag_result
                    else:
                        diag_full_response = diag_result
                    
                    # 保存reasoning和完整响应供后续使用
                    self._last_diagnosis_reasoning = diag_reasoning
                    self._last_diagnosis_full_response = diag_full_response
                    
                    # 记录本地模型响应信息
                    input_token_count = diag_inputs.input_ids.shape[1]
                    output_token_count = diag_generated.shape[0]
                    diagnosis_response_info = {
                        'retry_count': retry + 1,
                        'success': True,
                        'response_length': len(diag_result) if diag_result else 0,
                        'input_tokens': input_token_count,
                        'output_tokens': output_token_count,
                        'completion_tokens': output_token_count,  # 添加completion_tokens字段以保持一致性
                        'total_tokens': input_token_count + output_token_count,
                        'response_text': diag_result,  # 记录完整的响应文本
                        'reasoning': diag_reasoning,  # 记录reasoning（从<think>标签提取）
                        'full_response': diag_full_response  # 记录完整响应
                    }
                
                if diag_result and diag_result.strip():
                    break
                    
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"诊断生成失败，重试第 {retry + 1} 次: {e}")
                    diagnosis_response_info = {
                        'retry_count': retry + 1,
                        'success': False,
                        'error': str(e)
                    }
                    continue
                else:
                    logger.error(f"诊断生成失败: {e}")
                    diagnosis_response_info = {
                        'retry_count': retry + 1,
                        'success': False,
                        'error': str(e),
                        'final_error': True
                    }
                    raise
        
        # 将请求和响应信息添加到追踪记录
        diagnosis_request_info['response'] = diagnosis_response_info
        diagnosis_trace_entry['steps'].append(diagnosis_request_info)
        
        # 保存诊断的token信息、reasoning和完整响应，供后续记录reasoning使用
        if diagnosis_response_info:
            self._last_diagnosis_tokens = diagnosis_response_info.get('completion_tokens', 0)
            self._last_diagnosis_reasoning = diagnosis_response_info.get('reasoning', None)
            self._last_diagnosis_full_response = diagnosis_response_info.get('full_response', diag_result)
        else:
            self._last_diagnosis_tokens = 0
            self._last_diagnosis_reasoning = None
            self._last_diagnosis_full_response = diag_result
        
        diag_result = "诊断结束，你的诊断结果为：" + diag_result
        # 返回格式与 doctor_v2.py 一致：(diag_result, diag_reasoning)
        diag_reasoning = self._last_diagnosis_reasoning if hasattr(self, '_last_diagnosis_reasoning') else ""
        return diag_result, diag_reasoning
    
    def _build_input_prompt(self, dialogue_history, include_doctor_persona=False):
        """
        构建input_prompt（从messages中提取，去除think标签和RAG内容）
        
        Args:
            dialogue_history: 对话历史列表
            include_doctor_persona: 是否在user_content中包含doctor_persona（设置为True可确保prompt_tokens逐轮增加）
            
        Returns:
            tuple: (input_prompt, prompt_tokens_val)
                - input_prompt: 格式化的JSON字符串
                - prompt_tokens_val: prompt的token数
        """
        input_prompt = None
        cleaned_messages = None
        prompt_tokens_val = 0
        
        if not self.messages:
            return input_prompt, prompt_tokens_val
            
        try:
            # 清理messages，去除assistant消息中的think标签，同时清理user消息中对话历史字符串里的think标签和rag/cot内容
            cleaned_messages = []
            for msg in self.messages:
                cleaned_msg = msg.copy()
                if msg.get('role') == 'assistant' and 'content' in msg:
                    content = msg['content']
                    # 提取answer部分（去除think标签和answer标签）
                    if '<answer>' in content or '<Answer>' in content:
                        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
                        if answer_match:
                            cleaned_msg['content'] = answer_match.group(1).strip()
                        else:
                            # 如果没有匹配到，去除所有标签
                            cleaned_msg['content'] = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                            cleaned_msg['content'] = re.sub(r'<answer>.*?</answer>', '', cleaned_msg['content'], flags=re.DOTALL | re.IGNORECASE)
                            cleaned_msg['content'] = cleaned_msg['content'].strip()
                    elif '<think>' in content or '<Think>' in content:
                        # 如果没有answer标签，去除think标签
                        cleaned_msg['content'] = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()
                    else:
                        # 如果没有任何标签，保持原样
                        cleaned_msg['content'] = content
                elif msg.get('role') == 'user' and 'content' in msg:
                    # 从原始dialogue_history重新构建user消息的content，避免双重转义
                    # 首先清理对话历史中的think和answer标签
                    cleaned_dialogue_history = []
                    for hist_item in dialogue_history:
                        if isinstance(hist_item, str):
                            # 清理医生回复中的think和answer标签
                            if hist_item.startswith('医生:'):
                                # 提取answer内容
                                answer_match = re.search(r'<answer>(.*?)</answer>', hist_item, re.DOTALL | re.IGNORECASE)
                                if answer_match:
                                    cleaned_item = '医生: ' + answer_match.group(1).strip()
                                else:
                                    # 去除think标签
                                    cleaned_item = re.sub(r'<think>.*?</think>', '', hist_item, flags=re.DOTALL | re.IGNORECASE).strip()
                                    cleaned_item = re.sub(r'<answer>.*?</answer>', '', cleaned_item, flags=re.DOTALL | re.IGNORECASE).strip()
                                cleaned_dialogue_history.append(cleaned_item)
                            elif hist_item.startswith('患者本人:') or hist_item.startswith('患者:'):
                                # 清理患者回复中的think和answer标签
                                patient_prefix = '患者本人: ' if hist_item.startswith('患者本人:') else '患者: '
                                
                                # 先尝试提取answer内容（如果有）
                                answer_match = re.search(r'<answer>(.*?)</answer>', hist_item, re.DOTALL | re.IGNORECASE)
                                if answer_match:
                                    cleaned_item = patient_prefix + answer_match.group(1).strip()
                                else:
                                    # 如果没有answer标签，去除think标签，保留实际回复内容
                                    # 先去除think标签
                                    cleaned_item = re.sub(r'<think>.*?</think>', '', hist_item, flags=re.DOTALL | re.IGNORECASE)
                                    # 去除answer标签（如果有残留）
                                    cleaned_item = re.sub(r'<answer>.*?</answer>', '', cleaned_item, flags=re.DOTALL | re.IGNORECASE)
                                    # 去除残留的标签标记
                                    cleaned_item = re.sub(r'</?think>', '', cleaned_item, flags=re.IGNORECASE)
                                    cleaned_item = re.sub(r'</?answer>', '', cleaned_item, flags=re.IGNORECASE)
                                    cleaned_item = cleaned_item.strip()
                                    
                                    # 确保保留患者前缀
                                    if not cleaned_item.startswith(patient_prefix):
                                        # 如果去除标签后前缀也被去除了，重新添加
                                        content_without_prefix = cleaned_item.replace(patient_prefix, '').strip()
                                        cleaned_item = patient_prefix + content_without_prefix
                                
                                # 去除患者回复中的换行符（\n 和实际的换行）
                                if cleaned_item.startswith(patient_prefix):
                                    content_part = cleaned_item[len(patient_prefix):]
                                    # 去除所有换行符（包括 \n 和实际的换行）
                                    content_part = content_part.replace('\n', ' ').replace('\\n', ' ')
                                    # 去除多余的空格
                                    content_part = re.sub(r'\s+', ' ', content_part).strip()
                                    cleaned_item = patient_prefix + content_part
                                
                                cleaned_dialogue_history.append(cleaned_item)
                            else:
                                cleaned_dialogue_history.append(hist_item)
                        else:
                            cleaned_dialogue_history.append(hist_item)
                    
                    # 获取当前话题
                    current_topic = self.topic_seq[self.current_idx] if self.current_idx < len(self.topic_seq) else "患者的精神状况"
                    
                    # 获取empathy_text
                    empathy_text = ""
                    if hasattr(self, 'doctor_prompt') and self.doctor_prompt and self.doctor_prompt.get('empathy') == '有':
                        empathy_text = "在适当的时候提供与患者的共情"
                    else:
                        if include_doctor_persona:
                            empathy_text = "简洁的，口语化的表达进行文本生成"
                        else:
                            empathy_text = "简洁的生成"
                    
                    # 重新构建user消息的content（不包含RAG内容）
                    # 注意：不要将 dialogue_history 序列化为 JSON 字符串再嵌入，而是直接使用列表
                    # 这样在最终的 json.dumps() 时，会正确转义，避免双重转义
                    # 将列表转换为 JSON 格式的字符串（但不进行转义），然后嵌入
                    # 使用 json.dumps 生成 JSON 字符串，但之后会在最终的 json.dumps 中再次转义
                    # 解决方案：先序列化，然后在最终序列化后修复转义
                    dialogue_history_json_str = json.dumps(cleaned_dialogue_history, ensure_ascii=False)
                    
                    # 检查对话历史中是否有重复的问题
                    recent_doctor_questions = []
                    for item in cleaned_dialogue_history[-10:]:  # 检查最近10轮对话
                        if isinstance(item, str) and item.startswith('医生:'):
                            question = item.replace('医生:', '').strip()
                            if question:
                                recent_doctor_questions.append(question)
                    
                    # 构建防重复提示
                    anti_repeat_instruction = ""
                    if len(recent_doctor_questions) >= 2:
                        # 如果最近有多个问题，检查是否有重复
                        unique_questions = set(recent_doctor_questions)
                        if len(unique_questions) < len(recent_doctor_questions):
                            anti_repeat_instruction = "\n\n【重要】检查对话历史，如果患者已经回答过某个问题，绝对不要再问相同或类似的问题！必须基于患者已经回答的内容，提出新的、不同的问题。如果当前话题的信息已经足够，可以询问该话题的其他方面或切换到下一个话题。"
                    
                    if include_doctor_persona:
                        user_content = self.doctor_persona + self.patient_persona + f"\n你与患者的所有对话历史如下{dialogue_history_json_str}，" + \
                            f"\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{current_topic}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。。" + \
                            f"3.\n你每次只能围绕1个话题询问。使用{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头或相同的问题。{anti_repeat_instruction}\n输出有标点符号的一段文字，不要换行。"
                    else:
                        user_content = self.patient_persona + f"\n你与患者的所有对话历史如下{dialogue_history_json_str}，" + \
                            f"\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{current_topic}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。" + \
                            f"\n3.你每次只能围绕1个话题询问。使用口语化的表达，{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头或相同的问题。{anti_repeat_instruction}\n输出有标点符号的一段文字，不要换行。"
                    
                    cleaned_msg['content'] = user_content
                cleaned_messages.append(cleaned_msg)
            
            # 使用json.dumps序列化整个消息列表
            input_prompt = json.dumps(cleaned_messages, ensure_ascii=False, indent=2)
            
            # 修复双重转义问题：当我们将 dialogue_history_json_str 嵌入到 user_content 中，
            # 再对整个消息列表进行 json.dumps 时，JSON字符串中的转义字符会被再次转义
            # 解决方案：解析JSON，修复字符串值中的转义，然后重新序列化
            try:
                # 解析 JSON
                parsed_messages = json.loads(input_prompt)
                # 递归修复所有字符串值中的多重转义
                def fix_escapes_in_dict(obj):
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if isinstance(value, str):
                                # 修复字符串中的多重转义
                                # 在JSON解析后的Python字符串中：
                                # - `\\n` 表示字面上的 `\n`（一个反斜杠加n）
                                # - `\\\\n` 表示字面上的 `\\n`（两个反斜杠加n）
                                # - `\\\\\\"` 表示字面上的 `\\"`（两个反斜杠加引号）
                                # 我们需要将这些转换为实际的字符，然后让 json.dumps 重新转义
                                
                                # 先处理四重转义（\\\\\\" -> \"）
                                value = value.replace('\\\\\\\\"', '"')
                                value = value.replace('\\\\\\\\n', '\n')
                                value = value.replace('\\\\\\\\t', '\t')
                                value = value.replace('\\\\\\\\r', '\r')
                                
                                # 再处理三重转义（\\\" -> "）
                                value = value.replace('\\\\"', '"')
                                value = value.replace('\\\\n', '\n')
                                value = value.replace('\\\\t', '\t')
                                value = value.replace('\\\\r', '\r')
                                
                                obj[key] = value
                            else:
                                fix_escapes_in_dict(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix_escapes_in_dict(item)
                
                fix_escapes_in_dict(parsed_messages)
                # 重新序列化
                input_prompt = json.dumps(parsed_messages, ensure_ascii=False, indent=2)
            except Exception as e:
                # 如果解析失败，记录错误但继续使用原始 input_prompt
                logger.warning(f"[_build_input_prompt] 修复转义时出错: {e}")
            
            # 计算清理后的messages的token数（不包含RAG内容）
            # 这样prompt_tokens只包含system_tokens + dialogue_history_tokens + topic_instruction_tokens
            # 使用cleaned_messages确保prompt_tokens逐轮增大（因为对话历史逐轮增加）
            if cleaned_messages:
                prompt_tokens_val = llm_tools_api.estimate_tokens(cleaned_messages)
        except:
            pass
        
        return input_prompt, prompt_tokens_val
    
    def enable_reasoning_generation(self):
        """启用reasoning生成"""
        patient_id = self.patient_template.get('patient_id', 'unknown')
        self.reasoning_generator = ReasoningGenerator(patient_id)
        self.enable_reasoning = True
        self.turn_counter = 0
        
        # 更新系统提示词，添加thinking要求（用于非RAG/CoT阶段）
        thinking_instruction = "\n\n【重要】在生成回复前，请先进行思考，将思考过程放在<think>...</think>标签中，然后将实际回复放在<answer>...</answer>标签中。\n\n思考过程要求：\n1) 对患者当前描述的具体理解：要具体说明患者提到了什么症状、情绪、行为等，不要使用'患者描述了相关情况'这样的通用表述\n2) 基于对话历史的分析：结合之前的对话，分析患者症状的变化、特点、可能的关联等\n3) 下一步问诊的策略和目的：明确说明为什么要问这个问题，想了解什么信息，这个问题如何帮助诊断\n\n示例：\n<think>\n患者提到最近压力大、学习吃力，而且妹妹被孤立的事一直压在心里感到愧疚。情绪低落为主，持续了两年但最近一个月明显加重。还提到注意力差、对以前喜欢的事提不起劲。这些症状提示可能是抑郁相关的问题。我需要了解患者之前是否寻求过帮助，这有助于判断问题的严重程度和既往治疗情况。所以我要询问既往求助经历。\n</think>\n<answer>之前有没有去过心理科咨询过或者吃过什么药？</answer>"
        
        # 更新doctor_persona（系统提示词）
        # doctor_v2_rag_cot.py 使用 doctor_persona 而不是 system_prompt
        if hasattr(self, 'doctor_persona') and self.doctor_persona is not None:
            self.doctor_persona = self.doctor_persona + thinking_instruction
        
        # 更新messages中的system prompt（如果有）
        if self.messages and len(self.messages) > 0:
            for msg in self.messages:
                if msg.get('role') == 'system' and msg.get('content') is not None:
                    msg['content'] = msg['content'] + thinking_instruction
                    break
    
    def record_reasoning_turn_with_rag(
        self,
        patient_message: str,
        doctor_message: str,
        rag_trace_entry: Optional[Dict] = None,
        doctor_full_response: Optional[str] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        input_prompt: Optional[str] = None,
        is_diagnosis: bool = False
    ):
        """
        记录一个reasoning轮次（有RAG/CoT阶段）
        
        Args:
            patient_message: 患者消息
            doctor_message: 医生回复（已提取的纯文本）
            rag_trace_entry: RAG/CoT追踪条目（consultation_cot阶段）
            doctor_full_response: 医生完整响应（包含think标签，可选）
            tokens: 生成的token数
            prompt_tokens: prompt的token数
            input_prompt: 输入prompt（可选）
            is_diagnosis: 是否是诊断
        """
        if not (self.enable_reasoning and self.reasoning_generator):
            return
        
        self.turn_counter += 1
        
        # 从rag_trace_entry中提取步骤信息
        rag_search_step = None
        disease_extraction_step = None
        cot_reasoning_step = None
        
        if rag_trace_entry and rag_trace_entry.get('stage') == 'consultation_cot':
            steps = rag_trace_entry.get('steps', [])
            for step in steps:
                if step.get('step_name') == 'rag_search':
                    rag_search_step = step
                elif step.get('step_name') == 'disease_extraction':
                    disease_extraction_step = step
                elif step.get('step_name') == 'cot_reasoning':
                    cot_reasoning_step = step
        
        # 如果有RAG追踪，使用RAG格式；否则使用无RAG格式
        if rag_search_step or disease_extraction_step or cot_reasoning_step:
            # 获取当前话题（用于生成更准确的thinking）
            current_topic = None
            if rag_trace_entry:
                current_topic = rag_trace_entry.get('current_topic', '')
            if not current_topic:
                # 尝试从当前阶段获取话题
                if self.current_idx < len(self.topic_seq):
                    current_topic = self.topic_seq[self.current_idx]
            
            # 调用大模型将格式化的RAG/CoT内容转化为口语化的简单表达
            converted_think_content = None
            if rag_trace_entry:
                try:
                    # 提取格式化的RAG/CoT内容
                    formatted_rag_content = rag_trace_entry.get('final_combined_context', '')
                    if not formatted_rag_content:
                        # 如果没有final_combined_context，从各个步骤中组合
                        knowledge_context = rag_trace_entry.get('knowledge_context', '')
                        possible_diseases = rag_trace_entry.get('possible_diseases', '')
                        symptom_analysis = rag_trace_entry.get('symptom_analysis', '')
                        cot_reasoning = rag_trace_entry.get('cot_reasoning', '')
                        
                        parts = []
                        if knowledge_context:
                            parts.append(f"【诊疗指南参考】\n{knowledge_context}")
                        if possible_diseases:
                            parts.append(f"【3个候选疾病（按可能性从高到低排序）】\n{possible_diseases}")
                        if symptom_analysis:
                            parts.append(f"【3个疾病在{current_topic}方面的症状差异分析】\n{symptom_analysis}")
                        if cot_reasoning:
                            parts.append(f"【思考过程】\n{cot_reasoning}")
                        formatted_rag_content = "\n\n".join(parts)
                    
                    if formatted_rag_content:
                        # 调用大模型转换为口语化表达
                        conversion_prompt = f"""你是一名精神科医生，正在对患者进行问诊。以下是基于诊疗指南和推理过程的格式化分析内容：

{formatted_rag_content}

请将这些格式化的分析内容转化为一段自然、口语化的思考过程，就像医生在内心思考一样。要求：
1. 用第一人称（"我"）表达
2. 语言自然、口语化，不要使用结构化的标题、标号
3. 将诊疗指南的关键信息、候选疾病分析、症状差异分析、推理过程等内容自然地整合成连贯的思考
4. 思考应该包括：对患者症状的理解、可能的疾病判断、需要进一步了解的信息
5. 不要直接列出疾病名称的标记（如"疾病A"、"疾病B"等），而是自然地描述这些疾病的可能性，使用具体的疾病名称
6. 长度控制在200字以内

请直接输出思考内容，不要添加任何标签或格式："""
                        
                        conversion_messages = [
                            {"role": "system", "content": "你是一名专业的精神科医生，擅长将专业的医学分析转化为自然的口语化思考过程。"},
                            {"role": "user", "content": conversion_prompt}
                        ]
                        
                        conversion_response = self.client.chat.completions.create(
                            model=self.api_model_name,
                            messages=conversion_messages,
                            temperature=0.7,
                            max_tokens=300
                        )
                        
                        if conversion_response and conversion_response.choices:
                            converted_think_content = conversion_response.choices[0].message.content.strip()
                            logger.info(f"[RAG思考转换] 成功将RAG内容转换为口语化表达，长度: {len(converted_think_content)}")
                        else:
                            logger.warning("[RAG思考转换] 转换失败，使用默认thinking")
                    else:
                        logger.warning("[RAG思考转换] 没有找到RAG内容，使用默认thinking")
                except Exception as e:
                    logger.warning(f"[RAG思考转换] 转换过程出错: {e}，使用默认thinking")
            
            self.reasoning_generator.add_turn_with_rag(
                self.turn_counter,
                patient_message,
                doctor_message,
                doctor_full_response=doctor_full_response or doctor_message,
                rag_search_step=rag_search_step,
                disease_extraction_step=disease_extraction_step,
                cot_reasoning_step=cot_reasoning_step,
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                input_prompt=input_prompt,
                is_diagnosis=is_diagnosis,
                current_topic=current_topic,
                converted_think_content=converted_think_content  # 传递转换后的thinking内容
            )
        else:
            # 没有RAG追踪，使用无RAG格式
            self.reasoning_generator.add_turn_without_rag(
                self.turn_counter,
                patient_message,
                doctor_message,
                doctor_full_response=doctor_full_response or doctor_message,
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                input_prompt=input_prompt,
                is_diagnosis=is_diagnosis
            )
    
    def record_reasoning_turn(
        self, 
        patient_message: str, 
        doctor_message: str, 
        thinking: str = None,
        doctor_full_response: Optional[str] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        input_prompt: Optional[str] = None,
        is_greeting: bool = False,
        is_diagnosis: bool = False
    ):
        """
        记录一个reasoning轮次（qwen3格式）
        根据当前阶段决定使用RAG/CoT格式还是无RAG格式：
        - 评估阶段（ASSESSMENT）和深入阶段（DEEPDIVE）：使用RAG/CoT格式
        - 其他阶段（筛查、风险、总结）：使用无RAG格式
        
        Args:
            patient_message: 患者消息
            doctor_message: 医生回复（已提取的纯文本）
            thinking: 思考过程（已废弃，从doctor_full_response中提取）
            doctor_full_response: 医生完整响应（包含think标签，可选）
            tokens: 生成的token数
            prompt_tokens: prompt的token数
            input_prompt: 输入prompt（可选）
            is_greeting: 是否是问候语
            is_diagnosis: 是否是诊断
        """
        if not (self.enable_reasoning and self.reasoning_generator):
            return
        
        # 获取当前阶段
        current_phase = self._get_current_phase()
        
        # 判断是否应该使用RAG/CoT格式（只在评估和深入阶段）
        should_use_rag_format = False
        if current_phase in [DiagnosticPhase.ASSESSMENT, DiagnosticPhase.DEEPDIVE]:
            should_use_rag_format = True
        
        # 如果应该使用RAG格式，查找RAG追踪信息
        if should_use_rag_format:
            rag_trace_entry = None
            # 检查是否有当前的RAG追踪（从_cot_reasoning_with_rag设置的）
            if self.current_rag_trace:
                rag_trace_entry = self.current_rag_trace
                self.current_rag_trace = None  # 清除当前追踪
            else:
                # 尝试从rag_cot_trace中获取最新的consultation_cot追踪
                if self.rag_cot_trace:
                    for trace in reversed(self.rag_cot_trace):
                        if trace.get('stage') == 'consultation_cot':
                            rag_trace_entry = trace
                            break
            
            # 如果有RAG追踪，使用RAG格式
            if rag_trace_entry:
                self.record_reasoning_turn_with_rag(
                    patient_message, 
                    doctor_message, 
                    rag_trace_entry,
                    doctor_full_response=doctor_full_response or doctor_message,
                    tokens=tokens,
                    prompt_tokens=prompt_tokens,
                    input_prompt=input_prompt,
                    is_diagnosis=is_diagnosis
                )
            else:
                # 应该使用RAG格式但没有找到追踪，使用无RAG格式（降级处理）
                logger.warning(f"[Reasoning] 当前阶段 {current_phase.value if current_phase else 'Unknown'} 应使用RAG格式，但未找到RAG追踪，使用无RAG格式")
                self.turn_counter += 1
                self.reasoning_generator.add_turn_without_rag(
                    self.turn_counter,
                    patient_message,
                    doctor_message,
                    doctor_full_response=doctor_full_response or doctor_message,
                    tokens=tokens,
                    prompt_tokens=prompt_tokens,
                    input_prompt=input_prompt,
                    is_greeting=is_greeting,
                    is_diagnosis=is_diagnosis
                )
        else:
            # 使用无RAG格式（筛查、风险、总结阶段）
            self.turn_counter += 1
            self.reasoning_generator.add_turn_without_rag(
                self.turn_counter,
                patient_message,
                doctor_message,
                doctor_full_response=doctor_full_response or doctor_message,
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                input_prompt=input_prompt,
                is_greeting=is_greeting,
                is_diagnosis=is_diagnosis
            )
    
    def get_reasoning_data(self):
        """获取所有reasoning数据（qwen3格式）"""
        if self.reasoning_generator:
            # 返回simulation_dialogue（已经是qwen3格式）
            simulation_dialogue = self.reasoning_generator.get_all_turns()
            # 返回格式：{'patient_id': 'xxx', 'simulation_dialogue': [...]}
            patient_id = str(self.patient_template.get('patient_id', 'unknown'))
            return {
                'patient_id': patient_id,
                'simulation_dialogue': simulation_dialogue
            }
        return None


# 为了支持动态加载，提供 Doctor 别名
Doctor = DoctorV2RAGCoT

