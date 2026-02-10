#!/usr/bin/env python3
"""
Patient Agent FastAPI 服务

提供Patient Agent的API接口，支持patient_v1和patient_cot两种版本
支持两种模型部署方式：
1. OpenRouter API 模式 (PATIENT_USE_OPENROUTER=true)
2. 本地 VLLM 模式 (PATIENT_USE_OPENROUTER=false)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import json
import os
import uuid
import time
import asyncio
from dotenv import load_dotenv
import uvicorn

# 加载环境变量
load_dotenv()

# 导入Patient类
from patient_v1 import Patient as PatientV1
from patient_cot import Patient as PatientCOT
from patient_mdd5k import Patient as PatientMDD5K
from patient_v3 import Patient as PatientV3

# 创建FastAPI应用
app = FastAPI(
    title="Patient Agent API",
    description="精神科患者AI Agent API接口",
    version="1.0.0"
)

# ==================== 数据模型 ====================

class Message(BaseModel):
    """OpenAI消息格式"""
    role: str = Field(..., description="消息角色：system/user/assistant")
    content: str = Field(..., description="消息内容")

class PatientRequest(BaseModel):
    """
    Patient Agent请求参数（无状态设计）
    
    每次请求必须包含完整的对话历史，API不保存对话状态。
    """
    patient_id: str = Field(..., description="患者ID（字符串类型）")
    messages: List[Message] = Field(
        ..., 
        description="完整的对话历史（OpenAI格式），包括所有之前的user和assistant消息", 
        min_items=1
    )
    model_name: Optional[str] = Field(None, description="模型名称（OpenRouter模式必填，本地模式可选）")
    patient_version: Literal["v1", "mdd5k", "v3", "cot"] = Field(default="cot", description="Patient版本：v1, mdd5k, v3 或 cot")
    current_topic: Optional[str] = Field(None, description="当前话题（可选）")

class ChatMessage(BaseModel):
    """聊天消息格式（OpenAI标准）"""
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")

class Choice(BaseModel):
    """选择项（OpenAI标准）"""
    index: int = Field(..., description="选择项索引")
    message: ChatMessage = Field(..., description="消息内容")
    finish_reason: str = Field(..., description="完成原因")

class Usage(BaseModel):
    """使用统计（OpenAI标准）"""
    prompt_tokens: int = Field(..., description="输入tokens数")
    completion_tokens: int = Field(..., description="输出tokens数")
    total_tokens: int = Field(..., description="总tokens数")
    cost_usd: float = Field(..., description="成本（美元）")

class PatientResponse(BaseModel):
    """Patient Agent响应（OpenAI标准格式）"""
    id: str = Field(..., description="唯一ID")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="使用的模型")
    choices: List[Choice] = Field(..., description="选择项列表")
    usage: Usage = Field(..., description="使用统计")
    # 扩展字段（非标准OpenAI字段）
    patient_id: Optional[str] = Field(None, description="患者ID")
    patient_version: Optional[str] = Field(None, description="Patient版本")
    classification: Optional[Dict] = Field(None, description="问题分类信息（仅COT版本）")

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细错误信息")

# ==================== 全局变量 ====================

# 患者数据存储
PATIENTS_DATA = {}
PATIENT_DATA_FILE = None

# 模型配置
USE_OPENROUTER = os.getenv('PATIENT_USE_OPENROUTER', 'true').lower() == 'true'
OPENROUTER_MODEL = os.getenv('OPENROUTER_PATIENT_MODEL', 'qwen/qwen3-32b')
OFFLINE_MODEL = os.getenv('OFFLINE_PATIENT_MODEL', '../models/qwen3-32b')
OFFLINE_PORTS = os.getenv('OFFLINE_PATIENT_PORTS', '9040')
VLLM_IP = os.getenv('VLLM_PATIENT_IP', '')

# Patient实例缓存（避免重复初始化）
PATIENT_INSTANCES = {}

# 并发控制锁（保护缓存字典）
CACHE_LOCK = asyncio.Lock()

# 每个患者实例的创建锁（避免同一患者被并发创建多次）
CREATION_LOCKS = {}

# ==================== 辅助函数 ====================

def load_patient_data(file_path: str):
    """加载患者数据"""
    global PATIENTS_DATA, PATIENT_DATA_FILE
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"患者数据文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 将列表转换为字典，以patient_id为key
    PATIENTS_DATA = {patient['patient_id']: patient for patient in data}
    PATIENT_DATA_FILE = file_path
    
    print(f"✓ 成功加载 {len(PATIENTS_DATA)} 个患者数据")
    print(f"  数据文件: {file_path}")
    print(f"  患者ID范围: {min(PATIENTS_DATA.keys())} - {max(PATIENTS_DATA.keys())}")

def build_model_path(model_name: Optional[str] = None) -> str:
    """
    根据环境变量构建模型路径
    
    Args:
        model_name: 可选的模型名称，用于OpenRouter模式的覆盖
        
    Returns:
        str: 模型路径
        - OpenRouter模式: 返回模型名称（如 'qwen/qwen3-32b'）
        - 本地VLLM模式: 返回完整路径（如 'model@host:port'）
    """
    if USE_OPENROUTER:
        # OpenRouter 模式：使用指定的模型名称或默认值
        return model_name or OPENROUTER_MODEL
    else:
        # 本地 VLLM 模式：构建模型路径
        # 格式: model_name@host:port 或 model_name:port
        ports = OFFLINE_PORTS.split(',')
        port = ports[0].strip()  # 使用第一个端口
        
        if VLLM_IP:
            # 使用远程IP
            model_path = f"{OFFLINE_MODEL}@{VLLM_IP}:{port}"
        else:
            # 使用本地
            model_path = f"{OFFLINE_MODEL}@127.0.0.1:{port}"
        
        return model_path

def validate_model_name(model_name: str):
    """
    验证模型名称
    
    Args:
        model_name: 模型名称
        
    注意：
    - OpenRouter模式：验证是否为OpenRouter支持的模型
    - 本地VLLM模式：此参数将被忽略，使用环境变量中的配置
    """
    if USE_OPENROUTER:
        # OpenRouter 模式：验证模型名称格式
        # 通常格式为 "provider/model-name"
        if '/' not in model_name:
            raise HTTPException(
                status_code=400,
                detail=f"OpenRouter模型名称格式错误: {model_name}，应为 'provider/model-name' 格式"
            )
    # 本地模式不需要验证，使用环境变量配置

async def get_or_create_patient(patient_id: str, model_name: str, patient_version: str):
    """
    获取或创建Patient实例（线程安全版本）
    
    Args:
        patient_id: 患者ID
        model_name: 模型名称（OpenRouter模式使用，本地模式忽略）
        patient_version: Patient版本 ("v1" 或 "cot")
        
    Returns:
        Patient实例
        
    说明:
        使用双重检查锁（Double-Check Locking）模式确保并发安全：
        1. 快速路径：无锁检查缓存（读取）
        2. 慢速路径：加锁后再次检查并创建实例（写入）
    """
    # 构建实际使用的模型路径
    actual_model_path = build_model_path(model_name if USE_OPENROUTER else None)
    
    # 缓存key（使用实际模型路径）
    cache_key = f"{patient_id}_{actual_model_path}_{patient_version}"
    
    # 快速路径：无锁检查缓存（大多数情况下命中）
    if cache_key in PATIENT_INSTANCES:
        return PATIENT_INSTANCES[cache_key]
    
    # 慢速路径：需要创建实例
    # 为每个cache_key创建独立的锁
    async with CACHE_LOCK:
        # 确保该cache_key有对应的创建锁
        if cache_key not in CREATION_LOCKS:
            CREATION_LOCKS[cache_key] = asyncio.Lock()
        creation_lock = CREATION_LOCKS[cache_key]
    
    # 使用该cache_key的专属锁（避免不同患者之间互相阻塞）
    async with creation_lock:
        # 双重检查：其他协程可能已经创建了实例
        if cache_key in PATIENT_INSTANCES:
            return PATIENT_INSTANCES[cache_key]
    
    # 检查patient_id是否存在
    if patient_id not in PATIENTS_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"患者ID {patient_id} 不存在。有效范围: {min(PATIENTS_DATA.keys())} - {max(PATIENTS_DATA.keys())}"
        )
    
    # 获取患者数据
    patient_template = PATIENTS_DATA[patient_id]
    
    # 创建Patient实例
    try:
        print(f"[Patient API] 创建Patient实例:")
        print(f"  - 患者ID: {patient_id}")
        print(f"  - 版本: {patient_version}")
        print(f"  - 模型路径: {actual_model_path}")
        print(f"  - 模式: {'OpenRouter' if USE_OPENROUTER else '本地VLLM'}")
        
        if patient_version == "v1":
            patient = PatientV1(
                patient_template=patient_template,
                model_path=actual_model_path,
                use_api=True
            )
        elif patient_version == "mdd5k":
            patient = PatientMDD5K(
                patient_template=patient_template,
                model_path=actual_model_path,
                use_api=True
            )
        elif patient_version == "v3":
            patient = PatientV3(
                patient_template=patient_template,
                model_path=actual_model_path,
                use_api=True
            )
        else:  # cot
            patient = PatientCOT(
                patient_template=patient_template,
                model_path=actual_model_path,
                use_api=True,
                enable_chief_complaint=False  # API模式不需要主诉生成
            )
        
        # 缓存实例
        PATIENT_INSTANCES[cache_key] = patient
        return patient
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"创建Patient实例失败: {str(e)}"
        )

def extract_dialogue_history(messages: List[Message]) -> list:
    """从OpenAI格式的messages提取对话历史"""
    dialogue_history = []
    
    for msg in messages:
        if msg.role == "system":
            continue  # 跳过system消息
        elif msg.role == "user":
            # 假设user是医生
            dialogue_history.append(f"医生: {msg.content}")
        elif msg.role == "assistant":
            # 假设assistant是患者
            dialogue_history.append(f"患者本人: {msg.content}")
    
    return dialogue_history

def get_current_doctor_question(messages: List[Message]) -> str:
    """获取当前医生的问题（最后一条user消息）"""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    
    # 如果没有找到user消息，返回默认问题
    return "你有什么不舒服的地方吗？"

def estimate_tokens(text: str) -> int:
    """
    简单估算文本的token数量
    中文：每个字符约1个token
    英文：每4个字符约1个token
    """
    if not text:
        return 0
    
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    
    return chinese_chars + (other_chars // 4)

def calculate_usage(messages: List[Message], response: str, cost: float) -> Usage:
    """
    计算token使用量
    
    Args:
        messages: 输入消息列表
        response: 响应内容
        cost: API成本
        
    Returns:
        Usage对象
    """
    # 估算输入tokens
    prompt_text = " ".join([msg.content for msg in messages])
    prompt_tokens = estimate_tokens(prompt_text)
    
    # 估算输出tokens
    completion_tokens = estimate_tokens(response)
    
    # 总tokens
    total_tokens = prompt_tokens + completion_tokens
    
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost
    )

# ==================== API路由 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 从环境变量或默认路径加载患者数据
    data_file = os.getenv(
        'PATIENT_DATA_FILE',
        './raw_data/SMHC_LingxiDiag-16K_validation_data_100samples.json'
    )
    
    try:
        load_patient_data(data_file)
        print("\n" + "="*60)
        print("Patient Agent API 服务启动成功")
        print("="*60)
        print(f"设计模式: 无状态（Stateless）")
        print(f"  ⚠️  每次请求必须包含完整的对话历史")
        print(f"  ⚠️  API 不保存对话状态")
        print(f"模型模式: {'OpenRouter' if USE_OPENROUTER else '本地VLLM'}")
        if USE_OPENROUTER:
            print(f"OpenRouter模型: {OPENROUTER_MODEL}")
            print(f"API Key: {'已配置' if os.getenv('OPENROUTER_API_KEY') else '未配置'}")
        else:
            print(f"本地模型: {OFFLINE_MODEL}")
            print(f"VLLM端口: {OFFLINE_PORTS}")
            if VLLM_IP:
                print(f"VLLM IP: {VLLM_IP}")
            else:
                print(f"VLLM IP: 127.0.0.1 (本地)")
            print(f"完整模型路径: {build_model_path()}")
        print(f"支持的Patient版本: v1, mdd5k, v3, cot")
        print(f"并发支持: ✓ (异步锁保护)")
        print("="*60 + "\n")
    except Exception as e:
        print(f"✗ 启动失败: {str(e)}")
        raise

@app.get("/")
async def root():
    """根路径"""
    config_info = {
        "mode": "OpenRouter" if USE_OPENROUTER else "本地VLLM",
    }
    
    if USE_OPENROUTER:
        config_info["openrouter_model"] = OPENROUTER_MODEL
        config_info["api_key_configured"] = bool(os.getenv('OPENROUTER_API_KEY'))
    else:
        config_info["offline_model"] = OFFLINE_MODEL
        config_info["vllm_ports"] = OFFLINE_PORTS
        config_info["vllm_ip"] = VLLM_IP or "127.0.0.1"
        config_info["model_path"] = build_model_path()
    
    return {
        "service": "Patient Agent API",
        "version": "1.0.0",
        "status": "running",
        "design": "stateless",
        "description": "Each request should include complete conversation history",
        "model_config": config_info,
        "patient_versions": ["v1", "mdd5k", "v3", "cot"],
        "total_patients": len(PATIENTS_DATA),
        "endpoints": {
            "chat": "/api/v1/patient/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "patients_loaded": len(PATIENTS_DATA),
        "cached_instances": len(PATIENT_INSTANCES)
    }

@app.post("/api/v1/patient/chat", response_model=PatientResponse)
async def patient_chat(request: PatientRequest):
    """
    Patient Agent对话接口（无状态设计）
    
    **重要**: 此API是无状态的，每次请求必须包含完整的对话历史。
    API不会保存对话状态，所有上下文信息都需要通过messages参数传递。
    
    Args:
        request: PatientRequest - 包含以下字段：
            - patient_id: 患者ID
            - messages: 完整的对话历史（OpenAI格式），包括所有之前的对话轮次
            - model_name: 模型名称（OpenRouter模式必填）
            - patient_version: Patient版本（v1或cot）
            - current_topic: 当前话题（可选）
    
    Returns:
        PatientResponse - OpenAI标准格式，包含患者回复、token统计、成本等信息
        
    Example:
        第一轮对话：
        {
            "patient_id": "0",
            "messages": [
                {"role": "user", "content": "你好，今天感觉怎么样？"}
            ]
        }
        
        第二轮对话（包含完整历史）：
        {
            "patient_id": "0",
            "messages": [
                {"role": "user", "content": "你好，今天感觉怎么样？"},
                {"role": "assistant", "content": "今天还是有点累..."},
                {"role": "user", "content": "能具体说说吗？"}
            ]
        }
    """
    # 1. 验证和准备模型名称
    if USE_OPENROUTER and not request.model_name:
        raise HTTPException(
            status_code=400,
            detail="OpenRouter模式下必须提供model_name参数"
        )
    
    if request.model_name:
        validate_model_name(request.model_name)
    
    # 2. 获取实际使用的模型路径
    actual_model_path = build_model_path(request.model_name if USE_OPENROUTER else None)
    
    # 3. 获取或创建Patient实例（异步操作）
    try:
        patient = await get_or_create_patient(
            request.patient_id,
            request.model_name or "default",
            request.patient_version
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取Patient实例失败: {str(e)}")
    
    # 4. 提取对话历史和当前问题
    dialogue_history = extract_dialogue_history(request.messages)
    current_doctor_question = get_current_doctor_question(request.messages)
    current_topic = request.current_topic or "患者的精神状况"
    
    # 5. 调用Patient生成回复（放入线程池，避免阻塞事件循环）
    try:
        # patient_response_gen 为同步方法，使用线程池提升并发能力
        patient_response, cost, classification_info, patient_reasoning = await asyncio.to_thread(
            patient.patient_response_gen,
            current_topic,
            dialogue_history,
            current_doctor_question
        )
        
        # 6. 构建OpenAI标准格式响应
        # 生成唯一ID
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        
        # 创建时间戳
        created_timestamp = int(time.time())
        
        # 计算token使用量
        usage = calculate_usage(request.messages, patient_response, cost)
        
        # 构建choices
        choices = [
            Choice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=patient_response
                ),
                finish_reason="stop"
            )
        ]
        
        # 返回OpenAI标准格式响应
        return PatientResponse(
            id=completion_id,
            object="chat.completion",
            created=created_timestamp,
            model=actual_model_path,
            choices=choices,
            usage=usage,
            # 扩展字段
            patient_id=request.patient_id,
            patient_version=request.patient_version,
            classification=classification_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成患者回复失败: {str(e)}"
        )


# ==================== 主函数 ====================

def main():
    """启动FastAPI服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patient Agent API服务')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8001, help='服务器端口')
    parser.add_argument('--data-file', type=str, 
                       default='./raw_data/SMHC_LingxiDiag-16K_validation_data_100samples.json',
                       help='患者数据文件路径')
    parser.add_argument('--reload', action='store_true', help='开启自动重载（开发模式）')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['PATIENT_DATA_FILE'] = args.data_file
    
    # 启动服务
    uvicorn.run(
        "patient_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()

