"""Application configuration helpers for the Patient Agent backend."""

from __future__ import annotations

import os
from typing import Dict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def _env_bool(key: str, default: bool = False) -> bool:
    """Return environment variable as boolean with sensible defaults."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_agent_config(
    prefix: str,
    fallback: Dict[str, object],
    openrouter_fallback_model: str,
) -> Dict[str, object]:
    """
    Construct agent-specific configuration using environment overrides.

    Parameters
    ----------
    prefix:
        Environment variable prefix, e.g. ``PATIENT`` / ``DOCTOR`` / ``VERIFIER``.
    fallback:
        Baseline configuration to fall back to when prefix-specific env vars
        are not defined.
    openrouter_fallback_model:
        Default model identifier for OpenRouter when prefix-specific value missing.
    
    Supports both new and old environment variable names:
    - New: MODEL_MODE, OPENROUTER_{prefix}_MODEL, OFFLINE_{prefix}_MODEL, OFFLINE_{prefix}_PORTS
    - Old: {prefix}_USE_OPENROUTER, {prefix}_OPENROUTER_MODEL, {prefix}_MODEL_NAME, {prefix}_MODEL_PORT
    
    Also supports remote VLLM deployment via VLLM_{prefix}_IP environment variable.
    """
    # 读取模型模式（新变量名，优先）
    model_mode = os.environ.get("MODEL_MODE", "").lower()
    
    # 决定是否使用 OpenRouter
    if model_mode == "openrouter":
        use_openrouter = True
    elif model_mode == "offline":
        use_openrouter = False
    else:
        # 回退到旧的变量名或 fallback
        use_openrouter = _env_bool(
            f"{prefix}_USE_OPENROUTER", bool(fallback["use_openrouter"])
        )
    
    # OpenRouter 模型配置（新变量名优先，然后旧变量名，最后 fallback）
    openrouter_model = os.environ.get(
        f"OPENROUTER_{prefix}_MODEL",  # 新变量名
        os.environ.get(
            f"{prefix}_OPENROUTER_MODEL",  # 旧变量名
            openrouter_fallback_model,     # fallback
        ),
    )
    
    # 离线模型配置（新变量名优先，然后旧变量名，最后 fallback）
    local_name = os.environ.get(
        f"OFFLINE_{prefix}_MODEL",  # 新变量名
        os.environ.get(
            f"{prefix}_MODEL_NAME",  # 旧变量名
            fallback["local_model_name"],  # fallback
        ),
    )
    
    # 离线模型端口（新变量名优先，然后旧变量名，最后 fallback）
    # 注意：新变量名支持多端口（逗号分隔），这里只取第一个
    ports_str = os.environ.get(
        f"OFFLINE_{prefix}_PORTS",  # 新变量名
        os.environ.get(
            f"{prefix}_MODEL_PORT",  # 旧变量名
            str(fallback["local_model_port"]),  # fallback
        ),
    )
    # 如果是逗号分隔的多端口，取第一个
    local_port = int(ports_str.split(",")[0].strip())
    
    # 检查是否配置了远程 VLLM IP
    vllm_ip = os.environ.get(f"VLLM_{prefix}_IP", "").strip()
    
    # 如果配置了远程IP，将模型名称格式化为 model_name@host:port
    # 这样在创建客户端时可以正确解析
    if vllm_ip and not use_openrouter:
        local_name = f"{local_name}@{vllm_ip}:{local_port}"

    return {
        "use_openrouter": use_openrouter,
        "local_model_name": local_name,
        "local_model_port": local_port,
        "openrouter_model": openrouter_model,
    }


def load_config() -> Dict[str, object]:
    """Load application configuration from environment variables.
    
    Supports both new and old environment variable names for backwards compatibility.
    """
    # 默认模型路径（优先使用旧的 MODEL_NAME 保持兼容性）
    default_local_model = os.environ.get(
        "MODEL_NAME",
        "/mnt/tcci/shihao/outputs/dataset_v2/qwen3-32B_auxiliary_diagnosis_lora-sft_reasoning_kimi-k2-0905_v7_lr5e-6"
    )
    
    # 默认端口（优先使用旧的 MODEL_PORT 保持兼容性）
    default_local_port = int(os.environ.get("MODEL_PORT", "9019"))
    
    # 默认模式（新变量名优先，然后旧变量名）
    model_mode = os.environ.get("MODEL_MODE", "").lower()
    if model_mode == "openrouter":
        default_use_openrouter = True
    elif model_mode == "offline":
        default_use_openrouter = False
    else:
        # 回退到旧的 USE_OPENROUTER
        default_use_openrouter = _env_bool("USE_OPENROUTER", True)
    
    # 默认 OpenRouter 模型（旧变量名，保持兼容性）
    default_openrouter_model = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-32b")

    # 基础配置作为所有模型的默认值
    base_fallback = {
        "use_openrouter": default_use_openrouter,
        "local_model_name": default_local_model,
        "local_model_port": default_local_port,
        "openrouter_model": default_openrouter_model,
    }

    # EverPsychosis (Patient) - 默认使用 OpenRouter Qwen3-32B
    patient_fallback = {
        "use_openrouter": True,  # 默认使用 OpenRouter
        "local_model_name": default_local_model,
        "local_model_port": default_local_port,
        "openrouter_model": "qwen/qwen3-32b",  # 默认 Qwen3-32B
    }

    # EverPsychiatrist (Doctor) - 默认使用 OpenRouter Qwen3-32B  
    doctor_fallback = {
        "use_openrouter": True,  # 默认使用 OpenRouter
        "local_model_name": default_local_model,
        "local_model_port": default_local_port,
        "openrouter_model": "qwen/qwen3-32b",  # 默认 Qwen3-32B
    }

    # EverDiagnosis (Verifier) - 默认使用 VLLM 本地部署 port=9019
    verifier_fallback = {
        "use_openrouter": False,  # 默认使用本地 VLLM
        "local_model_name": default_local_model,
        "local_model_port": default_local_port,  # 使用默认端口（9019）
        "openrouter_model": default_openrouter_model,
    }

    patient_model = _build_agent_config(
        "PATIENT",
        patient_fallback,
        patient_fallback["openrouter_model"],
    )

    doctor_model = _build_agent_config(
        "DOCTOR", 
        doctor_fallback,
        doctor_fallback["openrouter_model"],
    )

    verifier_model = _build_agent_config(
        "VERIFIER",
        verifier_fallback,
        verifier_fallback["openrouter_model"],
    )

    config: Dict[str, object] = {
        "project_root": str(PROJECT_ROOT),
        "patient_data_path": os.environ.get(
            "PATIENT_DATA_PATH",
            str(PROJECT_ROOT / "raw_data" / "SMHC_EverDiag-16K_validation_data_100samples.json"),
        ),
        "max_patients": int(os.environ.get("MAX_PATIENTS", "100")),
        "backend_port": int(os.environ.get("BACKEND_PORT", "5001")),
        "user_db_path": os.environ.get(
            "USER_DB_PATH",
            str(PROJECT_ROOT / "patient_test_ui" / "data" / "users.json"),
        ),
        "openrouter": {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "site_url": os.environ.get("OPENROUTER_SITE_URL", ""),
            "site_name": os.environ.get("OPENROUTER_SITE_NAME", "Patient Agent Test"),
        },
        "models": {
            "patient": patient_model,
            "doctor": doctor_model,
            "verifier": verifier_model,
        },
        # Agent版本配置
        "agent_versions": {
            "patient": {
                "default": os.environ.get("PATIENT_VERSION", "v1"),  # 默认版本
                "available": ["v1", "cot"],  # 可用版本
            },
            "doctor": {
                "default": os.environ.get("DOCTOR_VERSION", "base"),  # 默认版本
                "available": ["base", "v1", "v2"],  # 可用版本
            },
        },
    }

    # Backwards compatibility for legacy callers.
    config["model_name"] = patient_model["local_model_name"]
    config["model_port"] = patient_model["local_model_port"]
    config["use_openrouter"] = patient_model["use_openrouter"]
    config["openrouter_model"] = patient_model["openrouter_model"]

    return config


CONFIG = load_config()
