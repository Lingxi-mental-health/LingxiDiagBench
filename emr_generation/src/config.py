"""
配置文件 - 存储项目配置
"""

from pathlib import Path


class Config:
    """项目配置类"""
    
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 数据目录
    MAPPING_DIR = PROJECT_ROOT / "mapping"
    REAL_EMRS_DIR = PROJECT_ROOT / "real_emrs"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    SYMPTOMS_DIR = PROJECT_ROOT / "symptoms"
    
    # 症状定义文件
    SYMPTOMS_FILE = SYMPTOMS_DIR / "251_Symptoms_Descriptions_cn-en.xlsx"
    
    # 默认数据文件
    DEFAULT_DATA_FILE = REAL_EMRS_DIR / "SMHC_Collected_train_data.json"
    
    # 映射文件
    DISTRIBUTION_MAPPING_FILE = MAPPING_DIR / "distribution_mapping.json"
    KEYWORD_MAPPING_FILE = MAPPING_DIR / "keyword_mapping.json"
    DIAGNOSIS_CODE_MAPPING_FILE = MAPPING_DIR / "diagnosis_code_mapping.json"
    PERSONAL_HISTORY_MAPPING_FILE = MAPPING_DIR / "personal_history_mapping.json"
    
    # LLM配置
    LLM_DEFAULT_MODEL = "../../models/Qwen3-32B"
    LLM_DEFAULT_HOST = "localhost"
    LLM_DEFAULT_PORT = 8000
    LLM_MAX_TOKENS = 4096
    LLM_TEMPERATURE = 0.7
    
    # 分词配置
    STOPWORDS = {"的", "了", "在", "是", "有", "和", "与", "及", "等", "为", "以", 
                 "患者", "无", "否认", "门诊", "就诊", "来", "故"}
    
    @classmethod
    def get_llm_base_url(cls, host: str = None, port: int = None) -> str:
        """获取LLM API base URL"""
        host = host or cls.LLM_DEFAULT_HOST
        port = port or cls.LLM_DEFAULT_PORT
        return f"http://{host}:{port}/v1"
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.MAPPING_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
