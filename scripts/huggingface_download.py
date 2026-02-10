#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face Dataset 下载工具

用于从 Hugging Face Hub 下载 LingxiDiag-16K 数据集。
支持使用 hf-mirror 镜像加速下载。

使用示例:
    # 下载数据集（使用镜像）
    python scripts/huggingface_download.py \
        --repo-name "your_username/lingxidiag-16k" \
        --output-dir "./downloaded_data" \
        --token "your_huggingface_token"
    
    # 下载数据集（不使用镜像）
    python scripts/huggingface_download.py \
        --repo-name "your_username/lingxidiag-16k" \
        --output-dir "./downloaded_data" \
        --no-mirror

环境变量配置（可选）:
    export HF_TOKEN=your_huggingface_token
"""

import json
import os
import sys
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, login

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目路径配置
PROJECT_ROOT = Path(__file__).parent.parent


class LingxiDatasetDownloader:
    """
    LingxiDiag-16K 数据集下载工具类
    支持使用hf-mirror镜像加速下载
    """
    
    def __init__(self, use_mirror: bool = True, mirror_url: str = "https://hf-mirror.com"):
        self.api = HfApi()
        self.use_mirror = use_mirror
        self.mirror_url = mirror_url
        
        if self.use_mirror:
            self._setup_mirror()
    
    def _setup_mirror(self):
        """设置Hugging Face镜像"""
        logger.info(f"🪞 配置Hugging Face镜像: {self.mirror_url}")
        os.environ["HF_ENDPOINT"] = self.mirror_url
        
        try:
            import datasets
            datasets.config.HF_ENDPOINT = self.mirror_url
            logger.info("✅ 镜像配置完成")
        except Exception as e:
            logger.warning(f"⚠️ 镜像配置警告: {str(e)}")
    
    def validate_token(self, token: str) -> bool:
        """
        验证 Hugging Face token 是否有效
        
        Args:
            token: Hugging Face 访问令牌
            
        Returns:
            token 是否有效
        """
        try:
            api = HfApi(token=token)
            user_info = api.whoami()
            logger.info(f"✅ Token 验证成功，用户: {user_info['name']}")
            return True
        except Exception as e:
            logger.error(f"❌ Token 验证失败: {str(e)}")
            return False
    
    def check_dataset_access(self, repo_name: str, token: str) -> bool:
        """
        检查是否有访问数据集的权限
        
        Args:
            repo_name: 数据集名称
            token: 访问令牌
            
        Returns:
            是否有访问权限
        """
        try:
            api = HfApi(token=token)
            dataset_info = api.dataset_info(repo_name)
            logger.info(f"✅ 数据集访问权限验证成功: {repo_name}")
            return True
        except Exception as e:
            logger.error(f"❌ 数据集访问权限验证失败: {str(e)}")
            if "401" in str(e):
                logger.info("💡 可能的原因:")
                logger.info("   1. Token 没有访问该数据集的权限")
                logger.info("   2. 数据集是私有的，但您不是协作者")
                logger.info("   3. Token 作用域权限不足")
            return False
    
    def download_dataset_cli(
        self, 
        repo_name: str, 
        token: Optional[str] = None, 
        local_dir: str = "./temp_dataset"
    ) -> str:
        """
        使用 huggingface-cli 下载数据集（备选方案）
        
        Args:
            repo_name: 仓库名称
            token: 访问令牌
            local_dir: 本地下载目录
            
        Returns:
            下载的本地目录路径
        """
        logger.info(f"🔧 使用 CLI 工具下载数据集: {repo_name}")
        
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        env = os.environ.copy()
        if self.use_mirror:
            env["HF_ENDPOINT"] = self.mirror_url
            logger.info(f"🪞 设置镜像: {self.mirror_url}")
        
        if token:
            env["HF_TOKEN"] = token
            logger.info("🔐 使用提供的token")
        
        try:
            cmd = [
                "huggingface-cli", "download",
                "--repo-type", "dataset",
                "--resume-download",
                repo_name,
                "--local-dir", str(local_path)
            ]
            
            logger.info(f"🚀 执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("✅ CLI 下载成功")
            logger.info(f"📁 下载目录: {local_path}")
            return str(local_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ CLI 下载失败: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("❌ 未找到 huggingface-cli 命令")
            logger.info("💡 请安装: pip install huggingface_hub[cli]")
            raise
    
    def load_parquet_from_download(self, local_dir: str) -> List[Dict]:
        """
        从下载的目录中加载parquet文件
        
        Args:
            local_dir: 下载的本地目录
            
        Returns:
            数据列表
        """
        local_path = Path(local_dir)
        data_dir = local_path / "data"
        
        if not data_dir.exists():
            data_dir = local_path
        
        parquet_files = list(data_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到 parquet 文件")
        
        logger.info(f"📄 找到 {len(parquet_files)} 个parquet文件")
        
        all_data = []
        for parquet_file in parquet_files:
            logger.info(f"📖 加载文件: {parquet_file.name}")
            try:
                df = pd.read_parquet(parquet_file)
                data = self._convert_df_to_json_compatible(df)
                all_data.extend(data)
                logger.info(f"✅ 加载 {len(data)} 条数据")
            except Exception as e:
                logger.error(f"❌ 加载 {parquet_file.name} 失败: {e}")
                raise
        
        logger.info(f"📋 总计加载: {len(all_data)} 条数据")
        return all_data
    
    def _convert_df_to_json_compatible(self, df: pd.DataFrame) -> List[Dict]:
        """
        将DataFrame转换为JSON兼容的数据列表
        
        Args:
            df: pandas DataFrame
            
        Returns:
            JSON兼容的字典列表
        """
        data = []
        for record in df.to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                if value is None:
                    clean_record[key] = None
                elif hasattr(value, 'tolist'):
                    clean_record[key] = value.tolist()
                elif hasattr(value, 'item') and hasattr(value, 'shape') and value.shape == ():
                    clean_record[key] = value.item()
                elif hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                    try:
                        clean_record[key] = list(value)
                    except Exception:
                        clean_record[key] = str(value)
                else:
                    clean_record[key] = value
            data.append(clean_record)
        return data
    
    def download_dataset(
        self, 
        repo_name: str, 
        split: Optional[str] = None, 
        token: Optional[str] = None,
        use_cli_fallback: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        从Hugging Face Hub下载数据集
        
        Args:
            repo_name: 仓库名称
            split: 数据分割名称（如 'train', 'validation', 'test'）
            token: 访问令牌
            use_cli_fallback: 如果datasets库失败，是否尝试使用CLI工具
            
        Returns:
            按分割名称组织的数据字典
        """
        # 验证token
        if token:
            logger.info("🔍 正在验证 Hugging Face token...")
            if not self.validate_token(token):
                raise ValueError("❌ 提供的 token 无效")
            
            logger.info("🔍 正在检查数据集访问权限...")
            if not self.check_dataset_access(repo_name, token):
                raise ValueError("❌ 没有访问该数据集的权限")
        
        if self.use_mirror:
            logger.info(f"📥 正在通过镜像 {self.mirror_url} 下载数据集: {repo_name}")
        else:
            logger.info(f"📥 正在从官方源下载数据集: {repo_name}")
        
        # 登录
        if token:
            login(token=token)
            logger.info("🔐 使用提供的token登录")
        else:
            env_token = os.getenv('HF_TOKEN')
            if env_token:
                login(token=env_token)
                token = env_token
                logger.info("🔐 使用环境变量HF_TOKEN登录")
        
        try:
            # 使用 datasets 库下载
            logger.info("🔄 尝试使用 datasets 库下载...")
            
            if split:
                dataset = load_dataset(repo_name, split=split, token=token)
                logger.info(f"✅ 成功下载 {split} 分割，共 {len(dataset)} 条数据")
                data = {split: [dict(item) for item in dataset]}
            else:
                dataset = load_dataset(repo_name, token=token)
                logger.info(f"✅ 成功下载数据集，包含分割: {list(dataset.keys())}")
                
                data = {}
                for split_name, split_data in dataset.items():
                    data[split_name] = [dict(item) for item in split_data]
                    logger.info(f"📊 {split_name}: {len(data[split_name])} 条数据")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ datasets 库下载失败: {str(e)}")
            
            if use_cli_fallback:
                logger.info("\n🔄 尝试使用 CLI 工具下载...")
                try:
                    temp_dir = f"./temp_{repo_name.split('/')[-1]}"
                    local_dir = self.download_dataset_cli(repo_name, token, temp_dir)
                    all_data = self.load_parquet_from_download(local_dir)
                    
                    # 清理临时目录
                    try:
                        shutil.rmtree(local_dir)
                        logger.info(f"🧹 清理临时目录: {local_dir}")
                    except Exception:
                        logger.warning(f"⚠️ 无法清理临时目录: {local_dir}")
                    
                    return {"all": all_data}
                    
                except Exception as cli_error:
                    logger.error(f"❌ CLI 下载也失败: {str(cli_error)}")
            
            raise Exception(f"所有下载方法都失败了。错误: {str(e)}")
    
    def save_to_json(
        self, 
        data: Dict[str, List[Dict]], 
        output_dir: Union[str, Path],
        format_type: str = "list",
        indent: int = 2
    ) -> List[str]:
        """
        将数据保存为JSON文件
        
        Args:
            data: 按分割名称组织的数据字典
            output_dir: 输出目录
            format_type: JSON格式类型 ('list', 'data_wrapper', 'lines')
            indent: JSON缩进空格数
            
        Returns:
            保存的文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for split_name, split_data in data.items():
            output_path = output_dir / f"LingxiDiag-16K_{split_name}_data.json"
            
            logger.info(f"💾 正在保存 {split_name} 到: {output_path}")
            
            if format_type == "list":
                output_data = split_data
            elif format_type == "data_wrapper":
                output_data = {"data": split_data}
            elif format_type == "lines":
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in split_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                logger.info(f"✅ 已保存为JSONL格式，共 {len(split_data)} 条数据")
                saved_files.append(str(output_path))
                continue
            else:
                raise ValueError("format_type 必须是 'list', 'data_wrapper' 或 'lines'")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=indent)
            
            logger.info(f"✅ 已保存为JSON格式，共 {len(split_data)} 条数据")
            saved_files.append(str(output_path))
        
        return saved_files
    
    def download_and_save(
        self, 
        repo_name: str, 
        output_dir: Union[str, Path],
        split: Optional[str] = None, 
        token: Optional[str] = None,
        format_type: str = "list",
        indent: int = 2
    ) -> List[str]:
        """
        下载数据集并保存为JSON文件
        
        Args:
            repo_name: 仓库名称
            output_dir: 输出目录
            split: 数据分割名称
            token: 访问令牌
            format_type: JSON格式类型
            indent: JSON缩进
            
        Returns:
            保存的文件路径列表
        """
        # 下载数据
        data = self.download_dataset(repo_name, split, token)
        
        # 保存为JSON
        saved_files = self.save_to_json(data, output_dir, format_type, indent)
        
        return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='从Hugging Face Hub下载LingxiDiag-16K数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--repo-name',
        type=str,
        required=True,
        help='Hugging Face仓库名称，格式为 "username/dataset-name"'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PROJECT_ROOT / "downloaded_data"),
        help='输出目录（默认: ./downloaded_data）'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face访问令牌（也可通过HF_TOKEN环境变量设置）'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'validation', 'test'],
        help='要下载的数据分割（默认下载全部）'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='list',
        choices=['list', 'data_wrapper', 'lines'],
        help='输出JSON格式（默认: list）'
    )
    
    parser.add_argument(
        '--no-mirror',
        action='store_true',
        help='不使用hf-mirror镜像（默认使用镜像加速）'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始下载LingxiDiag-16K数据集")
    logger.info("=" * 60)
    
    # 创建下载器
    downloader = LingxiDatasetDownloader(use_mirror=not args.no_mirror)
    
    # 下载并保存
    saved_files = downloader.download_and_save(
        repo_name=args.repo_name,
        output_dir=args.output_dir,
        split=args.split,
        token=args.token,
        format_type=args.format
    )
    
    logger.info("=" * 60)
    logger.info("✅ 数据集下载完成！")
    for file_path in saved_files:
        logger.info(f"📄 保存文件: {file_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

