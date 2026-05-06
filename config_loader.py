"""
YAML 配置加载模块
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


# 默认配置
DEFAULT_CONFIG = {
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-1-6-flash-250828",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "batch": {
        "size": 4,
        "overlap": 1,
    },
    "paths": {
        "survey_root": "survey_raw",
        "output_root": "llm-output",
    },
}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载 YAML 配置文件，缺失字段使用默认值

    Args:
        config_path: 配置文件路径（相对于工作目录）

    Returns:
        配置字典
    """
    config = DEFAULT_CONFIG.copy()

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        if user_config:
            # 递归合并
            _deep_merge(config, user_config)
    else:
        print(f"[WARN] 配置文件 {config_path} 不存在，使用默认配置")

    # 验证必要的环境变量
    if not os.getenv("ARK_API_KEY"):
        print("[WARN] 环境变量 ARK_API_KEY 未设置，豆包 API 调用将失败")
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("[WARN] 环境变量 DEEPSEEK_API_KEY 未设置，DeepSeek API 调用将失败")

    return config


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    递归合并字典，override 中的值会覆盖 base 中的值
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
