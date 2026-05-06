"""
DeepSeek 客户端模块

用于调用 DeepSeek 模型完成去冗余和格式统一任务
"""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from loguru import logger

from config_loader import load_config
from prompts import (
    DEEPSEEK_DEDUP_SYSTEM_PROMPT,
    DEEPSEEK_DEDUP_USER_PROMPT_TEMPLATE,
)


def call_deepseek_dedup(
    prev_table: str,
    next_table: str,
    prev_batch_index: int,
    config: Dict,
    sub_dir: str = "",
) -> Tuple[Dict[str, str], Dict]:
    """
    调用 DeepSeek 模型对前后两批的问答表进行去冗余和格式统一

    Args:
        prev_table: 前一批的问答表 Markdown 文本
        next_table: 后一批的问答表 Markdown 文本
        prev_batch_index: 前一批的批次索引
        config: 配置字典
        sub_dir: 子目录名（用于区分不同问卷的输出，如 "interview_protocol_CN_7"）

    Returns:
        (解析后的结果字典, DeepSeek 元数据)
        结果字典包含: prev_keep, prev_removed, next_adjusted
    """
    cfg = config["deepseek"]
    output_root = config["paths"]["output_root"]

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置")

    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=api_key,
    )

    # 构造消息
    user_prompt = DEEPSEEK_DEDUP_USER_PROMPT_TEMPLATE.format(
        prev_table=prev_table,
        next_table=next_table,
    )

    messages = [
        {"role": "system", "content": DEEPSEEK_DEDUP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 调用 API
    logger.info(f"正在调用 DeepSeek API 处理前后批次（前一批={prev_batch_index}，后一批={prev_batch_index+1}）的去冗余...")
    start_time = time.time()

    response = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=0.0,  # 使用确定性输出
    )

    elapsed = time.time() - start_time
    logger.info(f"DeepSeek API 返回成功，耗时 {elapsed:.1f}s")

    # 提取回复内容
    reply_content = response.choices[0].message.content
    if not reply_content:
        raise ValueError("DeepSeek 返回了空内容")

    # 尝试解析 JSON（可能被 ```json ... ``` 包裹）
    result = _parse_json_from_response(reply_content)

    # 保存去冗余结果（使用 sub_dir 隔离不同问卷的输出）
    if sub_dir:
        dedup_dir = os.path.join(output_root, sub_dir, "dedup", f"batch_{prev_batch_index:03d}_to_{prev_batch_index+1:03d}")
    else:
        dedup_dir = os.path.join(output_root, "dedup", f"batch_{prev_batch_index:03d}_to_{prev_batch_index+1:03d}")
    os.makedirs(dedup_dir, exist_ok=True)

    # 保存 LLM 原始回复
    raw_path = os.path.join(dedup_dir, "dedup_raw_response.md")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(reply_content)

    # 分别保存提取的三部分
    for key in ["prev_keep", "prev_removed", "next_adjusted"]:
        if key in result:
            file_path = os.path.join(dedup_dir, f"{key}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result[key])

    # 记录LLM元数据（兼容不同API的token字段名）
    usage = response.usage
    llm_metadata = {
        "model": response.model,
        "description": f"DeepSeek 去冗余: batch_{prev_batch_index:03d}_to_{prev_batch_index+1:03d}",
        "input_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt": messages[0]["content"] if messages and messages[0].get("role") == "user" else "",
        "response": reply_content,
    }

    logger.info(f"  tokens: input={llm_metadata['input_tokens']}, output={llm_metadata['output_tokens']}, total={llm_metadata['total_tokens']}")

    # 构建元数据（兼容旧格式，保留原有字段）
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prev_batch_index": prev_batch_index,
        "next_batch_index": prev_batch_index + 1,
        "sub_dir": sub_dir,
        "model": response.model,
        "usage": {
            "input_tokens": llm_metadata["input_tokens"],
            "output_tokens": llm_metadata["output_tokens"],
            "total_tokens": llm_metadata["total_tokens"],
        },
        "response_id": response.id,
        "elapsed_seconds": round(elapsed, 2),
        "output_files": {
            key: os.path.join(dedup_dir, f"{key}.md") for key in ["prev_keep", "prev_removed", "next_adjusted"]
        },
        "raw_response_file": raw_path,
    }

    meta_path = os.path.join(dedup_dir, "dedup_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"去冗余结果已存档: {dedup_dir}")

    # 日志记录前一批被去除的部分
    removed_content = result.get("prev_removed", "").strip()
    if removed_content:
        logger.info(f"前一批（batch_{prev_batch_index}）被去除的内容：\n{removed_content}")
    else:
        logger.info(f"前一批（batch_{prev_batch_index}）无内容被去除")

    return result, metadata


def _parse_json_from_response(text: str) -> Dict[str, str]:
    """
    从 LLM 回复中提取 JSON 并解析

    兼容 ```json ... ``` 包裹和纯 JSON 的情况
    """
    # 尝试直接解析
    text = text.strip()

    # 去掉 ```json ... ``` 包裹
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = text

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"无法直接解析 JSON，尝试修复: {e}")
        # 尝试提取最外层花括号
        brace_start = json_str.find('{')
        brace_end = json_str.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            json_str = json_str[brace_start:brace_end + 1]
            result = json.loads(json_str)
        else:
            raise ValueError(f"无法从 DeepSeek 回复中解析 JSON：{text[:500]}")

    required_keys = ["prev_keep", "prev_removed", "next_adjusted"]
    for key in required_keys:
        if key not in result:
            logger.warning(f"DeepSeek 回复缺少字段: {key}，设为空字符串")
            result[key] = ""

    return result
