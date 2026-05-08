"""
豆包（Doubao）视觉大模型客户端模块

使用 OpenAI 兼容的 API 接口调用豆包视觉模型
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple
from openai import OpenAI
from pathlib import Path
from loguru import logger

from config_loader import load_config
from image_loader import encode_images_base64, BatchInfo
from prompts import DOUBAO_SYSTEM_PROMPT, DOUBAO_USER_PROMPT_TEMPLATE


def call_doubao(batch: BatchInfo, config: Dict, sub_dir: str = "", max_retries: int = 5) -> Tuple[str, str, Dict]:
    """
    调用豆包模型处理一批图片

    Args:
        batch: 批次信息
        config: 配置字典
        sub_dir: 子目录名（用于区分不同问卷的输出，如 "interview_protocol_CN_7"）
        max_retries: 最大重试次数（当返回内容为空时重试）

    Returns:
        (markdown回复文本, markdown文件保存路径, 元数据字典)
    """
    cfg = config["doubao"]
    output_root = config["paths"]["output_root"]

    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("环境变量 ARK_API_KEY 未设置")

    client = OpenAI(
        base_url=cfg["base_url"],
        api_key=api_key,
    )

    # 编码图片
    logger.info(f"正在编码批次 {batch.batch_index} 的 {len(batch.image_paths)} 张图片...")
    content_list = encode_images_base64(batch.image_paths)

    # 添加用户文本指令
    user_prompt = DOUBAO_USER_PROMPT_TEMPLATE.format(total=len(batch.image_paths))
    content_list.append({
        "type": "input_text",
        "text": user_prompt,
    })

    # 构造请求 input
    request_input = [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": DOUBAO_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": content_list,
        },
    ]

    # 调用 API（带重试）
    for attempt in range(max_retries):
        logger.info(f"正在调用豆包 API 处理批次 {batch.batch_index}（共 {len(batch.image_paths)} 张图）...（尝试 {attempt+1}/{max_retries}）")
        start_time = time.time()

        try:
            response = client.responses.create(
                model=cfg["model"],
                input=request_input,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)

            # 判断错误类型：超时类错误可重试，解析类错误重试也没用
            is_timeout = any(kw in error_msg.lower() for kw in ["timeout", "timed out", "time out"])
            is_parse_error = any(kw in error_msg.lower() for kw in ["parse", "parsing"])

            if is_parse_error:
                # 解析类错误：重试 1 次即可，大概率是请求格式问题
                if attempt < 1:
                    logger.warning(f"豆包 API 请求解析异常（第 {attempt+1} 次，耗时 {elapsed:.1f}s）: {e}")
                    logger.warning(f"等待 10 秒后重试 1 次...")
                    time.sleep(10)
                    continue
                else:
                    logger.error(f"豆包 API 请求解析异常，重试后仍失败: {e}")
                    raise
            elif is_timeout:
                # 超时类错误：用更长的等待时间重试
                if attempt < max_retries - 1:
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s
                    logger.warning(f"豆包 API 请求超时（第 {attempt+1} 次，耗时 {elapsed:.1f}s）: {e}")
                    logger.warning(f"等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"豆包 API 请求在 {max_retries} 次重试后仍超时: {e}")
                    raise
            else:
                # 其他异常：标准指数退避
                if attempt < max_retries - 1:
                    wait = 5 ** attempt
                    logger.warning(f"豆包 API 请求异常（第 {attempt+1} 次，耗时 {elapsed:.1f}s）: {e}")
                    logger.warning(f"等待 {wait} 秒后重试...")
                    time.sleep(wait)
                    continue
                else:
                    raise


        elapsed = time.time() - start_time
        logger.info(f"豆包 API 返回成功，耗时 {elapsed:.1f}s")

        # 提取回复文本
        md_content = ""
        for item in response.output:
            if item.type == "message":
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        md_content = content_item.text
                        break
                break

        # 检查回复内容是否为空
        if not md_content or not md_content.strip():
            logger.warning(f"豆包 API 返回内容为空（第 {attempt+1} 次），output_tokens={response.usage.output_tokens if response.usage else 'N/A'}，但文本内容为空")
            if attempt < max_retries - 1:
                wait = 5 ** attempt
                logger.warning(f"等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            else:
                logger.error(f"豆包 API 在 {max_retries} 次重试后仍返回空内容")
                # 继续执行，保存空内容到文件，让上层逻辑处理
        else:
            # 内容非空，跳出重试循环
            break

    # 构建保存路径（使用 sub_dir 隔离不同问卷的输出）
    if sub_dir:
        batch_dir = os.path.join(output_root, sub_dir, "doubao_batches", f"batch_{batch.batch_index:03d}")
    else:
        batch_dir = os.path.join(output_root, "doubao_batches", f"batch_{batch.batch_index:03d}")
    os.makedirs(batch_dir, exist_ok=True)

    md_filename = f"batch_{batch.batch_index:03d}_response.md"
    md_path = os.path.join(batch_dir, md_filename)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 构建元数据（response 在 try 块中赋值，若全部重试失败则已 raise，此处 response 一定存在）
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "batch_index": batch.batch_index,
        "sub_dir": sub_dir,
        "model": response.model,
        "image_paths": batch.image_paths,
        "image_count": len(batch.image_paths),
        "is_first": batch.is_first,
        "is_last": batch.is_last,
        "output": {
            "text_file": md_path,
            "text_preview": md_content[:200] + "..." if len(md_content) > 200 else md_content,
            "text_length": len(md_content),
        },
        "usage": {
            "input_tokens": response.usage.input_tokens if response.usage else 0,
            "output_tokens": response.usage.output_tokens if response.usage else 0,
            "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens
                if response.usage and response.usage.output_tokens_details else None,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        },
        "response_id": response.id if response.id else "",
        "status": response.status if response.status else "completed",
        "elapsed_seconds": round(elapsed, 2),
        "retry_count": attempt,
    }


    meta_filename = f"batch_{batch.batch_index:03d}_meta.json"
    meta_path = os.path.join(batch_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"豆包回复已存档: {md_path}")
    logger.info(f"元数据已存档: {meta_path}")

    return md_content, md_path, metadata

