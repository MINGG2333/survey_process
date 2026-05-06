"""
表格合并模块

将所有去冗余后的问答表按照批次顺序合并为完整的表格
"""

import os
import re
from typing import List, Dict, Optional
from loguru import logger


def parse_table_from_md(md_text: str) -> List[str]:
    """
    从 Markdown 文本中解析表格，返回所有数据行（含表头）

    解析逻辑：
    1. 找到第一个以 | 开头且包含表头内容的行为表头
    2. 跳过表头分隔行（|---|）
    3. 提取所有以 | 开头的数据行

    Returns:
        数据行列表（每行是原始的 Markdown 行字符串，前导/尾随空格已被去掉）
    """
    lines = md_text.split("\n")
    table_lines = []
    in_table = False
    header_found = False

    for line in lines:
        stripped = line.strip()
        # 跳过空行
        if not stripped:
            if in_table:
                # 表格结束
                break
            continue

        if stripped.startswith("|"):
            # 检查是否是分隔行（|---|）
            if re.match(r'^\|[\s\-:]+\|', stripped.replace(' ', '')):
                continue  # 跳过表头分隔行

            # 这是一个表格行
            if not header_found:
                # 这是表头
                header_found = True
                in_table = True
                table_lines.append(stripped)
            elif in_table:
                table_lines.append(stripped)

    return table_lines


def merge_tables(sorted_tables: List[str]) -> str:
    """
    按顺序合并多个问答表，自动去重

    Args:
        sorted_tables: 按批次顺序排列的 Markdown 表格文本列表

    Returns:
        合并后的完整 Markdown 表格
    """
    if not sorted_tables:
        return ""

    all_rows = []  # [(批次索引, 行字符串), ...] 用于追踪来源
    seen_rows = set()  # 用于去重
    header = None

    for batch_idx, table_text in enumerate(sorted_tables):
        rows = parse_table_from_md(table_text)

        if not rows:
            logger.warning(f"表格 {batch_idx} 为空，跳过")
            continue

        # 提取表头（第一行）
        current_header = rows[0]

        if header is None:
            header = current_header
        elif current_header != header:
            # 如果表头不同，用第一个表头
            logger.warning(f"表格 {batch_idx} 的表头与第一个不同，使用统一表头")
            # 仍然保留数据行但使用第一个表头

        # 添加数据行（跳过表头）
        for row in rows[1:]:
            # 使用行内容作为唯一标识（去重用）
            row_key = row.strip()
            if row_key not in seen_rows:
                seen_rows.add(row_key)
                all_rows.append((batch_idx, row))

    if header is None:
        return ""

    # 按批次索引排序
    all_rows.sort(key=lambda x: x[0])

    # 拼接最终表格
    result_lines = [header]
    # 添加分隔行
    sep_parts = header.strip("|").split("|")
    sep_line = "|" + "|".join(["---"] * len(sep_parts)) + "|"
    result_lines.append(sep_line)

    for _, row in all_rows:
        result_lines.append(row)

    return "\n".join(result_lines)


def save_merged_table(tables: List[str], output_path: str) -> str:
    """
    合并并保存完整表格

    Args:
        tables: 按批次顺序排列的 Markdown 表格文本列表
        output_path: 输出文件路径

    Returns:
        合并后的完整表格文本
    """
    merged = merge_tables(tables)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)

    logger.info(f"完整问答表已保存到: {output_path}")
    logger.info(f"表格包含 {merged.count('|') // 4 - 1} 行数据（估算）")

    return merged
