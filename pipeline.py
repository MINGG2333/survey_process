"""
调查问卷图片处理主流程（Pipeline）

流程说明：
1. 加载配置
2. 扫描 survey_raw 下的所有子目录，每个子目录视为一套独立的问卷
3. 对每套问卷：
   a. 扫描该目录下的所有图片，排序
   b. 按 batch_size 和 overlap 划分为多批
   c. 每批调用豆包视觉大模型 -> 得到问答表
   d. 相邻两批调用 DeepSeek 进行去冗余和格式统一
   e. 合并所有去冗余后的问答表得到该问卷的完整结果
4. 每套问卷独立输出一个问答表文件
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger

from config_loader import load_config
from image_loader import list_survey_dirs, list_survey_images, split_into_batches
from doubao_client import call_doubao
from deepseek_client import call_deepseek_dedup
from table_merger import merge_tables, save_merged_table


def setup_logger(output_root: str):
    """配置 loguru 日志系统"""
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 移除默认 handler
    logger.remove()

    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
        level="INFO",
        colorize=True,
    )

    # 文件日志（详细）
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        encoding="utf-8",
        rotation="100 MB",
    )

    return log_file


def _load_dedup_cache(dedup_dir: str) -> Optional[Dict[str, str]]:
    """
    从缓存目录加载去冗余结果

    检查 dedup 目录下是否存在完整的三个文件（prev_keep.md, prev_removed.md, next_adjusted.md），
    且内容非空（prev_keep 和 next_adjusted 不能为空）。

    Returns:
        如果缓存有效，返回 dict {prev_keep, prev_removed, next_adjusted}；否则返回 None
    """
    required_files = ["prev_keep.md", "prev_removed.md", "next_adjusted.md"]
    result = {}

    for filename in required_files:
        filepath = os.path.join(dedup_dir, filename)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return None
        result[filename.replace(".md", "")] = content

    # prev_keep 和 next_adjusted 不能为空（空内容说明缓存无效）
    if not result.get("prev_keep", "").strip() or not result.get("next_adjusted", "").strip():
        return None

    return result


def process_single_survey(
    survey_root: str,
    survey_dir_name: str,
    output_root: str,
    config: Dict,
    skip_doubao: bool = False,
    skip_dedup: bool = False,
) -> Optional[str]:
    """
    处理单套问卷的完整流程

    Args:
        survey_root: survey 根目录
        survey_dir_name: 子目录名（问卷标识）
        output_root: 输出根目录
        config: 配置字典
        skip_doubao: 是否跳过豆包 API
        skip_dedup: 是否跳过去冗余

    Returns:
        合并后的完整问答表 Markdown 文本，失败返回 None
    """
    batch_size = config["batch"]["size"]
    overlap = config["batch"]["overlap"]

    logger.info(f"\n{'=' * 60}")

    logger.info(f"开始处理问卷: {survey_dir_name}")
    logger.info(f"{'=' * 60}")

    # Step 1: 扫描该问卷的图片
    logger.info(f"[{survey_dir_name}] Step 1: 扫描图片...")
    images = list_survey_images(survey_root, survey_dir_name)

    if not images:
        logger.warning(f"[{survey_dir_name}] 目录中未找到图片，跳过")
        return None

    logger.info(f"[{survey_dir_name}] 共发现 {len(images)} 张图片")

    # Step 2: 划分批次
    logger.info(f"[{survey_dir_name}] Step 2: 划分批次...")
    batches = split_into_batches(images, batch_size, overlap)
    logger.info(f"[{survey_dir_name}] 共划分为 {len(batches)} 批")
    for b in batches:
        img_names = [os.path.basename(p) for p in b.image_paths]
        logger.debug(f"  [{survey_dir_name}] 批次 {b.batch_index}: {len(b.image_paths)} 张图 - {img_names[0]}...{img_names[-1]}")

    # Step 3: 每批调用豆包
    logger.info(f"[{survey_dir_name}] Step 3: 调用豆包视觉大模型处理每批图片...")
    batch_tables = []

    for batch in batches:
        # 先检查是否有可用的缓存（内容非空）
        batch_dir = os.path.join(output_root, survey_dir_name, "doubao_batches", f"batch_{batch.batch_index:03d}")
        md_path = os.path.join(batch_dir, f"batch_{batch.batch_index:03d}_response.md")
        cached_content = None
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                cached_content = f.read().strip()

        if skip_doubao:
            # --skip-doubao 模式：只从缓存加载
            if cached_content is not None:
                batch_tables.append(cached_content)
                logger.info(f"[{survey_dir_name}] 从缓存加载批次 {batch.batch_index}: {md_path}")
            else:
                logger.warning(f"[{survey_dir_name}] 批次 {batch.batch_index} 的存档不存在: {md_path}")
                batch_tables.append("")
        elif cached_content:
            # 非 skip 模式但缓存存在且非空：跳过 API 调用，直接使用缓存
            batch_tables.append(cached_content)
            logger.info(f"[{survey_dir_name}] 批次 {batch.batch_index} 缓存有效，跳过 API 调用")
        else:
            # 无有效缓存：调用 API
            try:
                md_content, md_path, metadata = call_doubao(batch, config, sub_dir=survey_dir_name)
                batch_tables.append(md_content)
                logger.info(f"[{survey_dir_name}] 批次 {batch.batch_index} 处理完成")
            except Exception as e:
                logger.error(f"[{survey_dir_name}] 批次 {batch.batch_index} 处理失败: {e}")
                logger.warning(f"[{survey_dir_name}] 批次 {batch.batch_index} 使用空表继续")
                batch_tables.append("")
                raise e


    logger.info(f"[{survey_dir_name}] 豆包处理完成，共得到 {len(batch_tables)} 份问答表")

    # Step 4: 去冗余与格式统一
    logger.info(f"[{survey_dir_name}] Step 4: 去冗余与格式统一（调用 DeepSeek）...")
    final_tables = []

    if len(batches) == 1:
        if batch_tables:
            final_tables.append(batch_tables[0])
        logger.info(f"[{survey_dir_name}] 只有一批，无需去冗余")
    elif skip_dedup:
        logger.info(f"[{survey_dir_name}] 跳过 DeepSeek 去冗余，直接拼接")
        final_tables = batch_tables
    else:
        dedup_results = []

        for i in range(len(batches) - 1):
            prev_table = batch_tables[i]
            next_table = batch_tables[i + 1]

            if not prev_table or not next_table:
                logger.warning(f"[{survey_dir_name}] 批次 {i} 或 {i+1} 的表格为空，跳过去冗余")
                # 即使跳过，也要用 dict 格式占位，保证后续循环能统一处理
                dedup_results.append({
                    "prev_keep": prev_table if prev_table else "",
                    "prev_removed": "",
                    "next_adjusted": next_table if next_table else "",
                })
                continue

            # 检查是否有可用的去冗余缓存
            dedup_dir = os.path.join(output_root, survey_dir_name, "dedup", f"batch_{i:03d}_to_{i+1:03d}")
            cached_result = _load_dedup_cache(dedup_dir)
            if cached_result is not None:
                dedup_results.append(cached_result)
                logger.info(
                    f"[{survey_dir_name}] 从缓存加载去冗余结果（batch_{i}->{i+1}）: "
                    f"前一批保留={len(cached_result.get('prev_keep', ''))}字符, "
                    f"移除={len(cached_result.get('prev_removed', ''))}字符, "
                    f"后一批调整={len(cached_result.get('next_adjusted', ''))}字符"
                )
                continue

            try:
                result, metadata = call_deepseek_dedup(
                    prev_table=prev_table,
                    next_table=next_table,
                    prev_batch_index=i,
                    config=config,
                    sub_dir=survey_dir_name,
                )
                dedup_results.append(result)
                logger.info(
                    f"[{survey_dir_name}] 去冗余结果: "
                    f"前一批保留={len(result.get('prev_keep', ''))}字符, "
                    f"移除={len(result.get('prev_removed', ''))}字符, "
                    f"后一批调整={len(result.get('next_adjusted', ''))}字符"
                )
            except Exception as e:
                logger.error(f"[{survey_dir_name}] 批次 {i}->{i+1} 去冗余失败: {e}")
                logger.warning(f"[{survey_dir_name}] 使用原始表格继续（可能含冗余）")
                dedup_results.append({
                    "prev_keep": prev_table,
                    "prev_removed": "",
                    "next_adjusted": next_table,
                })


        # 从 dedup_results 重建 final_tables
        # dedup_results[i] 对应 batch_i 和 batch_{i+1} 的去冗余结果
        # 取第一个结果的 prev_keep 作为 batch_0，然后每个结果的 next_adjusted 作为后续批次
        for i, result in enumerate(dedup_results):
            if i == 0:
                final_tables.append(result.get("prev_keep", batch_tables[i]) if isinstance(result, dict) else batch_tables[i])
            final_tables.append(result.get("next_adjusted", batch_tables[i + 1]) if isinstance(result, dict) else batch_tables[i + 1])


        if not final_tables:
            final_tables = batch_tables

    logger.info(f"[{survey_dir_name}] 去冗余处理完成，共得到 {len(final_tables)} 份待合并的问答表")

    # Step 5: 合并
    logger.info(f"[{survey_dir_name}] Step 5: 合并表格...")
    survey_output_dir = os.path.join(output_root, survey_dir_name)
    os.makedirs(survey_output_dir, exist_ok=True)

    merged_output = os.path.join(survey_output_dir, "merged_qa_table.md")
    merged = save_merged_table(final_tables, merged_output)

    json_output = os.path.join(survey_output_dir, "merged_qa_table.json")
    _save_as_json(merged, json_output)

    logger.info(f"[{survey_dir_name}] 问卷处理完成！")
    logger.info(f"  问答表: {merged_output}")
    logger.info(f"  JSON:   {json_output}")
    logger.info(f"  图片数: {len(images)}, 批次数: {len(batches)}")

    return merged


def run_pipeline(
    survey_dir: Optional[str] = None,
    skip_doubao: bool = False,
    skip_dedup: bool = False,
    force: bool = False,
    config_path: str = "config.yaml",
):
    """
    运行完整处理流程

    遍历 survey_raw 下的每个子目录（每套问卷），独立处理并输出各自的问答表。
    默认跳过已生成合并结果的问卷，使用 --force 可强制重新处理。

    Args:
        survey_dir: 可选，指定只处理 survey_raw 下的某个子目录
        skip_doubao: 如果为 True，则跳过豆包 API 调用（用于调试/重跑去冗余和合并）
        skip_dedup: 如果为 True，则跳过 DeepSeek 去冗余（用于调试）
        force: 如果为 True，则强制重新处理所有问卷（忽略已有结果）
        config_path: 配置文件路径
    """
    # 加载配置
    config = load_config(config_path)
    survey_root = config["paths"]["survey_root"]
    output_root = config["paths"]["output_root"]
    batch_size = config["batch"]["size"]
    overlap = config["batch"]["overlap"]

    # 设置日志
    log_file = setup_logger(output_root)
    logger.info("=" * 60)
    logger.info("调查问卷图片处理管道启动")
    logger.info(f"配置文件: {config_path}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"批大小: {batch_size}, 重叠: {overlap}")
    logger.info(f"数据根目录: {survey_root}")
    logger.info("=" * 60)

    # 获取待处理的问卷目录列表
    if survey_dir:
        # 只处理指定的单个目录
        survey_dirs = [survey_dir]
        logger.info(f"指定处理目录: {survey_dir}")
    else:
        survey_dirs = list_survey_dirs(survey_root)
        logger.info(f"共发现 {len(survey_dirs)} 套问卷: {', '.join(survey_dirs)}")

    if not survey_dirs:
        logger.error(f"在 {survey_root} 中未找到任何子目录（问卷）！")
        return

    # 逐套处理问卷
    results = {}  # survey_dir_name -> merged_table or None
    for dir_name in survey_dirs:
        # 默认跳过已生成合并结果的问卷（除非 --force）
        merged_md_path = os.path.join(output_root, dir_name, "merged_qa_table.md")
        if not force and os.path.exists(merged_md_path):
            logger.info(f"[{dir_name}] 已存在合并结果，跳过（使用 --force 可强制重新处理）")
            # 读取已有结果用于统计
            try:
                with open(merged_md_path, "r", encoding="utf-8") as f:
                    results[dir_name] = f.read()
            except Exception:
                results[dir_name] = ""
            continue

        try:
            merged = process_single_survey(
                survey_root=survey_root,
                survey_dir_name=dir_name,
                output_root=output_root,
                config=config,
                skip_doubao=skip_doubao,
                skip_dedup=skip_dedup,
            )
            results[dir_name] = merged
        except Exception as e:
            logger.error(f"处理问卷 {dir_name} 时发生严重错误: {e}")
            logger.exception(e)
            results[dir_name] = None

    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("管道处理完成！")
    logger.info("=" * 60)

    success_count = sum(1 for v in results.values() if v is not None)
    fail_count = sum(1 for v in results.values() if v is None)
    logger.info(f"成功: {success_count} 套, 失败: {fail_count} 套")

    for dir_name, merged in results.items():
        if merged:
            row_estimate = merged.count('|') // 4 - 1
            logger.info(f"  ✅ {dir_name}: {row_estimate} 行数据")
        else:
            logger.info(f"  ❌ {dir_name}: 处理失败")

    logger.info("=" * 60)


def _save_as_json(md_table: str, output_path: str):
    """
    将 Markdown 表格解析为 JSON 并保存

    简化处理：只提取表格行，不做深度解析
    """
    import re

    lines = md_table.strip().split("\n")
    headers = []
    data_rows = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        # 跳过分隔行
        if re.match(r'^\|[\s\-:]+\|', stripped.replace(' ', '')):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        if not headers:
            headers = cells
        else:
            if len(cells) == len(headers):
                row_dict = dict(zip(headers, cells))
                data_rows.append(row_dict)

    json_data = {
        "headers": headers,
        "rows": data_rows,
        "row_count": len(data_rows),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调查问卷图片处理管道")
    parser.add_argument("--survey-dir", type=str, default=None,
                        help="指定处理 survey_raw 下的某个子目录（默认为所有）")
    parser.add_argument("--skip-doubao", action="store_true",
                        help="跳过豆包 API 调用，使用已有缓存")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="跳过 DeepSeek 去冗余，直接拼接")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理所有问卷（默认跳过已完成的）")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="配置文件路径（默认: config.yaml）")

    args = parser.parse_args()

    run_pipeline(
        survey_dir=args.survey_dir,
        skip_doubao=args.skip_doubao,
        skip_dedup=args.skip_dedup,
        force=args.force,
        config_path=args.config,
    )


