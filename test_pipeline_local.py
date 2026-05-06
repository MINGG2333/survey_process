#!/usr/bin/env python3
"""
本地测试脚本（不调用外部 API）

测试模块：
1. image_loader - 图片扫描与批次划分
2. table_merger - 表格解析与合并
3. prompts - 提示词
4. config_loader - 配置加载
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_loader import load_config
from image_loader import list_survey_images, split_into_batches, BatchInfo
from table_merger import parse_table_from_md, merge_tables


def test_config_loader():
    """测试配置加载"""
    print("=" * 60)
    print("[TEST] config_loader")
    print("=" * 60)

    config = load_config("config.yaml")
    assert config["batch"]["size"] == 10, "batch size should be 10"
    assert config["batch"]["overlap"] == 1, "overlap should be 1"
    assert "doubao" in config, "doubao config should exist"
    assert "deepseek" in config, "deepseek config should exist"
    print(f"  ✓ 配置加载成功: batch_size={config['batch']['size']}, overlap={config['batch']['overlap']}")
    print()


def test_image_loader_batching():
    """测试批次划分"""
    print("=" * 60)
    print("[TEST] image_loader - 批次划分")
    print("=" * 60)

    # 模拟83张图片 (如шо┐ш░ИхНПшоо_хЫЮхдН目录)
    fake_images = [f"/fake/path/img_{i:02d}.png" for i in range(1, 84)]

    # 测试 batch_size=10, overlap=1
    batches = split_into_batches(fake_images, batch_size=10, overlap=1)
    print(f"  83张图, batch_size=10, overlap=1 → {len(batches)} 批")
    for b in batches:
        img_range = f"[{os.path.basename(b.image_paths[0])} ... {os.path.basename(b.image_paths[-1])}]"
        count = f"{len(b.image_paths)}张"
        flags = ""
        if b.is_first:
            flags += " [第一批]"
        if b.is_last:
            flags += " [最后一批]"
        print(f"    批次 {b.batch_index}: {count} {img_range}{flags}")

    # 验证：
    # 1. 第一批有 batch_size 张
    assert len(batches[0].image_paths) == 10, f"第一批应该有10张，实际{len(batches[0].image_paths)}"
    # 2. 相邻批次之间重叠 overlap 张
    for i in range(len(batches) - 1):
        prev = batches[i].image_paths
        nxt = batches[i + 1].image_paths
        overlap_count = len(set(prev) & set(nxt))
        assert overlap_count == 1, f"批次{i}与批次{i+1}应重叠1张，实际{overlap_count}"
    print(f"  ✓ 批次划分正确")
    print()

    # 测试 batch_size=5, overlap=2
    batches2 = split_into_batches(fake_images, batch_size=5, overlap=2)
    print(f"  83张图, batch_size=5, overlap=2 → {len(batches2)} 批")
    for b in batches2[:3]:
        print(f"    批次 {b.batch_index}: {len(b.image_paths)}张")
    print(f"    ...")
    for b in batches2[-2:]:
        print(f"    批次 {b.batch_index}: {len(b.image_paths)}张")
    assert len(batches2[0].image_paths) == 5
    for i in range(len(batches2) - 1):
        prev = set(batches2[i].image_paths)
        nxt = set(batches2[i + 1].image_paths)
        assert len(prev & nxt) == 2, f"应重叠2张"
    print(f"  ✓ overlap=2 正确")
    print()

    # 测试边界情况：单张图
    single = split_into_batches(["img_1.png"], batch_size=10, overlap=1)
    assert len(single) == 1
    assert len(single[0].image_paths) == 1
    print(f"  ✓ 单张图片边界正确")
    print()


def test_table_merger():
    """测试表格解析与合并"""
    print("=" * 60)
    print("[TEST] table_merger")
    print("=" * 60)

    # 模拟两张问答表
    table1 = """| 问题类型 | 问题编号 | 问题内容 | 答案内容 |
|----------|----------|----------|----------|
| 基本信息采集 | Q1 | 问题1内容 | 答案1 |
| 基本信息采集 | Q2 | 问题2内容 | 答案2 |
| 网络安全风险 | Q3 | 问题3内容 | 答案3 |"""

    table2 = """| 问题类型 | 问题编号 | 问题内容 | 答案内容 |
|----------|----------|----------|----------|
| 网络安全风险 | Q3 | 问题3内容 | 答案3 |
| 网络安全风险 | Q4 | 问题4内容 | 答案4 |
| 网络安全风险 | Q5 | 问题5内容 | 答案5 |"""

    # 测试 parse_table_from_md
    rows1 = parse_table_from_md(table1)
    print(f"  table1 解析: {len(rows1)} 行 (含表头)")
    for r in rows1:
        print(f"    {r}")
    assert len(rows1) == 4, f"应为4行（1表头+3数据），实际{len(rows1)}"

    rows2 = parse_table_from_md(table2)
    assert len(rows2) == 4
    print()

    # 测试 merge_tables（含去重）
    merged = merge_tables([table1, table2])
    print(f"  合并后表格:")
    for line in merged.split("\n"):
        print(f"    {line}")

    merged_rows = parse_table_from_md(merged)
    # 表头 + 5个不重复数据行
    assert len(merged_rows) == 6, f"应为6行（1表头+5数据，Q3去重），实际{len(merged_rows)}"
    print(f"  ✓ 表格解析与合并正确（Q3已去重）")
    print()

    # 测试空表格
    empty_merged = merge_tables([])
    assert empty_merged == ""
    print(f"  ✓ 空表格合并边界正确")
    print()


def test_prompts():
    """测试提示词模板"""
    print("=" * 60)
    print("[TEST] prompts")
    print("=" * 60)
    from prompts import DOUBAO_USER_PROMPT_TEMPLATE, DOUBAO_SYSTEM_PROMPT

    prompt = DOUBAO_USER_PROMPT_TEMPLATE.format(total=10)
    print(f"  Doubao user prompt (10张图):")
    print(f"    {prompt[:80]}...")
    assert "{total}" not in prompt, "模板变量未被替换"
    assert "10" in prompt, "total 参数未正确嵌入"
    print(f"  ✓ 提示词模板正确")

    from prompts import DEEPSEEK_DEDUP_SYSTEM_PROMPT, DEEPSEEK_DEDUP_USER_PROMPT_TEMPLATE
    dedup_prompt = DEEPSEEK_DEDUP_USER_PROMPT_TEMPLATE.format(
        prev_table="prev table",
        next_table="next table",
    )
    assert "{prev_table}" not in dedup_prompt
    assert "{next_table}" not in dedup_prompt
    assert "prev table" in dedup_prompt
    print(f"  ✓ DeepSeek 去冗余提示词模板正确")
    print()


def test_real_image_scan():
    """测试实际的图片扫描"""
    print("=" * 60)
    print("[TEST] 真实图片扫描")
    print("=" * 60)

    survey_root = "survey_raw"
    if not os.path.exists(survey_root):
        print(f"  ⚠ {survey_root} 目录不存在，跳过")
        return

    images = list_survey_images(survey_root)
    print(f"  发现 {len(images)} 张图片")

    # 按目录分组
    dirs = {}
    for img in images:
        d = os.path.basename(os.path.dirname(img))
        dirs.setdefault(d, []).append(img)
    for dir_name, imgs in dirs.items():
        print(f"  {dir_name}: {len(imgs)} 张")
        if imgs:
            print(f"    首张: {os.path.basename(imgs[0])}")
            print(f"    末张: {os.path.basename(imgs[-1])}")

    assert len(images) > 0, "应至少找到一些图片"
    print(f"  ✓ 图片扫描成功")
    print()


def test_batch_simulation():
    """模拟完整批次划分并验证覆盖率和重叠"""
    print("=" * 60)
    print("[TEST] 完整批次模拟")
    print("=" * 60)

    # 模拟 шо┐ш░ИхНПшоо_хЫЮхдН 目录 83 张图
    total_images = 83
    fake_images = [f"page_{i:02d}.png" for i in range(1, total_images + 1)]

    batch_size = 10
    overlap = 1

    batches = split_into_batches(fake_images, batch_size, overlap)

    # 检查所有图片都被覆盖
    covered = set()
    for b in batches:
        for img in b.image_paths:
            covered.add(img)
    assert len(covered) == total_images, f"应覆盖所有{total_images}张图，实际覆盖{len(covered)}张"

    # 检查每张图片至少出现在一批中
    uncovered = set(fake_images) - covered
    assert len(uncovered) == 0, f"有{len(uncovered)}张图未被任何批次覆盖"

    print(f"  83张图 → {len(batches)}批 (batch_size={batch_size}, overlap={overlap})")
    print(f"  所有图片均被覆盖 ✓")

    # 验证批次间重叠
    total_overlap = 0
    for i in range(len(batches) - 1):
        prev_set = set(batches[i].image_paths)
        next_set = set(batches[i + 1].image_paths)
        overlap_count = len(prev_set & next_set)
        total_overlap += overlap_count
        assert overlap_count == overlap, f"批次{i}与{i+1}重叠应为{overlap}，实际{overlap_count}"
    print(f"  批次间重叠均为 {overlap} ✓")

    # 验证总批次数
    # 公式：ceil((total - batch_size) / (batch_size - overlap)) + 1
    import math
    expected_batches = math.ceil((total_images - batch_size) / (batch_size - overlap)) + 1
    assert len(batches) == expected_batches, f"批次数应为{expected_batches}，实际{len(batches)}"
    print(f"  批次数正确: {len(batches)} ✓")
    print()


if __name__ == "__main__":
    test_config_loader()
    test_image_loader_batching()
    test_table_merger()
    test_prompts()
    test_real_image_scan()
    test_batch_simulation()

    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
