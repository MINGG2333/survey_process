"""
图片加载与批处理划分模块
"""

import os
import base64
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BatchInfo:
    """描述一批图片的信息"""
    batch_index: int       # 批次编号（从0开始）
    image_paths: List[str] # 本批包含的图片路径列表
    is_first: bool = False # 是否是第一批
    is_last: bool = False  # 是否是最后一批


def list_survey_dirs(survey_root: str) -> List[str]:
    """
    列举 survey_root 下的所有子目录名（每个子目录视为一套独立的问卷）

    Args:
        survey_root: survey 根目录

    Returns:
        排序后的子目录名列表
    """
    dirs = []
    for entry in sorted(os.listdir(survey_root)):
        entry_path = os.path.join(survey_root, entry)
        if os.path.isdir(entry_path):
            dirs.append(entry)
    return dirs


def list_survey_images(survey_root: str, survey_dir: str) -> List[str]:
    """
    列举指定子目录下的所有图片文件，按序号排序

    Args:
        survey_root: survey 根目录
        survey_dir: 子目录名

    Returns:
        排序后的图片路径列表
    """
    target_dir = os.path.join(survey_root, survey_dir)
    if not os.path.isdir(target_dir):
        raise ValueError(f"目录不存在: {target_dir}")
    return _sorted_images_in_dir(target_dir)


def _sorted_images_in_dir(directory: str) -> List[str]:
    """
    对目录中的图片按文件名中的序号排序

    文件名格式示例：interview_protocol_CN_01.png、шо┐ш░ИхНПшоо_хЫЮхдН_23.png
    取最后一个数字后缀作为排序依据
    """
    images = []
    for f in os.listdir(directory):
        fpath = os.path.join(directory, f)
        if os.path.isfile(fpath) and _is_image_file(f):
            images.append(fpath)

    # 按文件名中的数字序号排序（取最后一个数字段）
    def sort_key(fpath):
        fname = os.path.basename(fpath)
        # 提取所有数字段，取最后一个
        nums = re.findall(r'(\d+)', fname)
        if nums:
            return int(nums[-1])
        return 0

    images.sort(key=sort_key)
    return images


def _is_image_file(filename: str) -> bool:
    """判断是否为图片文件"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def split_into_batches(
    image_paths: List[str],
    batch_size: int,
    overlap: int,
) -> List[BatchInfo]:
    """
    将图片列表划分为批，每批之间保持 overlap 张图片的重叠

    Args:
        image_paths: 排序后的图片路径列表
        batch_size: 每批的图片数量
        overlap: 相邻批次之间的重叠图片数量

    Returns:
        批次信息列表
    """
    if overlap >= batch_size:
        raise ValueError(f"overlap({overlap}) 必须小于 batch_size({batch_size})")
    if not image_paths:
        return []

    total = len(image_paths)
    batches = []
    start = 0
    batch_idx = 0

    while start < total:
        end = min(start + batch_size, total)
        batch_paths = image_paths[start:end]
        is_first = (batch_idx == 0)
        is_last = (end >= total)

        batches.append(BatchInfo(
            batch_index=batch_idx,
            image_paths=batch_paths,
            is_first=is_first,
            is_last=is_last,
        ))

        # 下一批的起始位置：往前移动 batch_size - overlap
        start += batch_size - overlap
        batch_idx += 1

    return batches


def encode_images_base64(image_paths: List[str]) -> List[Dict]:
    """
    将图片文件编码为 base64，构造 content_list 格式

    Returns:
        适用于 OpenAI Responses API 的 content 列表
    """
    content_list = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_base = base64.b64encode(f.read()).decode("utf-8")
        content_list.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img_base}"
        })
    return content_list
