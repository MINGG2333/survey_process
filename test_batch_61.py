#!/usr/bin/env python3
"""测试61张图片的批次划分"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from image_loader import split_into_batches

# 模拟 шо┐ш░ИхНПшоо_хЫЮхдН 的 61 张图片 (编号23-83)
fake_images = [f"page_{i:02d}.png" for i in range(23, 84)]
print(f"总图片数: {len(fake_images)}")  # 61

batches = split_into_batches(fake_images, batch_size=10, overlap=1)
print(f"批次数: {len(batches)}")
print()
for b in batches:
    first = os.path.basename(b.image_paths[0])
    last = os.path.basename(b.image_paths[-1])
    flags = ""
    if b.is_first: flags += " [FIRST]"
    if b.is_last: flags += " [LAST]"
    print(f"  批次{b.batch_index}: {len(b.image_paths):2d}张  {first} ~ {last}{flags}")

# 验证覆盖
covered = set()
for b in batches:
    for img in b.image_paths:
        covered.add(img)
assert len(covered) == 61, f"应覆盖61张，实际{len(covered)}"
print(f"\n全部{len(covered)}张图片均已覆盖 ✓")

# 验证重叠
for i in range(len(batches)-1):
    prev = set(batches[i].image_paths)
    nxt = set(batches[i+1].image_paths)
    overlap = len(prev & nxt)
    assert overlap == 1, f"批次{i}->{i+1}重叠应为1，实际{overlap}"
print("批次间重叠均为1 ✓")
