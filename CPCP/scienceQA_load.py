# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 11:02:40 2025

@author: 86135
"""
import json
import os
from typing import List, Dict

class DataSample:
    def __init__(self, index: str, original_data: Dict, quest: str, reason: str, answer: str):
        self.index = index  # JSON 中的键，例如 "14355"
        self.original_data = original_data  # 保留原始字段
        self.quest = quest  # 合并后的 quest 字段
        self.reason = reason  # 合并后的 reason 字段
        self.answer = answer  # 提取的 answer 字段

    def __repr__(self):
        return f"DataSample(index={self.index}, quest={self.quest[:50]}..., reason={self.reason[:50]}..., answer={self.answer}, original_data={self.original_data})"

class SmallClass:
    def __init__(self, label: str, data: List[DataSample]):
        self.label = label  # skill 值
        self.data = data  # DataSample 对象列表

    def __repr__(self):
        return f"SmallClass(label={self.label}, data={len(self.data)} samples)"

class BigClass:
    def __init__(self, label: str, data: List[SmallClass]):
        self.label = label  # category 值
        self.data = data  # SmallClass 对象列表

    def __repr__(self):
        return f"BigClass(label={self.label}, data={len(self.data)} small_classes)"

def load_and_process_json(json_path: str) -> List[BigClass]:
    """
    读取并处理 JSON 文件，返回按 category 和 skill 层次结构的 BigClass 对象列表。
    """
    # 第一步：加载并转换为 DataSample
    data_samples = []

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"JSON 文件不存在，请检查路径：{json_path}")
        return []

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)

            for key, sample in content.items():
                hint = sample.get("hint", "")
                question = sample.get("question", "")
                choices = sample.get("choices", [])
                lecture = sample.get("lecture", "")
                solution = sample.get("solution", "")
                answer = sample.get("answer", "")

                choices_str = "\n".join([f"{i}. {choice}" for i, choice in enumerate(choices)]) if choices else ""
                quest_parts = [part for part in [hint, question, choices_str] if part]
                quest = "\n".join(quest_parts)

                reason_parts = [part for part in [lecture, solution] if part]
                reason = "\n".join(reason_parts)

                data_sample = DataSample(
                    index=key,  # 保存 JSON 键作为样本序号
                    original_data=sample,
                    quest=quest,
                    reason=reason,
                    answer=answer
                )
                data_samples.append(data_sample)

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError in file {json_path}: {e}")
        with open(json_path, 'r', encoding='utf-8') as f:
            print(f"File content (first 500 characters): {f.read()[:500]}")
        return []
    except Exception as e:
        print(f"Error processing file {json_path}: {e}")
        return []

    # 第二步：按 category 和 skill 组织层次结构
    category_map = {}
    for sample in data_samples:
        category = sample.original_data.get("category", "Unknown")
        skill = sample.original_data.get("skill", "Unknown")

        # 初始化 big_class
        if category not in category_map:
            category_map[category] = {}
        # 初始化 small_class
        if skill not in category_map[category]:
            category_map[category][skill] = []
        # 添加 DataSample 到对应 skill
        category_map[category][skill].append(sample)

    # 第三步：构建 BigClass 和 SmallClass 对象
    big_classes = []
    for category, skill_map in category_map.items():
        small_classes = []
        for skill, samples in skill_map.items():
            small_class = SmallClass(label=skill, data=samples)
            small_classes.append(small_class)
        big_class = BigClass(label=category, data=small_classes)
        big_classes.append(big_class)

    return big_classes

# 主程序（用于测试）
def main():
    input_json_path = r"E:\project\OSAI\autoPrompt\dataset\ScienceQA-main\data\scienceqa\test.json"
    processed_data = load_and_process_json(input_json_path)

    print(f"\n共处理 {len(processed_data)} 个 BigClass (category)")
    total_samples = sum(len(small_class.data) for big_class in processed_data for small_class in big_class.data)
    print(f"总样本数: {total_samples}")

    if processed_data:
        print("\n层次结构示例（前两个 BigClass）：")
        for i, big_class in enumerate(processed_data[:2]):
            print(f"\nBigClass {i+1}:")
            print(f"  Label (category): {big_class.label}")
            print(f"  SmallClasses (skills): {len(big_class.data)} 个")
            for j, small_class in enumerate(big_class.data[:2]):
                print(f"    SmallClass {j+1}:")
                print(f"      Label (skill): {small_class.label}")
                print(f"      Samples: {len(small_class.data)} 条")
                for k, sample in enumerate(small_class.data[:1]):
                    print(f"        Sample {k+1}:")
                    print(f"          Index: {sample.index}")
                    print(f"          Quest: {sample.quest[:50]}...")
                    print(f"          Reason: {sample.reason[:50]}...")
                    print(f"          Answer: {sample.answer}")

if __name__ == "__main__":
    main()