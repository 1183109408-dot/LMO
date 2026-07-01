import json
import os
from typing import List, Union


class MathSample:
    def __init__(self, index: str, quest: str, reason: str, answer: str):
        self.index = index      # 样本序号，从 "0" 开始的字符串
        self.quest = quest      # problem
        self.reason = reason    # solution
        self.answer = answer    # 最终答案（通常已包含 \boxed{}）

    def __repr__(self):
        return (f"MathSample(index={self.index}, "
                f"quest={self.quest[:60]}..., "
                f"reason={self.reason[:60]}..., "
                f"answer={self.answer})")


def load_and_process_json(json_path: str = None) -> List[MathSample]:
    """
    加载 MATH-500 风格的 JSON / JSONL 文件（train.json 或 test.json 等）
    要求每条样本必须同时包含 problem、solution、answer 三个字段
    
    - 缺少 answer（或 answer 为空）的样本会被跳过
    - 返回 List[MathSample]
    """
    default_dir = r"E:\project\DPO\data\Math500"
    
    if json_path is None:
        # 如果没传路径，提示用户可能的文件名
        print("未指定路径，使用默认目录：", default_dir)
        print("建议明确传入文件名，例如：load_and_process_json(r'E:\\project\\DPO\\data\\Math500\\train.json')")
        return []

    if not os.path.exists(json_path):
        print(f"文件不存在：{json_path}")
        return []

    print(f"正在读取：{json_path}")

    samples = []
    skipped_count = 0
    line_count = 0

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试判断是 JSONL 还是单一大 JSON 数组
        if content.strip().startswith('['):
            # 是 JSON 数组格式
            try:
                raw_data = json.loads(content)
                iterator = enumerate(raw_data)
            except json.JSONDecodeError as e:
                print(f"JSON 数组解析失败：{e}")
                return []
        else:
            # 假设是 JSONL（每行一个对象）
            raw_lines = content.splitlines()
            iterator = enumerate(raw_lines)

        for i, item_or_line in iterator:
            line_count += 1

            if isinstance(item_or_line, str):
                line = item_or_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"第 {i+1} 行 JSON 解析失败，跳过")
                    continue
            else:
                # 已经是 dict（从 JSON 数组中来）
                item = item_or_line

            # 字段提取（兼容常见变体）
            problem = item.get("problem") or item.get("question", "")
            solution = item.get("solution", "")
            answer_raw = item.get("answer", "")

            answer = str(answer_raw).strip()

            # 必须同时有三个字段，且 answer 非空
            if not (problem.strip() and solution.strip() and answer):
                skipped_count += 1
                continue

            sample = MathSample(
                index=str(i),
                quest=problem.strip(),
                reason=solution.strip(),
                answer=answer
            )
            samples.append(sample)

    except Exception as e:
        print(f"读取/处理文件出错：{str(e)}")
        return []

    # 统计信息
    total_processed = line_count if 'raw_lines' in locals() else len(raw_data)
    print(f"文件中共处理 {total_processed:,} 条记录")
    print(f"成功加载 {len(samples):,} 条完整样本")
    if skipped_count > 0:
        print(f"跳过 {skipped_count} 条样本（缺少 problem / solution / answer 中的至少一项）")

    return samples


# ============================== 测试 / 使用示例 ==============================
def main():
    base_dir = r"E:\project\DPO\data\Math500"
    
    for name in ["train", "test"]:
        # 尝试常见文件名
        possible_files = [
            os.path.join(base_dir, f"{name}.json"),
            os.path.join(base_dir, f"{name}.jsonl"),
        ]
        
        json_path = None
        for p in possible_files:
            if os.path.exists(p):
                json_path = p
                break
        
        if not json_path:
            print(f"\n找不到 {name} 的 json / jsonl 文件")
            continue

        print("\n" + "="*70)
        print(f"处理 {name.upper()} 数据集")
        print("="*70)
        
        data = load_and_process_json(json_path)
        
        print(f"有效样本数：{len(data)}")
        
        # 打印前 2 条示例
        if data:
            print("\n前 2 条示例：")
            for i, sample in enumerate(data[:2]):
                print(f"\n--- 示例 {i+1}  (index: {sample.index}) ---")
                print(f"Quest : {sample.quest[:120]}{'...' if len(sample.quest)>120 else ''}")
                print(f"Answer: {sample.answer}")
                print(f"Reason（前150字符）: {sample.reason[:150]}{'...' if len(sample.reason)>150 else ''}")


if __name__ == "__main__":
    main()