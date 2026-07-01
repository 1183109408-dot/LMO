import openai
import random
import re
import os
import csv
import time
import requests
import json
from scienceQA_load import load_and_process_json  # 导入 ScienceQA 数据加载函数

# 设置远程 API 密钥和基础 URL（用于生成候选指令）
def api(content, retries=5, delay=5):
    openai.api_key = "sk-80b99456e5cc40439a37f02a0003f51e"
    openai.api_base = "https://api.deepseek.com/v1"
    for attempt in range(retries):
        try:
            chat_completion = openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": content}]
            )
            return chat_completion["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API 错误: {e}. 重试 {attempt + 1}/{retries}...")
            time.sleep(delay)
    print("API 调用失败，返回空响应")
    return "API错误"

# 本地 Llama 3.1 8B 模型调用类（用于指令评分）
class LocalLlama31_8B:
    def __init__(self, model_name="llama3.1:8b", host="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{host}/api/generate"
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt, max_tokens=300, temperature=0.2):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        for _ in range(5):
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload))
                response.raise_for_status()
                result = response.json()
                return result.get("response", "Error: No response text found")
            except requests.exceptions.RequestException as e:
                print(f"本地 Llama API 调用失败: {e}. 重试中...")
                time.sleep(5)
        print("本地 Llama API 调用失败，返回空响应")
        return "本地 Llama API 错误"

# 初始化本地模型
llama = LocalLlama31_8B()

def instruct_generate(small_class, K, num_instructions=20):
    """生成候选指令，使用远程 API"""
    instructions = []
    sample_weights = [0.0] * len(small_class.data)
    
    for _ in range(num_instructions):
        probabilities = [1.0 / (w + 1.0) for w in sample_weights]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        selected_indices = random.choices(range(len(small_class.data)), weights=probabilities, k=K)
        samples = [
            f"Question: {small_class.data[i].quest}\n"
            f"Reasoning Steps: {small_class.data[i].reason}\n"
            f"Answer: {small_class.data[i].answer}"
            for i in selected_indices
        ]
        
        for i in selected_indices:
            sample_weights[i] += 1.0
        
        content = (
            f"Based on the following examples of questions, reasoning steps, and answers, "
            f"provide a general reasoning process for solving {small_class.label} tasks (within 250 words):\n"
            + '\n\n'.join(samples)
        )
        respond = api(content)
        print(f"已生成一条候选指令，长度：{len(respond)}")
        instructions.append(respond)
    
    return instructions

def log_error_to_csv(sample, respond, csv_path=r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'):
    """记录未提取到答案的日志到 err_log.csv"""
    csv_filename = os.path.join(csv_path, 'err_log.csv')
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Question', 'Reasoning Steps', 'Answer', 'Model Output'])
        writer.writerow([sample.quest, sample.reason, sample.answer, respond])

def instruct_remark(small_class, instructions, history_scores, correct_samples):
    """为每条指令和无指令情况评分，使用本地 Llama 模型，并记录正确回答到临时对象"""
    current_remarks = []
    csv_path = r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'
    
    # 测试无指令情况（不记录样本）
    print("无指令")
    remark = 0
    for sample in small_class.data:
        prompt = (
            f"Please answer the following question:\n{sample.quest}\n"
            f"Provide a detailed step-by-step reasoning process.\n"
            f"At the end, conclude with the answer in the exact format '(Correct Answer: X)' where X is 0, 1, 2 or 3.\n"
            f"Ensure the answer is on a new line and matches this format exactly."
        )
        respond = llama.generate(prompt)
        answer = extract_answer(respond)
        real_answer = str(sample.answer)
        print(f"样本真实答案：{real_answer}，本地 Llama 输出答案：{answer if answer else '未提取到'}")
        if answer is None:
            log_error_to_csv(sample, respond, csv_path)
        elif answer == real_answer:
            remark += 1
    current_remarks.append(('No Instruction', remark))
    history_scores['No Instruction'] = history_scores.get('No Instruction', []) + [remark]
    
    # 测试每条指令，记录正确样本
    for instruction in instructions:
        print(f"正在评估指令: {instruction[:50]}...")
        remark = 0
        for sample in small_class.data:
            prompt = (
                f"{instruction}\n"
                f"Please answer the following question based on the above guidance:\n{sample.quest}\n"
                f"Provide a detailed step-by-step reasoning process.\n"
                f"At the end, conclude with the answer in the exact format '(Correct Answer: X)' where X is 0, 1, 2 or 3.\n"
                f"Ensure the answer is on a new line and matches this format exactly."
            )
            respond = llama.generate(prompt)
            answer = extract_answer(respond)
            real_answer = str(sample.answer)
            print(f"样本真实答案：{real_answer}，本地 Llama 输出答案：{answer if answer else '未提取到'}")
            if answer is None:
                log_error_to_csv(sample, respond, csv_path)
            elif answer == real_answer:
                remark += 1
                # 临时存储正确样本信息，使用sample.index作为index
                correct_samples.append({
                    'index': sample.index,  # 修改为sample.index
                    'quest': sample.quest,
                    'output': respond,
                    'answer': real_answer,
                    'label': small_class.label,
                    'instruction': instruction
                })
        current_remarks.append((instruction, remark))
        history_scores[instruction] = history_scores.get(instruction, []) + [remark]
    
    return current_remarks

def save_correct_samples_to_csv(correct_samples, history_scores, total_samples, csv_path=r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'):
    """将所有正确样本一次性写入CSV，包含instruction字段，使用sample.index"""
    csv_filename = os.path.join(csv_path, 'correct_samples.csv')
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Index', 'Question', 'Output', 'Answer', 'Label', 'Instruction', 'Score'])
        for sample in correct_samples:
            instruction = sample['instruction']
            avg_score = sum(history_scores[instruction]) / len(history_scores[instruction])
            norm_score = avg_score / total_samples
            writer.writerow([
                sample['index'],  # 使用sample.index
                sample['quest'],
                sample['output'],
                sample['answer'],
                sample['label'],
                sample['instruction'],
                f"{norm_score:.4f}"
            ])

def extract_answer(text):
    """从模型输出中提取答案（0、1 或 2）"""
    match = re.search(r'\(Correct Answer: ([0-2])\)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'[0-2]', text)
    return match.group(0) if match else None

def compute_average_scores(history_scores):
    """计算每条指令的平均分，保留浮点数"""
    avg_scores = {}
    for instr, scores in history_scores.items():
        avg_scores[instr] = sum(scores) / len(scores)
    return avg_scores

def load_best_instructions(csv_path=r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'):
    """读取已有的最佳指令"""
    csv_filename = os.path.join(csv_path, 'best_instructions.csv')
    best_instructions = {}
    if os.path.isfile(csv_filename):
        with open(csv_filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                best_instructions[row['Label']] = row['Instruction']
    return best_instructions

def save_best_instruction(label, instruction, csv_path=r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'):
    """保存最佳指令到 CSV"""
    csv_filename = os.path.join(csv_path, 'best_instructions.csv')
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Label', 'Instruction'])
        writer.writerow([label, instruction])

def save_final_remarks_to_csv(final_instructions, avg_scores, label, csv_path=r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'):
    """保存最终收敛的指令到 CSV，包括无指令，平均分保留两位小数"""
    csv_filename = os.path.join(csv_path, 'instructions.csv')
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Label', 'Instruction', 'Average_Score'])
        writer.writerow([label, "无指令", f"{avg_scores['No Instruction']:.2f}"])
        for instr in final_instructions:
            writer.writerow([label, instr, f"{avg_scores[instr]:.2f}"])

def iterate_instructions(small_class, K, k_percent=0.5, target_n=5):
    """迭代筛选指令，每轮使用上一轮高分指令，直到剩下 n 条，并在类别完成后保存样本"""
    instructions = instruct_generate(small_class, K)  # 初始 20 条指令
    history_scores = {}
    correct_samples = []  # 临时存储正确样本
    iteration = 0
    
    while len(instructions) > target_n:
        print(f"迭代 {iteration + 1}：当前指令数 {len(instructions)}，目标 {target_n}")
        
        # 评分当前指令并记录正确样本
        current_remarks = instruct_remark(small_class, instructions, history_scores, correct_samples)
        
        # 计算平均分并排序
        avg_scores = compute_average_scores(history_scores)
        sorted_remarks = sorted(
            [(instr, avg_scores[instr]) for instr, _ in current_remarks if instr != 'No Instruction'],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 计算保留数量 k
        k = max(1, int(len(instructions) * k_percent))
        if len(instructions) - k < target_n:
            k = len(instructions) - target_n  # 确保最后一步正好达到 target_n
        
        # 选择上一轮高分指令
        instructions = [r[0] for r in sorted_remarks[:k]]
        
        iteration += 1
    
    # 最终评分
    final_remarks = instruct_remark(small_class, instructions, history_scores, correct_samples)
    avg_scores = compute_average_scores(history_scores)
    sorted_final = sorted(
        [(instr, avg_scores[instr]) for instr, _ in final_remarks if instr != 'No Instruction'],
        key=lambda x: x[1],
        reverse=True
    )
    final_instructions = [r[0] for r in sorted_final]
    
    # 保存当前类别的所有正确样本
    total_samples = len(small_class.data)
    save_correct_samples_to_csv(correct_samples, history_scores, total_samples)
    
    # 输出结果
    print(f"收敛：剩余 {len(final_instructions)} 条指令")
    for instr, score in sorted_final:
        print(f"指令: {instr[:50]}... 平均分: {score:.2f}")
    print(f"无指令平均分: {avg_scores['No Instruction']:.2f}")
    
    # 返回最佳指令（得分最高的）
    best_instruction = sorted_final[0][0] if sorted_final else "No Instruction"
    return final_instructions, avg_scores, best_instruction

def main():
    json_path = r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\datas\scienceQA\train4.json'
    csv_path = r'E:\project\OSAI\autoPrompt\code\LearningToCompare_FSL-master\omniglot\instructions'
    big_classes = load_and_process_json(json_path)
    
    # 加载已有的最佳指令
    best_instructions = load_best_instructions(csv_path)
    
    K = 4
    target_n = 5
    for big_class in big_classes:
        for small_class in big_class.data:
            label = small_class.label
            print(f"\n检查 SmallClass: {label} ({len(small_class.data)} 样本)")
            
            # 检查是否已存在最佳指令
            if label in best_instructions:
                print(f"已存在最佳指令，跳过 {label}")
                continue
            
            print(f"处理 SmallClass: {label}")
            final_instructions, avg_scores, best_instruction = iterate_instructions(
                small_class, K, k_percent=0.5, target_n=target_n
            )
            
            # 保存所有指令和分数
            save_final_remarks_to_csv(final_instructions, avg_scores, label, csv_path)
            # 保存最佳指令
            save_best_instruction(label, best_instruction, csv_path)

if __name__ == "__main__":
    main()