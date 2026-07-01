import os
import torch
import logging
import csv
import re
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from Math500_load import load_and_process_json

# ==================== 100%对齐修正后训练代码的DeepSeek配置 ====================
TEST_JSON_PATH = r"/root/WuYanzu/DPO/data/Math500/test.json"
MODEL_BASE_PATH = r"/root/WuYanzu/DPO/model/deepseek-math-7b-base"

# 对齐训练代码的新保存目录（_fixed后缀）
SAVE_DIR = r"/root/WuYanzu/DPO/checkpoints/checkpoints_Math500_deepseek_lora_base"
OUTPUT_DIR = r"/root/WuYanzu/DPO/test_Math500_deepseek_lora_base"

MAX_NEW_TOKENS = 1024
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动加载最新 epoch
adapter_dirs = [d for d in os.listdir(SAVE_DIR) if d.startswith("adapter_epoch_") and os.path.isdir(os.path.join(SAVE_DIR, d))]
if not adapter_dirs:
    raise FileNotFoundError(f"No adapter found in {SAVE_DIR}")
latest_adapter = max(adapter_dirs, key=lambda x: int(x.split("_")[-1]))
ADAPTER_DIR = os.path.join(SAVE_DIR, latest_adapter)
print(f"使用最新训练参数: {ADAPTER_DIR}")

# ==================== 日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "test_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== 停止条件（完全不变） ====================
class StopOnMathAnswer(StoppingCriteria):
    def __init__(self, tokenizer, input_length):
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) <= self.input_length:
            return False
        new_tokens = input_ids[0, self.input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return bool(
            re.search(r'\\boxed\{.*?\}', decoded) or
            re.search(r'Answer:', decoded, re.IGNORECASE) or
            re.search(r'Final answer', decoded, re.IGNORECASE)
        )

# ==================== 答案提取 & 标准化（完全不变） ====================
def extract_answer_from_response(text):
    if not text:
        return None
    
    boxed_pattern = r'\\boxed\s*\{'
    match = re.search(boxed_pattern, text)
    if not match:
        ans_match = re.search(r'(?:Answer|Final answer|answer|Answer:)\s*[:=]?\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if ans_match:
            answer = ans_match.group(1).strip()
            answer = re.sub(r'[.。!！?？\n]*$', '', answer).strip()
            return answer if answer else None
        return None

    start_pos = match.end() - 1
    brace_count = 1
    i = start_pos + 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        boxed_content = text[start_pos + 1 : i - 1].strip()
        boxed_content = re.sub(r'^\$|\$$', '', boxed_content).strip()
        return boxed_content
    
    simple_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if simple_match:
        return simple_match.group(1).strip()
    
    return None

def normalize_math(ans):
    if not ans:
        return ""
    s = str(ans).strip()
    s = s.replace('\\\\', '\\')
    s = s.replace('\\pi', 'pi').replace('\\Pi', 'pi')
    s = s.replace('\\circ', 'circ').replace('°', 'circ')
    s = re.sub(r'\^\s*\{([^}]+)\}', r'^\1', s)
    s = re.sub(r'\\frac\s*\{?([^}{]+)\}?\s*\{?([^}{]+)\}?', r'\1/\2', s)
    s = re.sub(r'\\dfrac\s*\{?([^}{]+)\}?\s*\{?([^}{]+)\}?', r'\1/\2', s)
    s = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'√\1', s)
    s = re.sub(r'\\sqrt\s*([^\s\\{}]+)', r'√\1', s)
    s = re.sub(r'\\left\s*([(\[]?)', r'\1', s)
    s = re.sub(r'\\right\s*([)\]]?)', r'\1', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', s)
    s = re.sub(r'([0-9.])(circ)', r'\1^\2', s)
    s = s.replace('\\', '')
    s = re.sub(r'\s+', '', s)
    s = s.lower()
    s = s.replace('()', '').replace('[]', '').replace('{}', '')
    return s.strip()

# 严格相等匹配（与MathQA标准一致）
def is_correct_match(true_norm, pred_norm):
    if not true_norm or not pred_norm:
        return False
    return true_norm == pred_norm

# ==================== 主测试函数（100%对齐训练代码） ====================
def test_math500():
    logger.info(f"Starting MATH-500 test with DeepSeek-Math-7B-Base: {latest_adapter}")
    test_samples = load_and_process_json(TEST_JSON_PATH)
    if not test_samples:
        raise ValueError("Failed to load MATH-500 test data.")
    total_samples = len(test_samples)
    logger.info(f"Loaded {total_samples} test samples.")

    # ==================== Tokenizer（与训练代码完全一致，官方要求） ====================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE_PATH,
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",  # DeepSeek官方强制要求
        use_fast=False        # DeepSeek官方强制要求
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ==================== 模型加载（与训练代码完全一致） ====================
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False  # 与训练代码保持一致
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # 添加验证打印，确保加载正确
    print(f"\n✅ 模型数据类型: {base_model.dtype}")
    print(f"✅ 模型设备: {next(base_model.parameters()).device}")
    logger.info(f"PEFT adapter loaded: {ADAPTER_DIR}")
    logger.info(f"Model device: {next(model.parameters()).device}\n")

    results = []
    correct = 0

    for sample in test_samples:
        # ==================== Prompt（与训练代码完全一致） ====================
        prompt = (
            "### User:\n"
            "Solve the math problem step by step, show detailed reasoning, and put the final answer in \\boxed{}.\n\n"
            "### Problem:\n"
            f"{sample.quest}\n\n"
            "### Assistant:\n"
        )

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            input_length = inputs.input_ids.shape[1]

            # 生成参数（与训练代码的推理逻辑一致）
            generate_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stopping_criteria": StoppingCriteriaList([StopOnMathAnswer(tokenizer, input_length)]),
                "repetition_penalty": 1.05,
                "length_penalty": 1.0
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, **generate_kwargs)

            generated = outputs[0, input_length:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()

            pred_raw = extract_answer_from_response(response)
            pred_norm = normalize_math(pred_raw) if pred_raw else ""
            true_norm = normalize_math(str(sample.answer))
            is_correct = is_correct_match(true_norm, pred_norm)

            if is_correct:
                correct += 1

            results.append({
                "index": sample.index,
                "question": sample.quest,
                "true_answer": str(sample.answer).strip(),
                "model_output": response,
                "pred_raw": pred_raw if pred_raw else "(no boxed or Answer found)",
                "pred_norm": pred_norm,
                "true_norm": true_norm,
                "correct": is_correct
            })

            status = "Correct" if is_correct else "Wrong"
            logger.info(f"Sample {sample.index} | True: '{true_norm}' | Pred: '{pred_raw}' → {status}")

        except Exception as e:
            logger.error(f"Error on sample {sample.index}: {str(e)}")
            results.append({
                "index": sample.index,
                "question": sample.quest,
                "true_answer": str(sample.answer),
                "model_output": f"ERROR: {str(e)}",
                "pred_raw": None,
                "correct": False
            })

        torch.cuda.empty_cache()
        gc.collect()

    accuracy = correct / total_samples * 100 if total_samples > 0 else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"MATH-500 Test Completed! Accuracy: {accuracy:.2f}% ({correct}/{total_samples})")
    logger.info(f"{'='*60}\n")

    csv_path = os.path.join(OUTPUT_DIR, f"Math500_test_results_{latest_adapter}_DeepSeek.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys() if results else [])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved: {csv_path}")

    summary_path = os.path.join(OUTPUT_DIR, f"Math500_summary_{latest_adapter}_DeepSeek.txt")
    with open(summary_path, 'w') as f:
        f.write(f"MATH-500 Test Result (DeepSeek-Math-7B-Base + Base LoRA)\n")
        f.write(f"Adapter: {latest_adapter}\n")
        f.write(f"Prompt: DeepSeek User/Assistant format\n")
        f.write(f"Tokenizer: padding_side=left, use_fast=False\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Correct: {correct} / {total_samples}\n")

    return accuracy, results

if __name__ == "__main__":
    test_math500()