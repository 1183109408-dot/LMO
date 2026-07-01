import time
import torch
import numpy as np
import os
import csv
import joblib
import re
import gc
import traceback
import logging
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from data_deal.scienceQA_load import load_and_process_json
from transformers import BertModel, BertTokenizer

# File paths
TEST_JSON_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/datas/scienceQA/test4.json"
TRAIN_JSON_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/datas/scienceQA/train4.json"
BEST_INSTRUCTIONS_CSV = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/instructions/best_instructions.csv"
CLASSIFIER_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/classifier"
BERT_MODEL_PATH = r"/root/WuYanzu/OSAI/model/bert-base-cased"
MODEL_PATH = r"/root/WuYanzu/OSAI/model/MiniCPM5-1B-SFT"  # 改为 MiniCPM5-1B-SFT
OUTPUT_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/test04"
GPU = 0
MAX_LENGTH = 4096

# Configure logging
os.makedirs(OUTPUT_PATH, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_PATH, "test_info.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom stopping criteria
class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_text, input_length):
        self.tokenizer = tokenizer
        self.stop_text = stop_text
        self.stop_ids = tokenizer.encode(stop_text, add_special_tokens=False)
        self.input_length = input_length
        if not self.stop_ids or any(id is None for id in self.stop_ids):
            logger.error(f"Invalid stop_ids: {self.stop_ids}")
            self.stop_ids = [tokenizer.eos_token_id]

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) <= self.input_length + 10:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        try:
            new_tokens = input_ids[0, self.input_length:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            is_stop = bool(re.search(r"\(Correct Answer:\s*\d+\)", decoded, re.IGNORECASE))
            if is_stop:
                logger.info(f"StopOnAnswer triggered, matched text: {decoded[-50:]}")
            return torch.tensor([is_stop], dtype=torch.bool, device=input_ids.device)
        except Exception as e:
            logger.error(f"StopOnAnswer exception: {str(e)}")
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

# Local model class (adapted for MiniCPM)
class LocalModel:
    def __init__(self, model_path=MODEL_PATH):
        self.device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to eos_token_id: {self.tokenizer.eos_token_id}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{GPU}"},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        self.model.eval()

    def generate(self, prompt, max_tokens=1200, quest_options=None):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        
        for attempt in range(3):
            try:
                with torch.no_grad():
                    messages = [{"role": "user", "content": prompt}]
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        max_length=MAX_LENGTH - 512,
                        padding=True,
                        truncation=True
                    )
                    if inputs is None or not isinstance(inputs, torch.Tensor) or inputs.shape[1] == 0:
                        logger.error(f"Tokenizer returned invalid input: {inputs}")
                        return "Error: Tokenizer failed"
                    
                    input_ids = inputs.to(self.device)
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
                    
                    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask) or input_ids.shape != attention_mask.shape:
                        logger.error(f"Invalid input_ids or attention_mask: {input_ids.shape}, {attention_mask.shape}")
                        return "Error: Invalid input"
                    
                    input_length = input_ids.shape[1]
                    prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                    logger.info(f"Input token count: {input_length}, Prompt token count: {prompt_tokens}")
                    if input_length > MAX_LENGTH - 512:
                        logger.warning(f"Input token count {input_length} exceeds limit, truncated")
                    
                    stopping_criteria = StoppingCriteriaList([
                        StopOnAnswer(self.tokenizer, "(Correct Answer:", input_length)
                    ])
                    generate_kwargs = {
                        "max_new_tokens": max_tokens,
                        "do_sample": True,
                        "temperature": 0.4,
                        "top_p": 0.9,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "num_return_sequences": 1,
                        "stopping_criteria": stopping_criteria
                    }
                    
                    start_time = time.time()
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generate_kwargs
                    )
                    elapsed_time = time.time() - start_time
                    logger.info(f"Attempt {attempt + 1}, generation time: {elapsed_time:.2f} seconds")
                    
                    if not isinstance(outputs, torch.Tensor):
                        logger.error(f"Attempt {attempt + 1}, output is not a tensor, type: {type(outputs)}")
                        return "Error: Invalid output format"
                    if len(outputs.shape) == 1:
                        outputs = outputs.unsqueeze(0)
                        logger.warning(f"Attempt {attempt + 1}, output is 1D tensor, converted to: {outputs.shape}")
                    if outputs.shape[0] != 1 or outputs.shape[1] <= input_length:
                        logger.warning(f"Attempt {attempt + 1}, insufficient output, output shape: {outputs.shape}, input_length: {input_length}")
                        return "Error: Empty or insufficient output"
                    
                    generated_tokens = outputs[0, input_length:] if outputs.shape[1] > input_length else torch.tensor([], device=self.device)
                    new_token_count = generated_tokens.shape[0] if generated_tokens.shape else 0
                    logger.info(f"Generated new tokens: {new_token_count}")
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() if new_token_count > 0 else ""
                    
                    del input_ids, attention_mask, outputs
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                    
                    if response:
                        return response
                    logger.warning(f"Attempt {attempt + 1}, empty new tokens")
                    continue
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Attempt {attempt + 1}, out of memory error: {str(e)}")
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                continue
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}, generation failed: {str(e)}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
                continue
        
        return "Error: Generation failed after multiple attempts"

# BERT embedding
class BertEmbedding(torch.nn.Module):
    def __init__(self, model_path):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(model_path).to(f"cuda:{GPU}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
    
    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(f"cuda:{GPU}") for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]

# Extract answer
def extract_answer(text, quest_options=None):
    cleaned_text = re.sub(r'\$\s*\\boxed\s*\{(\d+)\}\s*\$', r'\boxed{\1}', text)
    cleaned_text = re.sub(r'\$\s*\$', '', cleaned_text)
    text_lower = cleaned_text.lower().replace('**', '')
    
    patterns = [
        r'\(correct answer:\s*(\d+)\)',
        r'the\s*final\s*answer\s*is\s*[:=]?\s*\\boxed{(\d+)}',
        r'the\s*final\s*answer\s*is\s*[:=]?\s*(\d+)',
        r'correct answer:\s*(\d+)',
        r'the\s*correct\s*answer\s*is\s*option\s*(\d+)',
        r'option\s*(\d+)\s*is\s*correct'
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            return match.group(1)
    
    if quest_options:
        options = [line for line in quest_options.split('\n') if re.match(r'\d+\.\s*.+', line)]
        for line in options:
            match = re.match(r'(\d+)\.\s*(.+)', line, re.IGNORECASE)
            if match:
                idx, opt = match.groups()
                opt_lower = opt.lower().strip()
                if re.search(rf'\b{opt_lower}\b.*?\b(correct|answer|right)\b', text_lower, re.IGNORECASE):
                    return idx
                if 'understatement' in opt_lower and 'adjustment' in text_lower:
                    return idx
                if 'antithesis' in opt_lower and 'antithesis' in text_lower:
                    return idx
                if 'bother' in opt_lower and ('bother' in text_lower or 'guilt' in text_lower or 'pressure' in text_lower):
                    return idx
                if 'fable' in opt_lower and 'fable' in text_lower:
                    return idx
    
    logger.warning(f"Answer extraction failed, generated content: {text[:200]}...")
    return None

# Load best instructions
def load_best_instructions(csv_path):
    best_instructions = {}
    if os.path.isfile(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                best_instructions[row['Label']] = row['Instruction']
        logger.info(f"Loaded best instructions: {len(best_instructions)} entries")
    return best_instructions

# Load training samples by category
def load_training_samples(json_path):
    train_big_classes = load_and_process_json(json_path)
    samples_by_category = {}
    for big_class in train_big_classes:
        for small_class in big_class.data:
            label = small_class.label
            if label not in samples_by_category:
                samples_by_category[label] = []
            for sample in small_class.data:
                samples_by_category[label].append({
                    "quest": sample.quest,
                    "reason": sample.reason,
                    "answer": sample.answer
                })
    logger.info(f"Loaded training samples: {sum(len(samples) for samples in samples_by_category.values())} samples across {len(samples_by_category)} categories")
    return samples_by_category

# Log errors to CSV
def log_error_to_csv(sample, respond, test_type, csv_path=OUTPUT_PATH, outputs_shape='N/A'):
    csv_filename = os.path.join(csv_path, f'{test_type}_err_log.csv')
    file_exists = os.path.isfile(csv_filename)
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Question', 'Options', 'Reasoning', 'Answer', 'Model Output', 'Output Shape'])
        writer.writerow([sample.quest, sample.quest, sample.reason, sample.answer, respond, outputs_shape])
    logger.info(f"Error logged to {csv_filename}")

# No-prompt test (Branch 1)
def test_no_prompt(big_classes, llama):
    category_stats = {}
    total_correct = 0
    total_samples = 0
    results = []
    
    for big_class in big_classes:
        for small_class in big_class.data:
            label = small_class.label
            logger.info(f"Testing category: {label}")
            if label not in category_stats:
                category_stats[label] = {"correct": 0, "total": 0}
            
            for sample in small_class.data:
                option_lines = [line for line in sample.quest.split('\n') if re.match(r'\d+\.\s*.+', line)]
                if not option_lines:
                    logger.error(f"Sample {sample.index}, no valid options found: {sample.quest}")
                    continue
                
                option_count = len(option_lines)
                options_text = '\n'.join(option_lines)
                prompt = (
                    f"Please answer the following question:\n{sample.quest}\n\n"
                    f"and conclude with the correct option in the format (Correct Answer: X), where X is the option number (0 to {option_count-1})."
                )
                
                respond = llama.generate(prompt=prompt, quest_options=options_text)
                answer = extract_answer(respond, quest_options=options_text)
                real_answer = str(sample.answer)
                
                if total_samples == 0:
                    logger.info(f"First sample {sample.index} full question:\n{sample.quest}")
                    logger.info(f"First sample {sample.index} full model output:\n{respond}")
                else:
                    logger.info(f"Sample {sample.index}, provided options: {option_lines}")
                    logger.info(f"Sample {sample.index}, true answer: {real_answer}, Llama output answer: {answer if answer else 'Not extracted'}")
                
                is_correct = answer == real_answer
                if answer is None:
                    log_error_to_csv(sample, respond, test_type="no_prompt")
                
                category_stats[label]["correct"] += 1 if is_correct else 0
                category_stats[label]["total"] += 1
                total_correct += 1 if is_correct else 0
                total_samples += 1
                
                results.append({
                    "index": sample.index,
                    "label": label,
                    "predicted_label": None,
                    "question": sample.quest,
                    "real_answer": real_answer,
                    "predicted_answer": answer if answer else "Not extracted",
                    "correct": is_correct,
                    "reasoning": respond
                })
    
    total_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    category_accuracies = {
        label: stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        for label, stats in category_stats.items()
    }
    return total_accuracy, category_accuracies, results

# Step-by-step test (Branch 2)
def test_step_by_step(big_classes, llama):
    category_stats = {}
    total_correct = 0
    total_samples = 0
    results = []
    
    for big_class in big_classes:
        for small_class in big_class.data:
            label = small_class.label
            logger.info(f"Testing category: {label}")
            if label not in category_stats:
                category_stats[label] = {"correct": 0, "total": 0}
            
            for sample in small_class.data:
                option_lines = [line for line in sample.quest.split('\n') if re.match(r'\d+\.\s*.+', line)]
                if not option_lines:
                    logger.error(f"Sample {sample.index}, no valid options found: {sample.quest}")
                    continue
                
                option_count = len(option_lines)
                options_text = '\n'.join(option_lines)
                prompt = (
                    f"Please solve the following question step by step:\n{sample.quest}\n\n"
                    f"Analyze each option (0 to {option_count-1}), provide detailed reasoning, "
                    f"and conclude with the correct option in the format (Correct Answer: X), where X is the option number (0 to {option_count-1})."
                )
                
                respond = llama.generate(prompt=prompt, quest_options=options_text)
                answer = extract_answer(respond, quest_options=options_text)
                real_answer = str(sample.answer)
                
                if total_samples == 0:
                    logger.info(f"First sample {sample.index} full question:\n{sample.quest}")
                    logger.info(f"First sample {sample.index} full model output:\n{respond}")
                else:
                    logger.info(f"Sample {sample.index}, provided options: {option_lines}")
                    logger.info(f"Sample {sample.index}, true answer: {real_answer}, Llama output answer: {answer if answer else 'Not extracted'}")
                
                is_correct = answer == real_answer
                if answer is None:
                    log_error_to_csv(sample, respond, test_type="step_by_step")
                
                category_stats[label]["correct"] += 1 if is_correct else 0
                category_stats[label]["total"] += 1
                total_correct += 1 if is_correct else 0
                total_samples += 1
                
                results.append({
                    "index": sample.index,
                    "label": label,
                    "predicted_label": None,
                    "question": sample.quest,
                    "real_answer": real_answer,
                    "predicted_answer": answer if answer else "Not extracted",
                    "correct": is_correct,
                    "reasoning": respond
                })
    
    total_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    category_accuracies = {
        label: stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        for label, stats in category_stats.items()
    }
    return total_accuracy, category_accuracies, results

# Few-shot test (Branch 3)
def test_few_shot(big_classes, llama, samples_by_category, classified_samples):
    category_stats = {}
    total_correct = 0
    total_samples = 0
    results = []
    
    for classified in classified_samples:
        sample = classified["info"]["sample"]
        true_label = classified["info"]["label"]
        predicted_label = classified["predicted_label"]
        
        if true_label not in category_stats:
            logger.info(f"Testing category: {true_label}")
            category_stats[true_label] = {"correct": 0, "total": 0}
        
        classification_correct = predicted_label == true_label
        logger.info(f"Sample {sample.index}, classifier predicted label: {predicted_label}, true label: {true_label}, classification correct: {classification_correct}")
        
        option_lines = [line for line in sample.quest.split('\n') if re.match(r'\d+\.\s*.+', line)]
        if not option_lines:
            logger.error(f"Sample {sample.index}, no valid options found: {sample.quest}")
            continue
        
        option_count = len(option_lines)
        options_text = '\n'.join(option_lines)
        
        # Select 4 random samples from predicted category
        few_shot_samples = samples_by_category.get(predicted_label, [])
        if len(few_shot_samples) < 4:
            logger.warning(f"Category {predicted_label} has only {len(few_shot_samples)} samples, using all available")
            selected_samples = few_shot_samples
        else:
            selected_samples = random.sample(few_shot_samples, 4)
        
        # Format few-shot examples
        examples_text = ""
        for idx, fs_sample in enumerate(selected_samples):
            examples_text += (
                f"Example {idx + 1}:\n"
                f"Question: {fs_sample['quest']}\n"
                f"Reasoning: {fs_sample['reason']}\n"
                f"Answer: (Correct Answer: {fs_sample['answer']})\n\n"
            )
        
        prompt = (
            f"Please solve the following question using the provided examples as a guide:\n\n"
            f"{examples_text}"
            f"Now solve this question:\n{sample.quest}\n\n"
            f"Analyze each option (0 to {option_count-1}), provide detailed reasoning, "
            f"and conclude with the correct option in the format (Correct Answer: X), where X is the option number (0 to {option_count-1})."
        )
        
        respond = llama.generate(prompt=prompt, quest_options=options_text)
        answer = extract_answer(respond, quest_options=options_text)
        real_answer = str(sample.answer)
        
        logger.info(f"Sample {sample.index}, provided options: {option_lines}")
        logger.info(f"Sample {sample.index}, true answer: {real_answer}, Llama output answer: {answer if answer else 'Not extracted'}")
        
        is_correct = answer == real_answer
        if answer is None:
            log_error_to_csv(sample, respond, test_type="few_shot")
        
        category_stats[true_label]["correct"] += 1 if is_correct else 0
        category_stats[true_label]["total"] += 1
        total_correct += 1 if is_correct else 0
        total_samples += 1
        
        results.append({
            "index": sample.index,
            "label": true_label,
            "predicted_label": predicted_label,
            "question": sample.quest,
            "real_answer": real_answer,
            "predicted_answer": answer if answer else "Not extracted",
            "correct": is_correct,
            "reasoning": respond
        })
    
    total_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    category_accuracies = {
        label: stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        for label, stats in category_stats.items()
    }
    return total_accuracy, category_accuracies, results

# Best instruction test (Branch 4)
def test_with_instruction(big_classes, llama, best_instructions, classified_samples):
    category_stats = {}
    total_correct = 0
    total_samples = 0
    results = []
    
    for classified in classified_samples:
        sample = classified["info"]["sample"]
        true_label = classified["info"]["label"]
        predicted_label = classified["predicted_label"]
        instruction = best_instructions.get(predicted_label, "No instruction found")
        
        if true_label not in category_stats:
            logger.info(f"Testing category: {true_label}")
            category_stats[true_label] = {"correct": 0, "total": 0}
        
        classification_correct = predicted_label == true_label
        logger.info(f"Sample {sample.index}, classifier predicted label: {predicted_label}, true label: {true_label}, classification correct: {classification_correct}")
        
        option_lines = [line for line in sample.quest.split('\n') if re.match(r'\d+\.\s*.+', line)]
        if not option_lines:
            logger.error(f"Sample {sample.index}, no valid options found: {sample.quest}")
            continue
        
        option_count = len(option_lines)
        options_text = '\n'.join(option_lines)
        prompt = (
            f"Instruction: {instruction}\n\n"
            f"Please answer the following question based on the instruction above:\n{sample.quest}\n\n"
            f"Analyze each option (0 to {option_count-1}), provide detailed reasoning, "
            f"and conclude with the correct option in the format (Correct Answer: X), where X is the option number (0 to {option_count-1})."
        )
        
        respond = llama.generate(prompt=prompt, quest_options=options_text)
        answer = extract_answer(respond, quest_options=options_text)
        real_answer = str(sample.answer)
        
        logger.info(f"Sample {sample.index}, provided options: {option_lines}")
        logger.info(f"Sample {sample.index}, true answer: {real_answer}, Llama output answer: {answer if answer else 'Not extracted'}")
        logger.info(f"Sample {sample.index}, instruction used (first 100 chars): {instruction[:100]}")
        
        is_correct = answer == real_answer
        if answer is None:
            log_error_to_csv(sample, respond, test_type="instruction")
        
        category_stats[true_label]["correct"] += 1 if is_correct else 0
        category_stats[true_label]["total"] += 1
        total_correct += 1 if is_correct else 0
        total_samples += 1
        
        results.append({
            "index": sample.index,
            "label": true_label,
            "predicted_label": predicted_label,
            "question": sample.quest,
            "real_answer": real_answer,
            "predicted_answer": answer if answer else "Not extracted",
            "correct": is_correct,
            "reasoning": respond
        })
    
    total_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    category_accuracies = {
        label: stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        for label, stats in category_stats.items()
    }
    return total_accuracy, category_accuracies, results

# Classify samples
def classify_samples(big_classes, feature_encoder, svm, projection_matrix, scaler):
    features = []
    sample_info = []
    for big_class in big_classes:
        for small_class in big_class.data:
            for sample in small_class.data:
                feat = feature_encoder([sample.quest])
                features.append(feat)
                sample_info.append({"index": sample.index, "label": small_class.label, "sample": sample})
    features_tensor = torch.cat(features, dim=0)
    features_np = features_tensor.detach().cpu().numpy()
    features_scaled = scaler.transform(features_np)
    reduced_features = features_scaled @ projection_matrix
    reduced_features = np.nan_to_num(reduced_features, 0)
    predicted_labels = svm.predict(reduced_features)
    return [{"info": info, "predicted_label": label} for info, label in zip(sample_info, predicted_labels)]

# Save results
def save_results(results, filename, csv_path=OUTPUT_PATH):
    csv_filename = os.path.join(csv_path, filename)
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {csv_filename}")

# Save accuracy summary
def save_accuracy_summary(total_accuracy, category_accuracies, results, filename, csv_path=OUTPUT_PATH):
    csv_filename = os.path.join(csv_path, filename)
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Accuracy (%)", "Sample Count"])
        for label, accuracy in category_accuracies.items():
            total_samples = sum(1 for r in results if r["label"] == label)
            writer.writerow([label, f"{accuracy:.2f}", total_samples])
        writer.writerow(["Total", f"{total_accuracy:.2f}", len(results)])
    logger.info(f"Accuracy summary saved to {csv_filename}")

# Main function
def main():
    logger.info("Starting test for MiniCPM5-1B-SFT model")
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(GPU)}")
        logger.info(f"Total memory: {torch.cuda.get_device_properties(GPU).total_memory / 1024**3:.2f} GB")
    else:
        logger.info("CUDA not available, falling back to CPU")
    
    logger.info("Loading test data")
    test_big_classes = load_and_process_json(TEST_JSON_PATH)
    if not test_big_classes:
        logger.error("Test data loading failed, exiting")
        return
    
    logger.info("Loading training data")
    samples_by_category = load_training_samples(TRAIN_JSON_PATH)
    
    logger.info("Initializing MiniCPM and BERT models")
    try:
        model = LocalModel()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    feature_encoder = BertEmbedding(BERT_MODEL_PATH)
    
    logger.info("Loading classifier")
    svm = joblib.load(os.path.join(CLASSIFIER_PATH, "svm_classifier.pkl"))
    projection_matrix = np.load(os.path.join(CLASSIFIER_PATH, "projection_matrix.npy"))
    scaler = joblib.load(os.path.join(CLASSIFIER_PATH, "scaler.pkl"))
    
    logger.info("Loading best instructions")
    best_instructions = load_best_instructions(BEST_INSTRUCTIONS_CSV)
    
    # 测试分支1：无提示测试
    logger.info("\nNo-prompt test (Branch 1)")
    no_prompt_total_accuracy, no_prompt_category_accuracies, no_prompt_results = test_no_prompt(test_big_classes, model)
    logger.info(f"No-prompt total accuracy: {no_prompt_total_accuracy:.2f}%")
    for label, accuracy in no_prompt_category_accuracies.items():
        logger.info(f"Category {label}: {accuracy:.2f}%")
    save_results(no_prompt_results, "no_prompt_test_results.csv")
    save_accuracy_summary(no_prompt_total_accuracy, no_prompt_category_accuracies, no_prompt_results, "no_prompt_accuracy_summary.csv")
    
    # 测试分支2：逐步推理测试
    logger.info("\nStep-by-step test (Branch 2)")
    step_total_accuracy, step_category_accuracies, step_results = test_step_by_step(test_big_classes, model)
    logger.info(f"Step-by-step total accuracy: {step_total_accuracy:.2f}%")
    for label, accuracy in step_category_accuracies.items():
        logger.info(f"Category {label}: {accuracy:.2f}%")
    save_results(step_results, "step_by_step_test_results.csv")
    save_accuracy_summary(step_total_accuracy, step_category_accuracies, step_results, "step_by_step_accuracy_summary.csv")
    
    # 分类样本，为分支3和4准备
    logger.info("\nClassifying samples for Branches 3 and 4")
    classified_samples = classify_samples(test_big_classes, feature_encoder, svm, projection_matrix, scaler)
    
    # 测试分支3：少样本测试
    logger.info("\nFew-shot test (Branch 3)")
    few_shot_total_accuracy, few_shot_category_accuracies, few_shot_results = test_few_shot(test_big_classes, model, samples_by_category, classified_samples)
    logger.info(f"Few-shot total accuracy: {few_shot_total_accuracy:.2f}%")
    for label, accuracy in few_shot_category_accuracies.items():
        logger.info(f"Category {label}: {accuracy:.2f}%")
    save_results(few_shot_results, "few_shot_test_results.csv")
    save_accuracy_summary(few_shot_total_accuracy, few_shot_category_accuracies, few_shot_results, "few_shot_accuracy_summary.csv")
    
    # 测试分支4：最佳指令测试
    logger.info("\nBest instruction test (Branch 4)")
    instr_total_accuracy, instr_category_accuracies, instr_results = test_with_instruction(test_big_classes, model, best_instructions, classified_samples)
    logger.info(f"Best instruction total accuracy: {instr_total_accuracy:.2f}%")
    for label, accuracy in instr_category_accuracies.items():
        logger.info(f"Category {label}: {accuracy:.2f}%")
    save_results(instr_results, "instruction_test_results.csv")
    save_accuracy_summary(instr_total_accuracy, instr_category_accuracies, instr_results, "instruction_accuracy_summary.csv")
    
    logger.info("Testing completed")

if __name__ == "__main__":
    main()