import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import os
import re
import gc
import random
import joblib
import hashlib
from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import logging
from data_deal.scienceQA_load import load_and_process_json
import json

# 参数设置
MODEL_PATH = r"/root/WuYanzu/OSAI/model/MiniCPM5-1B-SFT"          # MiniCPM5-1B-SFT
BERT_PATH = r"/root/WuYanzu/OSAI/model/bert-base-cased"
TRAIN_JSON_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/datas/scienceQA/train4.json"
CSV_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/instructions/correct_samples_with_weights.csv"
OUTPUT_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/finetuned_MiniCPM5_1B_4rank2"
CLASSIFIER_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/classifier"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
EPOCHS = 10
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.3
GPU = 0
MAX_LENGTH = 1024
MAX_RETRIES = 3
CONDITIONAL_LAYERS = 2
CONDITIONAL_FRONT = True

# 设置日志
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_PATH, exist_ok=True)
debug_handler = logging.FileHandler(os.path.join(OUTPUT_PATH, "debug_info.txt"))
debug_handler.setLevel(logging.ERROR)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(debug_handler)

# 用于初始 Loss 和 epoch 结束日志的单独 logger
info_logger = logging.getLogger('info_logger')
info_logger.handlers.clear()
info_logger.propagate = False
info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
info_logger.addHandler(info_handler)
info_logger.setLevel(logging.INFO)

# 禁用调试模式
DEBUG = False

# 禁用 tokenizer 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 设置 PyTorch 内存分配策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 计算数组哈希值
def compute_array_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()

# 自定义条件 LoRA 层（用于 q_proj）
class ConditionalLoRALayer(nn.Module):
    def __init__(self, base_layer, in_dim, out_dim, rank, alpha, label_vector_dim, device, dtype):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.A = nn.Parameter(torch.empty(in_dim, rank, device=device, dtype=dtype), requires_grad=True)
        nn.init.kaiming_uniform_(self.A, mode='fan_in', nonlinearity='relu')
        self.B = nn.Parameter(torch.zeros(rank, out_dim, device=device, dtype=dtype), requires_grad=True)
        self.B_prime = nn.Parameter(torch.empty(label_vector_dim, rank, device=device, dtype=dtype), requires_grad=True)
        nn.init.kaiming_uniform_(self.B_prime, mode='fan_in', nonlinearity='relu')
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.label_vector = None

    def forward(self, x):
        if self.label_vector is None:
            raise ValueError("label_vector 未设置")
        if self.label_vector.dim() != 2 or self.label_vector.shape != (1, self.B_prime.shape[0]):
            raise ValueError(f"label_vector 形状错误，预期 [1, {self.B_prime.shape[0]}], 实际 {self.label_vector.shape}")

        a_prime = self.label_vector.repeat(self.rank, 1)
        delta_W = self.scaling * (self.A @ a_prime @ self.B_prime @ self.B)

        with torch.no_grad():
            base_output = self.base_layer(x)
        lora_output = x @ delta_W
        return base_output + lora_output

# 自定义标准 LoRA 层
class CustomLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout, device, dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype), requires_grad=True)

        torch.nn.init.normal_(self.lora_A, mean=0.0, std=1.0)
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original = torch.matmul(x, self.weight.t())
        lora = self.dropout(x)
        lora = torch.matmul(lora, self.lora_A.t())
        lora = torch.matmul(lora, self.lora_B.t())
        lora = lora * self.scaling
        return original + lora

# 自定义 LoRA 模型
class ConditionalLoRAModel(nn.Module):
    def __init__(self, base_model, rank, alpha, dropout, label_vector_dim):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.label_vector_dim = label_vector_dim

        for name, param in self.base_model.named_parameters():
            param.requires_grad = False

        self.replace_linear_with_lora()

        trainable_params = [(name, param.shape, param.requires_grad) for name, param in self.named_parameters() if param.requires_grad]
        if not trainable_params:
            logger.error("No trainable parameters found after initialization!")
            raise ValueError("ConditionalLoRAModel has no trainable parameters.")
        trainable_param_count = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad)
        info_logger.info(f"Trainable parameters: {trainable_param_count}")

    def replace_linear_with_lora(self):
        # 直接使用每个 Linear 层的实际维度，不再依赖全局 hidden_size/kv_out_dim
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    device = module.weight.device
                    dtype = module.weight.dtype
                    out_features = module.weight.shape[0]
                    in_features = module.weight.shape[1]

                    if "q_proj" in name:
                        if layer_idx < CONDITIONAL_LAYERS:
                            lora_layer = ConditionalLoRALayer(
                                base_layer=module,
                                in_dim=in_features,
                                out_dim=out_features,
                                rank=self.rank,
                                alpha=self.alpha,
                                label_vector_dim=self.label_vector_dim,
                                device=device,
                                dtype=dtype
                            )
                            lora_layer.A.data = lora_layer.A.data.to(device=device, dtype=dtype)
                            lora_layer.B.data = lora_layer.B.data.to(device=device, dtype=dtype)
                            lora_layer.B_prime.data = lora_layer.B_prime.data.to(device=device, dtype=dtype)
                        else:
                            lora_layer = CustomLoRALayer(
                                in_features=in_features,
                                out_features=out_features,
                                rank=self.rank,
                                alpha=self.alpha,
                                dropout=self.dropout,
                                device=device,
                                dtype=dtype
                            )
                            lora_layer.weight.data = module.weight.data.clone()
                            lora_layer.lora_A.data = lora_layer.lora_A.data.to(device=device, dtype=dtype)
                            lora_layer.lora_B.data = lora_layer.lora_B.data.to(device=device, dtype=dtype)
                    elif "k_proj" in name or "v_proj" in name:
                        lora_layer = CustomLoRALayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=self.rank,
                            alpha=self.alpha,
                            dropout=self.dropout,
                            device=device,
                            dtype=dtype
                        )
                        lora_layer.weight.data = module.weight.data.clone()
                        lora_layer.lora_A.data = lora_layer.lora_A.data.to(device=device, dtype=dtype)
                        lora_layer.lora_B.data = lora_layer.lora_B.data.to(device=device, dtype=dtype)
                    else:
                        continue

                    parent_name = name.rsplit('.', 1)[0]
                    parent_module = layer.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], lora_layer)

    def forward(self, input_ids, attention_mask=None, label_vectors=None, labels=None):
        if label_vectors is None and CONDITIONAL_LAYERS > 0:
            raise ValueError("label_vectors 必须提供")
        if label_vectors is not None:
            if label_vectors.dim() < 2:
                label_vectors = label_vectors.unsqueeze(0)
            if label_vectors.shape[0] != input_ids.shape[0]:
                raise ValueError(f"label_vectors batch size {label_vectors.shape[0]} 不匹配 input_ids {input_ids.shape[0]}")

            for i in range(input_ids.shape[0]):
                for layer in self.base_model.model.layers:
                    if hasattr(layer.self_attn.q_proj, 'label_vector'):
                        layer.self_attn.q_proj.label_vector = label_vectors[i:i+1].to(device=f"cuda:{GPU}")

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )
        return outputs

# 保存 LoRA 检查点
def save_custom_lora_checkpoint(model, tokenizer, output_path, epoch):
    checkpoint_dir = os.path.join(output_path, f"epoch_{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, ConditionalLoRALayer):
            lora_state_dict[f"{name}.A"] = module.A
            lora_state_dict[f"{name}.B"] = module.B
            lora_state_dict[f"{name}.B_prime"] = module.B_prime
        elif isinstance(module, CustomLoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A
            lora_state_dict[f"{name}.lora_B"] = module.lora_B
    torch.save(lora_state_dict, os.path.join(checkpoint_dir, "lora_model.bin"))
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL_PATH,
        "r": model.rank,
        "lora_alpha": model.alpha,
        "lora_dropout": model.dropout,
        "label_vector_dim": model.label_vector_dim,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "conditional_layers": CONDITIONAL_LAYERS,
        "conditional_front": CONDITIONAL_FRONT
    }
    with open(os.path.join(checkpoint_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)
    tokenizer.save_pretrained(checkpoint_dir)
    info_logger.info(f"保存检查点: {checkpoint_dir}")

# 加载 LoRA 检查点
def load_custom_lora_checkpoint(model, tokenizer, checkpoint_path):
    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    if config["r"] != model.rank or config["lora_alpha"] != model.alpha or config["lora_dropout"] != model.dropout or config["label_vector_dim"] != model.label_vector_dim or config["conditional_layers"] != CONDITIONAL_LAYERS:
        raise ValueError("Checkpoint config does not match model config")
    lora_state_dict = torch.load(os.path.join(checkpoint_path, "lora_model.bin"))
    for name, module in model.named_modules():
        if isinstance(module, ConditionalLoRALayer):
            module.A.data = lora_state_dict[f"{name}.A"].to(device=module.A.device, dtype=module.A.dtype)
            module.B.data = lora_state_dict[f"{name}.B"].to(device=module.B.device, dtype=module.B.dtype)
            module.B_prime.data = lora_state_dict[f"{name}.B_prime"].to(device=module.B_prime.device, dtype=module.B_prime.dtype)
        elif isinstance(module, CustomLoRALayer):
            module.lora_A.data = lora_state_dict[f"{name}.lora_A"].to(device=module.lora_A.device, dtype=module.lora_A.dtype)
            module.lora_B.data = lora_state_dict[f"{name}.lora_B"].to(device=module.lora_B.device, dtype=module.lora_B.dtype)
    tokenizer_new = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer_new.pad_token
    tokenizer.pad_token_id = tokenizer_new.pad_token_id
    info_logger.info(f"加载检查点: {checkpoint_path}")

# BERT 嵌入
class BertEmbedding(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.cuda(GPU) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]

def apply_transform(features, projection_matrix, scaler):
    features_np = features.detach().cpu().numpy()
    features_scaled = scaler.transform(features_np)
    reduced_features = features_scaled @ projection_matrix
    reduced_features = np.nan_to_num(reduced_features, nan=0.0, posinf=0.0, neginf=0.0)
    return reduced_features

# 自定义 collate_fn
class CustomDataCollator:
    def __init__(self, tokenizer, label_vectors):
        self.tokenizer = tokenizer
        self.label_vectors = np.copy(label_vectors)
        if len(self.label_vectors) <= 1:
            logger.error(f"label_vectors has insufficient categories: {len(self.label_vectors)}")
            raise ValueError(f"label_vectors must have more than 1 category, got {len(self.label_vectors)}")

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            raise ValueError("Empty batch detected")
        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"]
            }
            for item in batch
        ]
        label_indices = [item["label_idx"] for item in batch]

        if any(idx < 0 or idx >= len(self.label_vectors) for idx in label_indices):
            invalid_indices = [idx for idx in label_indices if idx < 0 or idx >= len(self.label_vectors)]
            logger.error(f"Invalid label_indices detected: {invalid_indices}")
            raise ValueError(f"Invalid label_indices: {invalid_indices}, must be in [0, {len(self.label_vectors)-1}]")

        label_vectors = torch.from_numpy(self.label_vectors[label_indices]).to(device=f"cuda:{GPU}", dtype=torch.float32)

        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_vectors": label_vectors
        }

# 数据集类
class ScienceQADataset(Dataset):
    def __init__(self, samples, label_indices, tokenizer, label_to_idx, max_length=1024, len_original_samples=0, original_idx=None, generated_idx=None):
        self.samples = samples
        self.label_indices = label_indices
        self.tokenizer = tokenizer
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        self.len_original_samples = len_original_samples
        self.original_idx = original_idx
        self.generated_idx = generated_idx
        self.problem_samples = []

        invalid_indices = [idx for idx in label_indices if idx < 0 or idx >= len(self.label_to_idx)]
        if invalid_indices:
            logger.error(f"Invalid label_indices in dataset: {invalid_indices}")
            raise ValueError(f"Invalid label_indices: {invalid_indices}, must be in [0, {len(self.label_to_idx)-1}]")

        system_prompt = "You are a helpful assistant. Answer the question by evaluating each option and selecting the correct one, ending with (Correct Answer: X)."
        template_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ]
        template_encoding = self.tokenizer.apply_chat_template(
            template_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        self.template_length = template_encoding.shape[-1]
        # 若 tokenizer 未自带 chat_template，则使用 MiniCPM 的 ChatML 格式模板
        if not hasattr(self.tokenizer, 'chat_template') or not self.tokenizer.chat_template:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if loop.first and messages[0]['role'] != 'system' %}"
                "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
                "{% endif %}"
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = sample["prompt"]
        answer = sample["answer"]
        weight = sample["weight"]
        label_idx = self.label_indices[idx]

        max_content_length = max(100, self.max_length - self.template_length - 10)
        max_prompt_tokens = int(max_content_length * 0.7)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)

        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[:max_prompt_tokens]
            prompt = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

        answer_suffix = re.search(r"\(Correct Answer:.*?\)$", answer)
        if answer_suffix:
            answer_suffix_text = answer_suffix.group(0)
            reasoning_text = answer[:answer_suffix.start()].strip()
        else:
            reasoning_text = answer
            answer_suffix_text = ""
        answer_suffix_tokens = self.tokenizer.encode(answer_suffix_text, add_special_tokens=False)
        reasoning_tokens = self.tokenizer.encode(reasoning_text, add_special_tokens=False)

        answer_suffix_len = len(answer_suffix_tokens)
        max_answer_tokens = max_content_length - len(prompt_tokens)
        max_reasoning_tokens = max(50, max_answer_tokens - answer_suffix_len - 20)

        if len(reasoning_tokens) > max_reasoning_tokens:
            reasoning_tokens = reasoning_tokens[:max_reasoning_tokens]
            reasoning_text = self.tokenizer.decode(reasoning_tokens, skip_special_tokens=True)
            answer = f"{reasoning_text}\n{answer_suffix_text}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the question by evaluating each option and selecting the correct one, ending with (Correct Answer: X)."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        try:
            encoding = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
        except Exception as e:
            error_msg = f"apply_chat_template failed for sample {idx}: {str(e)}"
            logger.error(error_msg)
            self.problem_samples.append({
                "index": idx,
                "prompt": prompt,
                "answer": answer,
                "error": error_msg
            })
            raise ValueError(error_msg)

        input_ids = encoding.squeeze()
        if input_ids.dim() > 1:
            error_msg = f"Unexpected encoding shape for sample {idx}: {input_ids.shape}"
            logger.error(error_msg)
            self.problem_samples.append({
                "index": idx,
                "prompt": prompt,
                "answer": answer,
                "error": error_msg
            })
            raise ValueError(error_msg)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # MiniCPM 的 assistant 标记
        assistant_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if not assistant_tokens:
            error_msg = f"Failed to encode assistant tokens for sample {idx}"
            logger.error(error_msg)
            self.problem_samples.append({
                "index": idx,
                "prompt": prompt,
                "answer": answer,
                "error": error_msg
            })
            raise ValueError(error_msg)

        input_ids_list = input_ids.tolist()
        answer_start = -1
        for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
            if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
                answer_start = i
                break

        if answer_start == -1:
            # 备用：仅查找 "assistant"
            alternative_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
            for i in range(len(input_ids_list) - len(alternative_tokens) + 1):
                if input_ids_list[i:i+len(alternative_tokens)] == alternative_tokens:
                    answer_start = i
                    break

        if answer_start == -1:
            error_msg = f"Assistant start token not found in sample {idx}"
            logger.error(error_msg)
            self.problem_samples.append({
                "index": idx,
                "prompt": prompt,
                "answer": answer,
                "error": error_msg
            })
            raise ValueError(error_msg)

        labels = input_ids.clone()
        labels[:answer_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weight": weight,
            "label_idx": label_idx
        }

    def save_problem_samples(self, output_path):
        if self.problem_samples:
            problem_df = pd.DataFrame(self.problem_samples)
            problem_csv = os.path.join(output_path, "problem_samples.csv")
            problem_df.to_csv(problem_csv, index=False)
            info_logger.info(f"Problem samples saved to: {problem_csv}")

# 检查最新 epoch 参数
def get_latest_epoch_path(output_path):
    if not os.path.exists(output_path):
        return None, 0
    epoch_dirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("epoch_")]
    if not epoch_dirs:
        return None, 0
    epoch_nums = [int(re.match(r"epoch_(\d+)", d).group(1)) for d in epoch_dirs]
    latest_epoch = max(epoch_nums)
    latest_path = os.path.join(output_path, f"epoch_{latest_epoch}")
    return latest_path, latest_epoch

# 主函数
def main():
    info_logger.info("Entering main function (MiniCPM5-1B-SFT)")
    torch.nn.attention.sdpa_kernel(backends=["math"])
    torch.cuda.empty_cache()
    gc.collect()

    loss_csv_path = os.path.join(OUTPUT_PATH, "epoch_losses.csv")
    if os.path.exists(loss_csv_path):
        loss_records = pd.read_csv(loss_csv_path).to_dict('records')
    else:
        loss_records = []

    pkl_path = os.path.join(CLASSIFIER_PATH, "class_label_vectors.pkl")
    n_components_path = os.path.join(CLASSIFIER_PATH, "n_components_lda.npy")
    svm_path = os.path.join(CLASSIFIER_PATH, "svm_classifier.pkl")
    scaler_path = os.path.join(CLASSIFIER_PATH, "scaler.pkl")
    projection_matrix_path = os.path.join(CLASSIFIER_PATH, "projection_matrix.npy")

    class_label_vectors = joblib.load(pkl_path)
    n_components_lda = np.load(n_components_path).item()
    svm = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    projection_matrix = np.load(projection_matrix_path)

    label_vectors = []
    label_to_idx = {}
    idx_to_label = {}
    for idx, (label, vector) in enumerate(class_label_vectors.items()):
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm
        else:
            normalized_vector = vector
        label_vectors.append(normalized_vector)
        label_to_idx[label] = idx
        idx_to_label[idx] = label
    label_vectors = np.array(label_vectors, dtype=np.float32)
    label_vector_dim = label_vectors.shape[1]

    original_label_vectors = np.copy(label_vectors)
    original_label_vectors_hash = compute_array_hash(original_label_vectors)

    bert_encoder = BertEmbedding(BERT_PATH).cuda(GPU)

    big_classes = load_and_process_json(TRAIN_JSON_PATH)
    original_samples = []
    original_label_indices = []
    for big_class in big_classes:
        for small_class in big_class.data:
            label = small_class.label
            label_idx = label_to_idx[label]
            for sample in small_class.data:
                options = re.findall(r'(\d+)\.\s*([^\n]+)', sample.quest)
                if len(options) < 2:
                    continue
                prompt = f"Question: {sample.quest}\nEvaluate each option (0, 1, 2, etc.) and explain why it is correct or incorrect. Select the correct option and end with (Correct Answer: X)."
                reason_clean = re.sub(r"^(Option \d+:.*?\n)?", "", sample.reason, flags=re.MULTILINE).strip()
                correct_option = str(sample.answer)
                incorrect_options = [idx for idx, _ in options if idx != correct_option]
                incorrect_str = ", ".join(incorrect_options[:-1]) + " and " + incorrect_options[-1] if len(incorrect_options) > 1 else incorrect_options[0]
                reasoning_text = reason_clean
                for idx, opt in options:
                    if idx != correct_option:
                        reasoning_text += f"\nIn contrast, Option {idx} ({opt}) is incorrect because it does not match the context of the question."
                summary = f"Based on the above reasoning, Option {correct_option} is correct, Options {incorrect_str} are incorrect. (Correct Answer: {correct_option})"
                answer = f"{reasoning_text}\n{summary}"
                original_samples.append({
                    "prompt": prompt,
                    "answer": answer,
                    "weight": 1.0
                })
                original_label_indices.append(label_idx)

    generated_df = pd.read_csv(CSV_PATH)
    generated_samples = []
    generated_label_indices = []
    for _, row in generated_df.iterrows():
        label = row.get('label', None)
        if label is None or label not in label_to_idx:
            question = row['Question']
            features = bert_encoder([question])
            reduced_features = apply_transform(features, projection_matrix, scaler)
            label = svm.predict(reduced_features)[0]
            label_idx = label_to_idx[label]
        else:
            label_idx = label_to_idx[label]
        options = re.findall(r'(\d+)\.\s*([^\n]+)', row['Question'])
        if len(options) < 2:
            continue
        prompt = f"Question: {row['Question']}\nEvaluate each option (0, 1, 2, etc.) and explain why it is correct or incorrect. Select the correct option and end with (Correct Answer: X)."
        output = row['Output']
        answer_text = str(row['Answer'])
        output_clean = re.sub(r"^(Option \d+:.*?\n)?", "", output, flags=re.MULTILINE)
        output_clean = re.sub(r"\(Correct Answer:.*?\)$", "", output_clean).strip()
        correct_option = answer_text
        incorrect_options = [idx for idx, _ in options if idx != correct_option]
        incorrect_str = ", ".join(incorrect_options[:-1]) + " and " + incorrect_options[-1] if len(incorrect_options) > 1 else incorrect_options[0]
        reasoning_text = output_clean
        for idx, opt in options:
            if idx != correct_option:
                reasoning_text += f"\nIn contrast, Option {idx} ({opt}) is incorrect because it does not match the context of the question."
        summary = f"Based on the above reasoning, Option {correct_option} is correct, Options {incorrect_str} are incorrect. (Correct Answer: {correct_option})"
        answer = f"{reasoning_text}\n{summary}"
        generated_samples.append({
            "prompt": prompt,
            "answer": answer,
            "weight": row["weight"]
        })
        generated_label_indices.append(label_idx)

    all_samples = original_samples + generated_samples
    all_label_indices = original_label_indices + generated_label_indices

    original_idx = random.randint(0, len(original_samples)-1) if original_samples else None
    generated_idx = random.randint(len(original_samples), len(all_samples)-1) if generated_samples else None

    latest_epoch_path, start_epoch = get_latest_epoch_path(OUTPUT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = ScienceQADataset(
        samples=all_samples,
        label_indices=all_label_indices,
        tokenizer=tokenizer,
        label_to_idx=label_to_idx,
        max_length=MAX_LENGTH,
        len_original_samples=len(original_samples),
        original_idx=original_idx,
        generated_idx=generated_idx
    )

    weights = [s["weight"] for s in all_samples]
    num_samples = len(train_dataset) // 3

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": f"cuda:{GPU}"},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = ConditionalLoRAModel(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, label_vector_dim)

    if latest_epoch_path:
        load_custom_lora_checkpoint(model, tokenizer, latest_epoch_path)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )

    model.train()
    for epoch in range(start_epoch, EPOCHS):
        torch.manual_seed(42 + epoch)
        np.random.seed(42 + epoch)

        current_label_vectors_hash = compute_array_hash(label_vectors)
        if current_label_vectors_hash != original_label_vectors_hash:
            label_vectors = np.copy(original_label_vectors)

        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.synchronize()

        epoch_output_path = os.path.join(OUTPUT_PATH, f"epoch_{epoch+1}")
        epoch_checkpoint_path = os.path.join(OUTPUT_PATH, f"temp_epoch_{epoch+1}_start_checkpoint.pth")

        os.makedirs(OUTPUT_PATH, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, epoch_checkpoint_path)

        total_loss = 0
        accumulation_steps = 0
        retry_count = 0
        while retry_count < MAX_RETRIES:
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                collate_fn=CustomDataCollator(tokenizer, label_vectors),
                shuffle=False
            )

            restart_epoch = False
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
            for step, batch in enumerate(progress_bar):
                try:
                    input_ids = batch["input_ids"].to(device=f"cuda:{GPU}")
                    attention_mask = batch["attention_mask"].to(device=f"cuda:{GPU}")
                    labels = batch["labels"].to(device=f"cuda:{GPU}")
                    label_vectors_batch = batch["label_vectors"].to(device=f"cuda:{GPU}")

                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids, attention_mask=attention_mask, label_vectors=label_vectors_batch, labels=labels)
                        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                    loss.backward()

                    batch_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
                    accumulation_steps += 1

                    if step == 0 and epoch == start_epoch:
                        info_logger.info(f"Initial Loss: {batch_loss:.4f}")

                    if step == 0:
                        no_grad_params = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
                        if no_grad_params:
                            logger.error(f"以下参数无梯度: {no_grad_params}")
                            raise RuntimeError(f"检测到 LoRA 参数无梯度: {no_grad_params}")

                    if accumulation_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()
                        optimizer.zero_grad()

                    total_loss += batch_loss

                    del outputs, loss
                    torch.cuda.empty_cache()
                    gc.collect()

                except torch.cuda.OutOfMemoryError as e:
                    retry_count += 1
                    logger.error(f"CUDA Out of Memory in Epoch {epoch+1}, Step {step}, restarting epoch (Retry {retry_count}/{MAX_RETRIES})")
                    if retry_count >= MAX_RETRIES:
                        total_loss = 0
                        accumulation_steps = 0
                        retry_count = 0
                        for _ in range(10):
                            torch.cuda.empty_cache()
                            gc.collect()
                        torch.cuda.synchronize()
                        continue

                    try:
                        del input_ids, attention_mask, labels, label_vectors_batch
                    except NameError:
                        pass
                    for _ in range(10):
                        torch.cuda.empty_cache()
                        gc.collect()
                    torch.cuda.synchronize()

                    checkpoint = torch.load(epoch_checkpoint_path, map_location="cpu")
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    total_loss = 0
                    accumulation_steps = 0
                    restart_epoch = True
                    break

            progress_bar.close()
            if not restart_epoch:
                break

        train_dataset.save_problem_samples(OUTPUT_PATH)

        if not restart_epoch:
            avg_loss = total_loss / len(train_loader)
            info_logger.info(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

            loss_records.append({"Epoch": epoch + 1, "Average_Loss": avg_loss})
            loss_df = pd.DataFrame(loss_records)
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            loss_df.to_csv(loss_csv_path, index=False)

            if avg_loss < 0.1:
                info_logger.info("Loss 过低，可能过拟合，提前停止")
                break

            # 打印每层 LoRA 参数的取值范围
            for layer_idx, layer in enumerate(model.base_model.model.layers):
                for proj_name, module in [('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj), ('v_proj', layer.self_attn.v_proj)]:
                    if isinstance(module, ConditionalLoRALayer):
                        a_min, a_max = module.A.min().item(), module.A.max().item()
                        b_min, b_max = module.B.min().item(), module.B.max().item()
                        b_prime_min, b_prime_max = module.B_prime.min().item(), module.B_prime.max().item()
                        info_logger.info(
                            f"Epoch {epoch+1}, Layer {layer_idx} {proj_name} (ConditionalLoRALayer): "
                            f"A min={a_min:.4f}, max={a_max:.4f}; B min={b_min:.4f}, max={b_max:.4f}; "
                            f"B_prime min={b_prime_min:.4f}, max={b_prime_max:.4f}"
                        )
                    elif isinstance(module, CustomLoRALayer):
                        a_min, a_max = module.lora_A.min().item(), module.lora_A.max().item()
                        b_min, b_max = module.lora_B.min().item(), module.lora_B.max().item()
                        info_logger.info(
                            f"Epoch {epoch+1}, Layer {layer_idx} {proj_name} (CustomLoRALayer): "
                            f"lora_A min={a_min:.4f}, max={a_max:.4f}; lora_B min={b_min:.4f}, max={b_max:.4f}"
                        )

        save_custom_lora_checkpoint(model, tokenizer, OUTPUT_PATH, epoch + 1)

        if os.path.exists(epoch_checkpoint_path):
            os.remove(epoch_checkpoint_path)

        if epoch + 1 == EPOCHS:
            info_logger.info("Final epoch reached. Loss history:")
            loss_df = pd.read_csv(loss_csv_path)
            for idx, row in loss_df.iterrows():
                info_logger.info(f"Epoch {int(row['Epoch'])}: Average Loss = {row['Average_Loss']}")

        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()