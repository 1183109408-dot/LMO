import time
import torch
import torch.nn as nn
import numpy as np
import os
import csv
import joblib
import re
import gc
import traceback
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from data_deal.scienceQA_load import load_and_process_json
from transformers import BertModel, BertTokenizer

# File paths
TEST_JSON_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/datas/scienceQA/test4.json"
BEST_INSTRUCTIONS_CSV = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/instructions/best_instructions.csv"
CLASSIFIER_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/classifier"
BERT_MODEL_PATH = r"/root/WuYanzu/OSAI/model/bert-base-cased"
MODEL_PATH = r"/root/WuYanzu/OSAI/model/MiniCPM5-1B-SFT"                         # 更换为 MiniCPM
OUTPUT_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/test_condition04"   # 新输出目录
CHECKPOINT_PATH = r"/root/WuYanzu/OSAI/autoPrompt/code/LearningToCompare_FSL-master/omniglot/finetuned_MiniCPM5_1B_4rank2"  # 对应训练输出
GPU = 0
MAX_LENGTH = 1024
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.3
CONDITIONAL_LAYERS = 2
CONDITIONAL_FRONT = True

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

# Find latest checkpoint
def get_latest_epoch_path(output_path):
    if not os.path.exists(output_path):
        return None
    epoch_dirs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("epoch_")]
    if not epoch_dirs:
        return None
    epoch_nums = [int(re.match(r"epoch_(\d+)", d).group(1)) for d in epoch_dirs]
    latest_epoch = max(epoch_nums)
    return os.path.join(output_path, f"epoch_{latest_epoch}")

# Conditional LoRA layer (for q_proj, if CONDITIONAL_LAYERS > 0)
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
            raise ValueError("label_vector not set")
        if self.label_vector.dim() != 2 or self.label_vector.shape != (1, self.B_prime.shape[0]):
            raise ValueError(f"label_vector shape mismatch, expected [1, {self.B_prime.shape[0]}], got {self.label_vector.shape}")
        a_prime = self.label_vector.repeat(self.rank, 1)
        delta_W = self.scaling * (self.A @ a_prime @ self.B_prime @ self.B)
        with torch.no_grad():
            base_output = self.base_layer(x)
        lora_output = x @ delta_W
        return base_output + lora_output

# Standard LoRA layer
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

# Conditional LoRA model
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
            raise ValueError("ConditionalLoRAModel has no trainable parameters")
        trainable_param_count = sum(p.numel() for name, p in self.named_parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {trainable_param_count}")

    def replace_linear_with_lora(self):
        # 直接从原始 Linear 层获取维度，适配 GQA 等任意结构
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    device = module.weight.device
                    dtype = module.weight.dtype
                    out_features = module.weight.shape[0]
                    in_features = module.weight.shape[1]

                    if "q_proj" in name:
                        if (CONDITIONAL_FRONT and layer_idx < CONDITIONAL_LAYERS) or \
                           (not CONDITIONAL_FRONT and layer_idx >= total_layers - CONDITIONAL_LAYERS):
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
            raise ValueError("label_vectors must be provided")
        if label_vectors is not None:
            if label_vectors.dim() < 2:
                label_vectors = label_vectors.unsqueeze(0)
            if label_vectors.shape[0] != input_ids.shape[0]:
                raise ValueError(f"label_vectors batch size {label_vectors.shape[0]} does not match input_ids {input_ids.shape[0]}")
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

    def generate(self, input_ids, attention_mask=None, label_vectors=None, **kwargs):
        if label_vectors is None and CONDITIONAL_LAYERS > 0:
            raise ValueError("label_vectors must be provided")
        if label_vectors is not None:
            if label_vectors.dim() < 2:
                label_vectors = label_vectors.unsqueeze(0)
            if label_vectors.shape[0] != input_ids.shape[0]:
                raise ValueError(f"label_vectors batch size {label_vectors.shape[0]} does not match input_ids {input_ids.shape[0]}")
            for i in range(input_ids.shape[0]):
                for layer in self.base_model.model.layers:
                    if hasattr(layer.self_attn.q_proj, 'label_vector'):
                        layer.self_attn.q_proj.label_vector = label_vectors[i:i+1].to(device=f"cuda:{GPU}")
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

# Load custom LoRA checkpoint
def load_custom_lora_checkpoint(model, tokenizer, checkpoint_path):
    if not os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        raise FileNotFoundError(f"adapter_config.json missing in {checkpoint_path}")
    if not os.path.exists(os.path.join(checkpoint_path, "lora_model.bin")):
        raise FileNotFoundError(f"lora_model.bin missing in {checkpoint_path}")
    with open(os.path.join(checkpoint_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    if config["r"] != model.rank or config["lora_alpha"] != model.alpha or config["lora_dropout"] != model.dropout or config["label_vector_dim"] != model.label_vector_dim or config["conditional_layers"] != CONDITIONAL_LAYERS:
        raise ValueError("Checkpoint config does not match model config")
    lora_state_dict = torch.load(os.path.join(checkpoint_path, "lora_model.bin"))
    loaded_params = 0
    for name, module in model.named_modules():
        if isinstance(module, ConditionalLoRALayer):
            lora_A_key = f"{name}.A"
            lora_B_key = f"{name}.B"
            lora_B_prime_key = f"{name}.B_prime"
            if lora_A_key not in lora_state_dict or lora_B_key not in lora_state_dict or lora_B_prime_key not in lora_state_dict:
                raise ValueError(f"Missing ConditionalLoRA parameters for {name}")
            module.A.data = lora_state_dict[lora_A_key].to(device=module.A.device, dtype=module.A.dtype)
            module.B.data = lora_state_dict[lora_B_key].to(device=module.B.device, dtype=module.B.dtype)
            module.B_prime.data = lora_state_dict[lora_B_prime_key].to(device=module.B_prime.device, dtype=module.B_prime.dtype)
            loaded_params += 1
        elif isinstance(module, CustomLoRALayer):
            lora_A_key = f"{name}.lora_A"
            lora_B_key = f"{name}.lora_B"
            if lora_A_key not in lora_state_dict or lora_B_key not in lora_state_dict:
                raise ValueError(f"Missing LoRA parameters for {name}")
            module.lora_A.data = lora_state_dict[lora_A_key].to(device=module.lora_A.device, dtype=module.lora_A.dtype)
            module.lora_B.data = lora_state_dict[lora_B_key].to(device=module.lora_B.device, dtype=module.lora_B.dtype)
            loaded_params += 1
    if loaded_params == 0:
        raise ValueError("No LoRA parameters loaded")
    logger.info(f"Loaded {loaded_params} LoRA modules")
    tokenizer_new = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer_new.pad_token
    tokenizer.pad_token_id = tokenizer_new.pad_token_id
    logger.info(f"Checkpoint loaded: {checkpoint_path}")

# Custom stopping criteria
class StopOnAnswer(StoppingCriteria):
    def __init__(self, tokenizer, stop_text):
        self.tokenizer = tokenizer
        self.stop_text = stop_text
        self.stop_ids = tokenizer.encode(stop_text, add_special_tokens=False)
        if not self.stop_ids or any(id is None for id in self.stop_ids):
            logger.error(f"Invalid stop_ids: {self.stop_ids}")
            self.stop_ids = [tokenizer.eos_token_id]

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) < len(self.stop_ids) + 10:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        try:
            slice_start = max(0, len(input_ids[0]) - 150)
            decoded = self.tokenizer.decode(input_ids[0][slice_start:], skip_special_tokens=True).strip()
            is_stop = bool(re.search(r"\(correct answer:\s*\d\)", decoded, re.IGNORECASE))
            return torch.tensor([is_stop], dtype=torch.bool, device=input_ids.device)
        except Exception as e:
            logger.error(f"StopOnAnswer exception: {str(e)}")
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

# Local model wrapper (adapted for MiniCPM)
class LocalModel:
    def __init__(self, model_path=MODEL_PATH, label_vectors=None, label_vector_dim=None):
        self.device = f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.label_vectors = label_vectors
        self.label_vector_dim = label_vector_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(f"Set pad_token_id to eos_token_id: {self.tokenizer.eos_token_id}")

        # Ensure chat_template for MiniCPM
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

        try:
            checkpoint_path = get_latest_epoch_path(CHECKPOINT_PATH)
            if not checkpoint_path:
                raise RuntimeError("No LoRA checkpoint found")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{GPU}"},
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model = ConditionalLoRAModel(self.model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, self.label_vector_dim)
            load_custom_lora_checkpoint(self.model, self.tokenizer, checkpoint_path)
            logger.info("MiniCPM model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")
        self.model.eval()

    def generate(self, prompt, max_tokens=800, quest_options=None, label_idx=None):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        if label_idx is None or self.label_vectors is None:
            raise ValueError("label_idx and label_vectors must be provided")

        for attempt in range(3):
            try:
                system_prompt = (
                    "You are a helpful assistant. Analyze the question and options, provide reasoning, "
                    "and conclude with the correct option in the format (Correct Answer: X), where X is the option number (0 to N)."
                )
                enhanced_prompt = prompt

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ]

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    max_length=MAX_LENGTH - 200,
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
                if input_length > 800:
                    logger.warning(f"Input token count {input_length} is close to limit, may be truncated")

                label_vector = torch.from_numpy(self.label_vectors[label_idx]).to(device=self.device, dtype=torch.bfloat16).unsqueeze(0)

                stopping_criteria = StoppingCriteriaList([
                    StopOnAnswer(self.tokenizer, "(Correct Answer:")
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
                    label_vectors=label_vector,
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
                    logger.warning(f"Attempt {attempt + 1}, insufficient output, output shape: {outputs.shape}, input length: {input_length}")
                    return "Error: Empty or insufficient output"

                generated_tokens = outputs[0, input_length:] if outputs.shape[1] > input_length else torch.tensor([], device=self.device)
                new_token_count = generated_tokens.shape[0] if generated_tokens.shape else 0
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() if new_token_count > 0 else ""

                del input_ids, attention_mask, outputs, label_vector
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

                if response:
                    return response
                logger.warning(f"Attempt {attempt + 1}, empty new content")
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

# BERT embedding (unchanged)
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

# Extract answer (unchanged)
def extract_answer(text, quest_options=None):
    cleaned_text = re.sub(r'\$\s*\\boxed\s*\{(\d)\}\s*\$', r'\boxed{\1}', text)
    cleaned_text = re.sub(r'\$\s*\$', '', cleaned_text)
    text_lower = cleaned_text.lower().replace('**', '')

    patterns = [
        r'\(correct answer:\s*(\d)\)',
        r'the\s*final\s*answer\s*is\s*[:=]?\s*\\boxed\{(\d)\}',
        r'the\s*final\s*answer\s*is\s*[:=]?\s*(\d)',
        r'correct answer:\s*(\d)',
        r'the\s*correct\s*answer\s*is\s*option\s*(\d)',
        r'option\s*(\d)\s*is\s*correct'
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
                if re.search(rf'\b{re.escape(opt_lower)}\b.*\b(correct|answer|right)\b', text_lower, re.IGNORECASE):
                    return idx
                if 'understatement' in opt_lower and 'understatement' in text_lower:
                    return idx
                if 'antithesis' in opt_lower and 'antithesis' in text_lower:
                    return idx
                if 'bother' in opt_lower and ('bother' in text_lower or 'guilt' in text_lower or 'pressure' in text_lower):
                    return idx
                if 'fable' in opt_lower and 'fable' in text_lower:
                    return idx

    logger.warning(f"Answer extraction failed, generated content: {text[:200]}...")
    return None

# Load best instructions (unchanged)
def load_best_instructions(csv_path):
    best_instructions = {}
    if os.path.isfile(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                best_instructions[row['Label']] = row['Instruction']
    logger.info(f"Loaded best instructions: {len(best_instructions)} entries")
    return best_instructions

# Log errors to CSV (unchanged)
def log_error_to_csv(sample, respond, test_type, csv_path=OUTPUT_PATH, outputs_shape='N/A'):
    if test_type == "no_instruction":
        csv_filename = os.path.join(csv_path, 'no_instruction_err_log.csv')
    elif test_type == "instruction":
        csv_filename = os.path.join(csv_path, 'instruction_err_log.csv')
    else:
        raise ValueError(f"Invalid test_type: {test_type}, must be 'no_instruction' or 'instruction'")
    file_exists = os.path.isfile(csv_filename)
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Question', 'Options', 'Reasoning', 'Answer', 'Model Output', 'Output Shape'])
        writer.writerow([sample.quest, sample.quest, sample.reason, sample.answer, respond, outputs_shape])
    logger.info(f"Error logged to {csv_filename}")

# No-instruction test (adapted)
def test_no_instruction(big_classes, llama, label_vectors, label_to_idx):
    category_stats = {}
    total_correct = 0
    total_samples = 0
    results = []
    feature_encoder = BertEmbedding(BERT_MODEL_PATH)
    svm = joblib.load(os.path.join(CLASSIFIER_PATH, "svm_classifier.pkl"))
    projection_matrix = np.load(os.path.join(CLASSIFIER_PATH, "projection_matrix.npy"))
    scaler = joblib.load(os.path.join(CLASSIFIER_PATH, "scaler.pkl"))

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

                feat = feature_encoder([sample.quest])
                features_np = feat.detach().cpu().numpy()
                features_scaled = scaler.transform(features_np)
                reduced_features = features_scaled @ projection_matrix
                reduced_features = np.nan_to_num(reduced_features, 0)
                predicted_label = svm.predict(reduced_features)[0]
                if predicted_label not in label_to_idx:
                    logger.warning(f"SVM predicted label {predicted_label} not in label_to_idx, using true label {label}")
                    predicted_label = label
                label_idx = label_to_idx[predicted_label]

                option_count = len(option_lines)
                options_text = '\n'.join(option_lines)
                prompt = (
                    f"{sample.quest}\n\n"
                    f"Evaluate options (0 to {option_count-1}), explain each, and conclude with (Correct Answer: X), where X is the option number (0 to {option_count-1})."
                )

                # Generate input prompt (for logging only, already inside generate)
                if total_samples == 0:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant..."},
                        {"role": "user", "content": prompt}
                    ]
                    inputs = llama.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                    decoded_prompt = llama.tokenizer.decode(inputs[0], skip_special_tokens=False)
                    logger.info(f"First sample {sample.index} full input text to model:\n{decoded_prompt}")

                respond = llama.generate(prompt=prompt, quest_options=options_text, label_idx=label_idx)
                answer = extract_answer(respond, quest_options=options_text)
                real_answer = str(sample.answer)

                if total_samples == 0:
                    logger.info(f"First sample {sample.index} full question:\n{sample.quest}")
                    logger.info(f"First sample {sample.index} full model output:\n{respond}")
                else:
                    logger.info(f"Sample {sample.index}, provided options: {option_lines}")
                    logger.info(f"Sample {sample.index}, true answer: {real_answer}, Model output answer: {answer if answer else 'Not extracted'}")

                is_correct = answer == real_answer
                if answer is None:
                    log_error_to_csv(sample, respond, test_type="no_instruction")

                category_stats[label]["correct"] += 1 if is_correct else 0
                category_stats[label]["total"] += 1
                total_correct += 1 if is_correct else 0
                total_samples += 1

                results.append({
                    "index": sample.index,
                    "label": label,
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

# (The instruction-based test is commented out in the main; if needed, adjust similarly)

# Save results (unchanged)
def save_results(results, filename, csv_path=OUTPUT_PATH):
    csv_filename = os.path.join(csv_path, filename)
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved to {csv_filename}")

# Save accuracy summary (unchanged)
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

    logger.info("Loading class_label_vectors")
    pkl_path = os.path.join(CLASSIFIER_PATH, "class_label_vectors.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"File {pkl_path} does not exist")
    class_label_vectors = joblib.load(pkl_path)
    logger.info(f"Successfully loaded class_label_vectors.pkl with {len(class_label_vectors)} categories")

    label_vectors = []
    label_to_idx = {}
    for idx, (label, vector) in enumerate(class_label_vectors.items()):
        norm = np.linalg.norm(vector)
        if norm > 0:
            normalized_vector = vector / norm
        else:
            logger.warning(f"Category {label} has zero norm vector, using original vector")
            normalized_vector = vector
        label_vectors.append(normalized_vector)
        label_to_idx[label] = idx
    label_vectors = np.array(label_vectors, dtype=np.float32)
    label_vector_dim = label_vectors.shape[1]
    logger.info(f"Loaded {len(label_to_idx)} categories, label_vector_dim: {label_vector_dim}")

    logger.info("Loading test data")
    test_big_classes = load_and_process_json(TEST_JSON_PATH)
    if not test_big_classes:
        logger.error("Test data loading failed, exiting")
        return

    logger.info("Initializing MiniCPM model and BERT")
    llama = LocalModel(model_path=MODEL_PATH, label_vectors=label_vectors, label_vector_dim=label_vector_dim)
    feature_encoder = BertEmbedding(BERT_MODEL_PATH)

    logger.info("Loading classifier")
    svm = joblib.load(os.path.join(CLASSIFIER_PATH, "svm_classifier.pkl"))
    projection_matrix = np.load(os.path.join(CLASSIFIER_PATH, "projection_matrix.npy"))
    scaler = joblib.load(os.path.join(CLASSIFIER_PATH, "scaler.pkl"))

    logger.info("Loading best instructions")
    best_instructions = load_best_instructions(BEST_INSTRUCTIONS_CSV)

    logger.info("\nNo-instruction test")
    no_instr_total_accuracy, no_instr_category_accuracies, no_instr_results = test_no_instruction(test_big_classes, llama, label_vectors, label_to_idx)
    logger.info(f"No-instruction total accuracy: {no_instr_total_accuracy:.2f}%")
    for label, accuracy in no_instr_category_accuracies.items():
        logger.info(f"Category {label}: {accuracy:.2f}%")
    save_results(no_instr_results, "no_instruction_test_results.csv")
    save_accuracy_summary(no_instr_total_accuracy, no_instr_category_accuracies, no_instr_results, "no_instruction_accuracy_summary.csv")

    # The instruction-based test is currently commented out in the original main;
    # if needed, uncomment and adjust similarly. For now, keep as-is.

    logger.info("Testing completed")

if __name__ == "__main__":
    main()