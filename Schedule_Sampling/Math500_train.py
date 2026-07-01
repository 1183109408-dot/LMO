import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from Math500_load import load_and_process_json
import pandas as pd
from datetime import datetime

# ==================== 配置区（100%对齐第一份DeepSeek配置） ====================
NUM_EPOCHS = 3
# 模型路径与第一份一致
MODEL_BASE_PATH = r"/root/WuYanzu/DPO/model/deepseek-math-7b-base"
JSON_PATH = r"/root/WuYanzu/DPO/data/Math500/train.json"
# 保存目录对齐第一份的正则强度LAMBDA_REG=3.0
SAVE_DIR = r"/root/WuYanzu/DPO/checkpoints/checkpoints_Math500_deepseek_posMask6_reg5"
os.makedirs(SAVE_DIR, exist_ok=True)
csv_save_path = os.path.join(SAVE_DIR, "training_metrics.csv")

LAMBDA_REG = 5.0
K_ALPHA = 1.0
ALPHA_MIN = 0.001
ALPHA_MAX = 1.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(f"Using device: {device}")
print(f"Training metrics will be saved to: {csv_save_path}")
print(f"Using LAMBDA_REG = {LAMBDA_REG} for delta_p regularization")

# ==================== Dataset（100%对齐第一份DeepSeek官方Prompt格式） ====================
class Math500Dataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer
        
        # 完全替换为第一份的DeepSeek风格Prompt（核心修正）
        self.prompt_prefix = (
            "### User:\n"
            "Solve the math problem step by step, show detailed reasoning, and put the final answer in \\boxed{}.\n\n"
            "### Problem:\n"
        )
        # 完全替换为第一份的DeepSeek回答前缀
        self.answer_prefix = "\n\n### Assistant:\n"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 文本构造逻辑与第一份完全一致
        prompt_text = self.prompt_prefix + sample.quest
        answer_text = self.answer_prefix + sample.reason + "\n\n" + f"\\boxed{{{sample.answer}}}"

        # 编码逻辑与第一份完全一致
        prompt_enc = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        prompt_ids = prompt_enc["input_ids"].squeeze(0)
        
        answer_enc = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=True,
            max_length=1024 - len(prompt_ids),
            return_tensors="pt"
        )
        answer_ids = answer_enc["input_ids"].squeeze(0)

        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        if len(input_ids) > 1024:
            input_ids = input_ids[:1024]
        seq_len = len(input_ids)

        labels = torch.ones_like(input_ids) * -100
        prompt_len = min(len(prompt_ids), seq_len)
        labels[prompt_len:] = input_ids[prompt_len:]

        return {"input_ids": input_ids, "labels": labels}

# ==================== 主程序（对齐第一份参数 + 保留第二份核心方法） ====================
def main():
    samples = load_and_process_json(JSON_PATH)
    print(f"Loaded {len(samples)} samples")
    
    # ==================== Tokenizer（100%对齐第一份DeepSeek参数） ====================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_BASE_PATH,
        local_files_only=True,
        trust_remote_code=True,
        # 显式对齐第一份的参数
        padding_side="left",
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = Math500Dataset(samples, tokenizer)

    # 打印第一条样本（与第一份逻辑完全一致）
    if len(dataset) > 0:
        first_sample = dataset[0]
        first_text = tokenizer.decode(first_sample['input_ids'], skip_special_tokens=False)
        print("\n第一个样本的处理后训练文本：")
        print("=" * 80)
        print(first_text)
        print("=" * 80 + "\n")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    # ==================== 模型加载（100%对齐第一份DeepSeek参数） ====================
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False,
        low_cpu_mem_usage=True
    )
    
    base_model.gradient_checkpointing_enable()

    # LoRA配置（与第一份完全一致）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    # 断点续训（与第一份逻辑完全一致）
    existing_epochs = [int(d.split("_")[-1]) for d in os.listdir(SAVE_DIR)
                       if d.startswith("adapter_epoch_") and os.path.isdir(os.path.join(SAVE_DIR, d))]
    start_epoch = max(existing_epochs) + 1 if existing_epochs else 1
    if start_epoch > 1:
        print(f"Resuming from epoch {start_epoch}...")
        model = PeftModel.from_pretrained(base_model, os.path.join(SAVE_DIR, f"adapter_epoch_{start_epoch-1}"))
    else:
        model = get_peft_model(base_model, lora_config)
        print("Starting from scratch.")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # 嵌入层兼容性代码（保留第二份的健壮实现，避免第一份硬编码报错）
    try:
        embedding_weight = model.model.embed_tokens.weight.detach()
    except AttributeError:
        try:
            embedding_weight = model.model.model.embed_tokens.weight.detach()
        except AttributeError:
            embedding_weight = model.transformer.wte.weight.detach()
        
    t_global = 1.0
    global_step = 0

    log_interval = 50
    # 统计器保留第二份的方法相关指标
    accum = {
        "loss": 0.0, "penalty": 0.0, "delta_p": 0.0, "delta_t": 0.0,
        "ph": 0.0, "ps": 0.0, "t_global": 0.0, "avg_alpha": 0.0,
        "max_alpha": 0.0, "min_alpha": 0.0
    }
    count = 0
    csv_data = []
    if not os.path.exists(csv_save_path):
        pd.DataFrame(columns=[
            'epoch','batch_start','batch_end','loss','penalty','avg_delta_p',
            'delta_t','t_global','avg_alpha','avg_max_alpha','avg_min_alpha',
            'avg_p_hard','avg_p_soft','timestamp'
        ]).to_csv(csv_save_path, index=False)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n{'='*40} EPOCH {epoch}/{NUM_EPOCHS} DeepSeek-Math-7B-Base + Token-Level-WGAPS + Pos Alpha {'='*40}")
        for batch in dataloader:
            global_step += 1
            count += 1

            input_ids = batch["input_ids"][0].to(device)
            labels = batch["labels"][0].to(device)
            seq_len = input_ids.shape[0]

            # Step 1: Hard Pass（保留第二份方法逻辑）
            with torch.no_grad():
                outputs = model(input_ids=input_ids.unsqueeze(0))
                hard_logits = outputs.logits[0, :-1, :]
                hard_probs = F.softmax(hard_logits, dim=-1)
                
                target_labels = labels[1:]
                valid_mask = (target_labels != -100)
                valid_target_tokens = target_labels[valid_mask].unsqueeze(-1)
                valid_hard_probs = hard_probs[valid_mask]
                
                if valid_target_tokens.shape[0] == 0:
                    print(f"Batch {global_step}: No valid target tokens, skip")
                    count -= 1
                    continue
                p_hard = valid_hard_probs.gather(-1, valid_target_tokens).squeeze(-1)

            # 位置权重计算（保留第二份方法逻辑）
            target_seq_len = valid_target_tokens.shape[0]
            target_positions = torch.arange(0, target_seq_len, dtype=torch.float32, device=device)
            target_pos_weight = torch.exp(-0.5 * target_positions / target_seq_len)
            target_pos_weight = target_pos_weight / target_pos_weight.mean()
            
            t_token = torch.ones(seq_len-1, dtype=torch.float32, device=device)
            valid_pos_indices = torch.where(valid_mask)[0]
            assign_len = min(len(valid_pos_indices), len(target_pos_weight))
            for i in range(assign_len):
                t_token[valid_pos_indices[i]] = torch.clamp(t_global * target_pos_weight[i], 0.0, 1.0)

            # Soft embeds（对齐第一份的写法：detach().requires_grad_(True)）
            if hasattr(model.model, 'embed_tokens'):
                input_embeds = model.model.embed_tokens(input_ids.unsqueeze(0))
            elif hasattr(model.model.model, 'embed_tokens'):
                input_embeds = model.model.model.embed_tokens(input_ids.unsqueeze(0))
            else:
                input_embeds = model.transformer.wte(input_ids.unsqueeze(0))
            
            soft_embeds = input_embeds.clone()
            
            target_abs_pos_list = valid_pos_indices + 1
            idx = 0
            for pos in target_abs_pos_list:
                if torch.rand(1, device=device) < t_token[pos-1]:
                    idx += 1
                    continue
                if idx < target_seq_len:
                    soft_embeds[0, pos] = hard_probs[valid_mask][idx] @ embedding_weight
                idx += 1

            # 对齐第一份的soft_embeds写法
            soft_embeds = soft_embeds.detach().requires_grad_(True)
            soft_out = model(inputs_embeds=soft_embeds)
            soft_logits = soft_out.logits[0, :-1, :]
            soft_probs = F.softmax(soft_logits, dim=-1)
            
            valid_soft_probs = soft_probs[valid_mask]
            p_soft = valid_soft_probs.gather(-1, valid_target_tokens).squeeze(-1)

            # ==================== 核心方法：Token级Alpha + delta_p正则（完全保留第二份） ====================
            delta_p = p_hard - p_soft
            alpha_token = K_ALPHA * delta_p * target_pos_weight
            alpha_token = torch.clamp(alpha_token, min=ALPHA_MIN, max=ALPHA_MAX)
            
            avg_alpha_batch = alpha_token.mean().item()
            max_alpha_batch = alpha_token.max().item()
            min_alpha_batch = alpha_token.min().item()
            delta_p_seq_mean = delta_p.mean().item()

            # Token级损失计算（完全保留第二份）
            prob_loss = (1.0 - p_soft ** alpha_token) / (alpha_token + 1e-8)
            ce_loss = prob_loss.mean()
            penalty = torch.exp(-ce_loss).item()

            reg_loss = LAMBDA_REG * delta_p_seq_mean
            total_loss = ce_loss + reg_loss

            # delta_t更新（对齐第一份的.item()写法，避免张量累加）
            eta = 5e-5
            delta_t = eta * (LAMBDA_REG * delta_p_seq_mean - t_global)
            t_global += delta_t
            t_global = max(0.01, min(0.99, t_global))

            # 反向传播（与第一份逻辑完全一致）
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 日志统计（保留第二份的方法相关指标）
            accum["loss"] += total_loss.item()
            accum["penalty"] += penalty
            accum["delta_p"] += delta_p_seq_mean
            accum["delta_t"] += delta_t
            accum["ph"] += p_hard.mean().item()
            accum["ps"] += p_soft.mean().item()
            accum["t_global"] += t_global
            accum["avg_alpha"] += avg_alpha_batch
            accum["max_alpha"] += max_alpha_batch
            accum["min_alpha"] += min_alpha_batch

            if global_step % log_interval == 0 and count > 0:
                avg = {k: v / count for k, v in accum.items()}
                print(f"STEP {global_step:5d} │ "
                      f"AVG_LOSS {avg['loss']:.4f} │ AVG_t_g {avg['t_global']:.4f} │ "
                      f"AVG_α {avg['avg_alpha']:.4f} │ MAX_α {avg['max_alpha']:.4f} │ MIN_α {avg['min_alpha']:.4f} │ "
                      f"AVG_PENALTY {avg['penalty']:.4f} │ AVG_Δp {avg['delta_p']:.6f} │ AVG_delta_t {avg['delta_t']:.6f} │ "
                      f"AVG_p_h {avg['ph']:.4f} → AVG_p_s {avg['ps']:.4f}")
                row = {
                    "epoch": epoch,
                    "batch_start": global_step - log_interval + 1,
                    "batch_end": global_step,
                    "loss": round(avg['loss'], 6),
                    "penalty": round(avg['penalty'], 6),
                    "avg_delta_p": round(avg['delta_p'], 7),
                    "delta_t": round(avg['delta_t'], 6),
                    "t_global": round(avg['t_global'], 6),
                    "avg_alpha": round(avg['avg_alpha'], 6),
                    "avg_max_alpha": round(avg['max_alpha'], 6),
                    "avg_min_alpha": round(avg['min_alpha'], 6),
                    "avg_p_hard": round(avg['ph'], 6),
                    "avg_p_soft": round(avg['ps'], 6),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                csv_data.append(row)
                pd.DataFrame(csv_data).to_csv(csv_save_path, mode='a', header=False, index=False)
                csv_data = []
                accum = {k: 0.0 for k in accum}
                count = 0

        save_path = os.path.join(SAVE_DIR, f"adapter_epoch_{epoch}")
        model.save_pretrained(save_path)
        print(f"Epoch {epoch} 完成 → {save_path}")
        if csv_data:
            pd.DataFrame(csv_data).to_csv(csv_save_path, mode='a', header=False, index=False)

    print(f"\n训练结束，所有指标已保存至：{csv_save_path}")

if __name__ == "__main__":
    main()