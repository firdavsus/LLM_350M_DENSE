import torch
import math
import json
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import LLM, Config

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
# Chat format
# Standard chat formatting templates.
# Loss is computed ONLY on assistant tokens (everything else is masked -100).
# ─────────────────────────────────────────────────────────────────────────────
SYS_OPEN = "System:\n"
USR_OPEN = "User:\n"
AST_OPEN = "Assistant:\n"

def format_sample(x):
    conversations = x["conversations"]

    # Initialize sequence with a starting EOS token as a boundary indicator
    input_ids = [tokenizer.bos_token_id]
    labels    = [-100]  # First token does not contribute to loss

    for turn in conversations:
        role  = turn["from"]
        value = turn["value"].strip()

        if role == "system":
            # Separate roles via a simple newline instead of overloading EOS
            text = f"{SYS_OPEN}{value}\n"
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(toks)
            labels.extend([-100] * len(toks))

        elif role == "human":
            # Separate roles via a newline. Prompt stays open right at AST_OPEN
            text = f"{USR_OPEN}{value}\n{AST_OPEN}"
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(toks)
            labels.extend([-100] * len(toks))

        elif role == "gpt":
            # Assistant response: compute loss here. EOS token strictly closes the turn.
            text = f"{value}{tokenizer.eos_token}"
            toks = tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(toks)
            labels.extend(toks)

        else:
            continue

    return {"input_ids": input_ids, "labels": labels}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Processing
# ─────────────────────────────────────────────────────────────────────────────
config     = Config()
MAX_LENGTH = config.block_size   # 2048

ds = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
dataset = ds.map(format_sample, remove_columns=ds.column_names, num_proc=4)

def keep_valid(example):
    total_len  = len(example["input_ids"])
    answer_len = sum(1 for t in example["labels"] if t != -100)
    # Filter sequences fitting context window that contain a meaningful target output
    return total_len <= MAX_LENGTH and answer_len >= 10

dataset = dataset.filter(keep_valid, num_proc=4)
print(f"Dataset size after filtering: {len(dataset):,}")

# Train / eval split
eval_dataset  = dataset.select(range(2_000))
train_dataset = dataset.select(range(2_000, len(dataset)))
print(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────
def causal_lm_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    labels    = [torch.tensor(f["labels"],    dtype=torch.long) for f in features]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels    = pad_sequence(labels,    batch_first=True, padding_value=-100)
    
    # Generate attention mask to safely exclude padding tokens from attention calculations
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

# ─────────────────────────────────────────────────────────────────────────────
# Model — Load Pretrained Weights
# ─────────────────────────────────────────────────────────────────────────────
model = LLM(config).to(device)

PRETRAIN_CKPT = "model_save/checkpoint-1061189/pytorch_model.bin"
state = torch.load(PRETRAIN_CKPT, map_location=device)
clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(clean, strict=True)
print("✅ Pretrained checkpoint loaded.")

# HuggingFace Trainer compatibility patches
model.can_generate      = lambda: True
model.config            = config
model.config.model_type = "llm"

def _to_json_string(self):
    return json.dumps(self.__dict__, indent=2)

Config.to_json_string = _to_json_string

# ─────────────────────────────────────────────────────────────────────────────
# Step Counting & Optimization Math
# ─────────────────────────────────────────────────────────────────────────────
PER_DEVICE_BS = 16
GRAD_ACC      = 2
NUM_EPOCHS    = 1

steps_per_epoch = math.ceil(len(train_dataset) / (PER_DEVICE_BS * GRAD_ACC))
total_steps     = steps_per_epoch * NUM_EPOCHS

# Fine-tuning schedule: 5% Warmup | 75% Stable | 20% Decay
num_warmup = max(50, int(0.05 * total_steps))
num_decay  = int(0.20 * total_steps)
num_stable = total_steps - num_warmup - num_decay

print(f"Total FT steps : {total_steps:,}")
print(f"Warmup         : {num_warmup:,}")
print(f"Stable         : {num_stable:,}")
print(f"Decay          : {num_decay:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────
param_dict     = {n: p for n, p in model.named_parameters() if p.requires_grad}
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optimizer = AdamW(
    [
        {"params": decay_params,   "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=2e-5,  # Lowered step rate specifically for SFT to prevent memory collapse
    betas=(0.9, 0.98),
)

# ─────────────────────────────────────────────────────────────────────────────
# WSD Scheduler 
# ─────────────────────────────────────────────────────────────────────────────
MIN_LR_RATIO = 0.1   # Floor learning rate = 2e-6 at final optimization boundary

def wsd_lambda(step):
    if step < num_warmup:
        return float(step) / float(max(1, num_warmup))
    if step < num_warmup + num_stable:
        return 1.0
    progress = float(step - num_warmup - num_stable) / float(max(1, num_decay))
    progress = min(1.0, progress)
    # Cosine decay down to MIN_LR_RATIO
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, wsd_lambda)

# ─────────────────────────────────────────────────────────────────────────────
# TrainingArguments
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./ft_out",

    per_device_train_batch_size=PER_DEVICE_BS,
    per_device_eval_batch_size=PER_DEVICE_BS,
    gradient_accumulation_steps=GRAD_ACC,

    num_train_epochs=NUM_EPOCHS,
    max_grad_norm=0.5,

    learning_rate=2e-5,
    lr_scheduler_type="constant",   # Set to constant; scheduling is manually handled below

    warmup_ratio=0.0,
    warmup_steps=0,

    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_steps=200,
    save_steps=200,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    weight_decay=0.01,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    save_safetensors=False,
    prediction_loss_only=True,
    label_names=["labels"],
    metric_for_best_model="loss",
    greater_is_better=False,
    load_best_model_at_end=True,
)

# Initialize standard Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  
    data_collator=causal_lm_collator,
    optimizers=(optimizer, scheduler),
)

trainer.train()