import torch
import gc
import math
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from torch.utils.data import DataLoader, IterableDataset

from model import LLM, Config 


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class BinDataset(IterableDataset):
    def __init__(self, data_path, block_size, sos_token_id):
        self.data_path   = data_path
        self.block_size  = block_size
        self.sos_token_id = sos_token_id

    def __iter__(self):
        data         = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        total_tokens = len(data)
        stride       = self.block_size - 1          # one slot reserved for BOS

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx, end_idx = 0, total_tokens - stride
        else:
            per_worker = (total_tokens - stride) // worker_info.num_workers
            start_idx  = worker_info.id * per_worker
            end_idx    = start_idx + per_worker
            if start_idx % stride != 0:
                start_idx += stride - (start_idx % stride)

        for i in range(start_idx, end_idx, stride):
            chunk   = data[i : i + stride].astype(np.int64)
            t_data  = torch.from_numpy(chunk)

            sos_tensor = torch.tensor([self.sos_token_id], dtype=torch.long)
            input_ids  = torch.cat([sos_tensor, t_data])   # [block_size]
            labels     = input_ids.clone()

            yield {"input_ids": input_ids, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# WSD Learning-Rate Schedule
# ─────────────────────────────────────────────────────────────────────────────
def get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay, min_lr_ratio=0.1):
    """
    Warmup → Stable → Cosine-Decay schedule.
    Replaces the plain cosine scheduler that was listed in TrainingArguments
    (which had no effect once custom optimizers were passed anyway, causing
    silent confusion).
    """
    def lr_lambda(step):
        if step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if step < num_warmup + num_stable:
            return 1.0
        progress = float(step - num_warmup - num_stable) / float(max(1, num_decay))
        progress = min(1.0, progress)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer subclass (custom DataLoader for IterableDataset)
# ─────────────────────────────────────────────────────────────────────────────
class LLMTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=8,
            pin_memory=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

config = Config()
model  = LLM(config)

# HuggingFace Trainer compatibility
model.can_generate       = lambda: True
model.config             = config
model.config.model_type  = "llm"

def _to_json_string(self):
    import json
    return json.dumps(self.__dict__, indent=2)

Config.to_json_string = _to_json_string

model = torch.compile(model)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
# Step / epoch counting
# ─────────────────────────────────────────────────────────────────────────────
BIN_PATH    = "data_mixed/train.bin"
raw_data    = np.memmap(BIN_PATH, dtype=np.uint16, mode="r")

total_sequences = len(raw_data) // config.block_size
eff_batch_size  = config.batch_size * config.grad_acc

steps_per_epoch = total_sequences // eff_batch_size
total_max_steps = steps_per_epoch * config.num_train_epochs

print(f"Total sequences : {total_sequences:,}")
print(f"Steps per epoch : {steps_per_epoch:,}")
print(f"Total steps     : {total_max_steps:,}")
print(f"Batch size      : {config.batch_size}")

# ─────────────────────────────────────────────────────────────────────────────
# Optimiser + Scheduler   (FIX 4: created exactly ONCE)
# ─────────────────────────────────────────────────────────────────────────────
num_warmup = config.warm_up
num_decay  = int(total_max_steps * 0.15)
num_stable = total_max_steps - num_warmup - num_decay

optimizer  = model.configure_optimizers(
    config.weight_decay, config.learning_rate, config.betas, "cuda"
)
scheduler  = get_wsd_schedule(optimizer, num_warmup, num_stable, num_decay)
optimizers = (optimizer, scheduler)  

# ─────────────────────────────────────────────────────────────────────────────
# TrainingArguments
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./model_save",

    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.grad_acc,

    max_steps=total_max_steps,
    num_train_epochs=config.num_train_epochs,
    warmup_steps=config.warm_up,
    max_grad_norm=1.0,

    learning_rate=config.learning_rate,
    lr_scheduler_type="constant",

    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_steps=5_000,
    eval_steps=5_000,
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    save_total_limit=5,
    weight_decay=config.weight_decay,
    remove_unused_columns=False,
    save_safetensors=False,
    gradient_checkpointing=False,
    prediction_loss_only=True,
    label_names=["labels"],
    metric_for_best_model="loss",
    greater_is_better=False,
)

# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────
sos_id = (
    tokenizer.bos_token_id
    if tokenizer.bos_token_id is not None
    else tokenizer.eos_token_id
)

train_dataset = BinDataset(BIN_PATH,                    config.block_size, sos_id)
eval_dataset  = BinDataset("data_mixed/test.bin",       config.block_size, sos_id)

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
trainer = LLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    optimizers=optimizers,
)

trainer.train()