import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from model import LLM, Config

# ----------------------
# SETTINGS
# ----------------------
BATCH = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# LOAD DATASET
# ----------------------
dataset = load_dataset("hellaswag", split="validation")

# ----------------------
# LOAD TOKENIZER
# ----------------------
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# ----------------------
# LOAD MODEL
# ----------------------
config = Config()
model = LLM(config).to(DEVICE)

state_dict = torch.load(
    "ft_out/checkpoint-10263/pytorch_model.bin",
    # "model_save/checkpoint-1061189/pytorch_model.bin",
    map_location=DEVICE
)
clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict, strict=True)
model.eval()

# ----------------------
# EVALUATION LOOP
# ----------------------
correct = 0
total = 0
sos_id = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
print(sos_id)
print("Start evaluating...")

for i in range(0, len(dataset), BATCH):
    batch = dataset.select(range(i, min(i + BATCH, len(dataset))))

    texts = []
    gold_labels = []

    # Build sequences (BATCH x 4 options)
    for ex in batch:
        ctx = ex["ctx"]
        for opt in ex["endings"]:
            texts.append(sos_id + ctx + " " + opt)
        gold_labels.append(ex["label"])

    # Tokenize
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Mask padding in labels
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        token_loss = token_loss.view(labels[:, 1:].shape)
        seq_loss = token_loss.sum(dim=1) / (labels[:, 1:] != -100).sum(dim=1)

    # Reshape to (BATCH, 4) for 4 options per example
    seq_loss = seq_loss.view(len(batch), 4)

    # Pick the option with lowest loss
    preds = seq_loss.argmin(dim=1).tolist()
    gold_labels = [int(x) for x in gold_labels] 

    # Update accuracy
    for pred, gold in zip(preds, gold_labels):
        correct += int(pred == gold)
        total += 1

    if (i // BATCH) % 100 == 0:
        print(f"Processed {i} / {len(dataset)} examples")
        print(correct, total)

accuracy = correct / total
print("HellaSwag accuracy:", accuracy)
