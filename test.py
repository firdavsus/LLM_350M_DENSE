import torch
from transformers import AutoTokenizer
from model import LLM, Config
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

# Load model configuration
config = Config() 
model = LLM(config).to(device)

# Load fine-tuned checkpoint
state_dict = torch.load(
    "ft_out/checkpoint-10263/pytorch_model.bin", # Point this to your ft_out/ if testing recent fine-tunes
    map_location=device
)

# Remove torch.compile wrapper if present
clean_state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in state_dict.items()
}

# Load weights
model.load_state_dict(clean_state_dict, strict=True)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print("✅ Model loaded successfully")

@torch.no_grad()
def sample_generate(
    model,
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=None,         
    eos_token_id=None,
    repetition_penalty=1.15,
    block_size=2048 # Fallback if config.block_size is named differently
):
    """
    input_ids: (B, T)
    returns:   (B, T + max_new_tokens)
    """
    model.eval()
    
    # Try to grab block size from config to prevent positional embedding crashes
    max_seq_len = getattr(model.config, 'block_size', getattr(model.config, 'max_seq_len', block_size))

    for _ in range(max_new_tokens):
        # Prevent Out-Of-Bounds error by truncating to max sequence length
        context_ids = input_ids[:, -max_seq_len:]

        # Forward pass
        out = model(context_ids)
        logits = out.logits[:, -1, :]  # (B, vocab)

        # --- Repetition Penalty Logic ---
        if repetition_penalty != 1.0:
            for b in range(input_ids.shape[0]):
                for token_id in set(input_ids[b].tolist()):
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

        # Temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K
        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(1, indices, values)

        # Top-P (nucleus)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)

            cutoff = cumulative_probs > top_p
            cutoff[:, 1:] = cutoff[:, :-1].clone()
            cutoff[:, 0] = False

            sorted_logits[cutoff] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Optional EOS stop
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                break

    return input_ids


sos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

system_message = "Your name is D4, you are a highly logical AI assistant. You must follow a strict format: first, write a single brief sentence of your logical reasoning. Second, provide the final answer clearly and concisely. If you do not know the answer, say 'I do not know'."

while True:
    user_input = input(">>> ")
    if user_input.lower() == "stop":
        break

    prompt = (
        f"System:\n{system_message}\n\n"
        f"User:\n{user_input}\n\n"
        f"Assistant:\n"
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    sos_tensor = torch.tensor([[sos_id]], device=device)
    input_ids = torch.cat([sos_tensor, input_ids], dim=1)
    
    out = sample_generate(
        model,
        input_ids,
        max_new_tokens=128,    
        temperature=0.8,        
        top_k=30,                    
        top_p=0.8,                  
        repetition_penalty=1.15,   
        eos_token_id=tokenizer.eos_token_id
    )
    
    input_len = input_ids.shape[1]
    generated_tokens = out[0][input_len:]
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(response.strip())
    print("-" * 40)

print("ez!")