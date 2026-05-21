import inspect
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput
import math

class Config:
    drop = 0.0
    dim = 1024
    heads = 16
    layers = 24
    ffn_dim = 2736
    block_size = 2048
    emb_num = 50432

    batch_size: int = 32
    grad_acc: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.98)
    warm_up: int = 4000

    eos_token_id = 0
    bos_token_id = 0
    pad_token_id = 0


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Embedding
# ─────────────────────────────────────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, -1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ─────────────────────────────────────────────────────────────────────────────
# MLP  (SwiGLU gated variant)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, 2 * config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.drop)

        self.fc2.is_residual_proj = True

    def forward(self, x):
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return self.dropout(self.fc2(x))


# ─────────────────────────────────────────────────────────────────────────────
# Self-Attention with RoPE
# ─────────────────────────────────────────────────────────────────────────────
class XSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.heads = config.dim, config.heads
        self.d_k = self.dim // self.heads
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.Wo = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RotaryEmbedding(self.d_k)
        self.attn_dropout = nn.Dropout(config.drop)

        self.Wq.is_attention = True
        self.Wk.is_attention = True
        self.Wv.is_attention = True
        self.Wo.is_residual_proj = True

    def forward(self, x):
        B, T, D = x.shape
        Q = self.Wq(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        Q, K = self.rope(Q, K, T)

        att_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        return self.Wo(att_output.transpose(1, 2).contiguous().view(B, T, D))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.dim)
        self.attn = XSA(config)
        self.ln_2 = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────
class LLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.emb_num, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm_f = nn.RMSNorm(config.dim)

        self.lm_head = nn.Linear(config.dim, config.emb_num, bias=False)

        # Weight tying
        self.embeddings.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print("Number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters()) / 1e6,))

    @torch.no_grad()
    def _init_weights(self, module):
        n_layer = self.config.layers
        
        if isinstance(module, nn.Linear):
            if module is self.lm_head:
                return 
            
            w_fan_in = module.weight.shape[-1]
            base_std = (1.0 / w_fan_in) ** 0.5

            if hasattr(module, 'is_residual_proj'):
                final_std = base_std / math.sqrt(2 * n_layer)
            elif hasattr(module, 'is_attention'):
                final_std = base_std * 0.7
            else:
                final_std = base_std

            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=final_std, a=-2*final_std, b=2*final_std
            )
            
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(
                module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04
            )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params   = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors:     {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        x = self.embeddings(input_ids)          # [B, T, D]

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # ── FIX 3: label_smoothing removed for pretraining ───────────────
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=0.0,   # 0.0 for pretraining; use 0.1 at fine-tune
                reduction="mean",
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutput(loss=loss, logits=logits)