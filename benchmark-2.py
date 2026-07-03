import torch
import torch.nn as nn
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from model import LLM, Config
from transformers import AutoTokenizer
from lm_eval.api.instance import Instance

# --- 1. LOAD YOUR MODEL ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
model = LLM(config).to(DEVICE)

state_dict = torch.load("ft_out/checkpoint-16712/pytorch_model.bin", map_location=DEVICE)
clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

# Satisy HF internal requirements for lm-eval
if not hasattr(model, "tie_weights"):
    model.tie_weights = lambda: None
if not hasattr(model, "config"):
    model.config = config
model.device = DEVICE

tokenizer = AutoTokenizer.from_pretrained("data_mixed/custom_multilingual_tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# --- 2. WRAP FOR LM-EVAL WITH CHAT FORMATTING ---
class InstructHFLM(HFLM):
    def __init__(self, system_message, **kwargs):
        super().__init__(**kwargs)
        self.system_message = system_message
        self.bos_string = self.tokenizer.bos_token 

    def _format_context(self, context):
        return (
            f"{self.bos_string}System:\n{self.system_message}\n"
            f"User:\n{context.strip()}\n"
            f"Assistant:\n"
        )

    def loglikelihood(self, requests):
        patched_requests = []
        for req in requests:
            context, continuation = req.args
            formatted_context = self._format_context(context)
            
            new_req = Instance(
                request_type=req.request_type,
                doc=req.doc,
                arguments=(formatted_context, continuation),
                idx=req.idx
            )
            
            if hasattr(req, 'metadata'):
                new_req.metadata = req.metadata
            if hasattr(req, 'kwargs'):
                new_req.kwargs = req.kwargs
                
            patched_requests.append(new_req)
            
        return super().loglikelihood(patched_requests)

    def generate_until(self, requests):
        patched_requests = []
        for req in requests:
            context, gen_kwargs = req.args
            formatted_context = self._format_context(context)
            
            new_req = Instance(
                request_type=req.request_type,
                doc=req.doc,
                arguments=(formatted_context, gen_kwargs),
                idx=req.idx
            )
            
            if hasattr(req, 'metadata'):
                new_req.metadata = req.metadata
            if hasattr(req, 'kwargs'):
                new_req.kwargs = req.kwargs
                
            patched_requests.append(new_req)
            
        return super().generate_until(patched_requests)

# Define the persona
SYSTEM_PROMPT = "You are D4, a highly logical AI assistant. You think step-by-step and answer clearly."

# Initialize our custom wrapper
lm_obj = InstructHFLM(
    system_message=SYSTEM_PROMPT,
    pretrained=model,
    tokenizer=tokenizer,
    batch_size=4,
    device=DEVICE
)

# ... Proceed to Step 4 (Run Evaluation) as normal ...

# --- 4. RUN EVALUATION ---
task_list = ["hellaswag", "mmlu", "arc_challenge", "gsm8k"]

print("Starting benchmarks with D4 Chat Formatting...")
results = simple_evaluate(
    model=lm_obj,
    tasks=task_list,
    num_fewshot=2, 
)

# --- 5. PRINT RESULTS ---
print("\n" + "="*50)
print(f"{'Task':<25} | {'Score':<10}")
print("-" * 40)

for task_name, scores in results["results"].items():
    score = scores.get("acc_norm,none", scores.get("acc,none", scores.get("exact_match,none", 0)))
    print(f"{task_name:<25} | {score*100:>8.2f}%")