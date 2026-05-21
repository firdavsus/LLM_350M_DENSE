import json
import matplotlib.pyplot as plt
import os

# 1. Load trainer state
# checkpoint_path = "model_save/checkpoint-1061189/trainer_state.json"
checkpoint_path =  "ft_out/checkpoint-10263/trainer_state.json"
with open(checkpoint_path, "r") as f:
    data = json.load(f)

logs = data.get("log_history", [])

# 2. Separate logs by key to avoid length mismatch
train_logs = [e for e in logs if "loss" in e]
eval_logs = [e for e in logs if "eval_loss" in e]

# Extract Training data
train_steps = [e["step"] for e in train_logs]
train_losses = [e["loss"] for e in train_logs]
lrs = [e["learning_rate"] for e in train_logs]
grad_norms = [e["grad_norm"] for e in train_logs]

# Extract Eval data
eval_steps = [e["step"] for e in eval_logs]
eval_losses = [e["eval_loss"] for e in eval_logs]

# 3. Create output folder
file_path =  "training_curves_finetune.png"

# 4. Plotting
plt.figure(figsize=(18, 5))

# Subplot 1: Training vs Eval Loss
plt.subplot(1, 3, 1)
plt.plot(train_steps, train_losses, label="Train Loss", color="red", alpha=0.5)
if eval_losses:
    # We use markers ('o') because eval steps are much less frequent
    plt.plot(eval_steps, eval_losses, label="Eval Loss", color="black", marker='o', linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss Convergence")
plt.legend()
plt.grid(True)

# Subplot 2: Learning Rate
plt.subplot(1, 3, 2)
plt.plot(train_steps, lrs, color="blue")
plt.xlabel("Step")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.grid(True)

# Subplot 3: Gradient Norm
plt.subplot(1, 3, 3)
plt.plot(train_steps, grad_norms, color="green")
plt.xlabel("Step")
plt.ylabel("Grad Norm")
plt.title("Gradient Stability")
plt.grid(True)

plt.tight_layout()
plt.savefig(file_path, dpi=300)
print(f"Plot saved to {file_path}")
plt.close()
