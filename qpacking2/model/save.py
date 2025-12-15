"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/30

# Description: 
# ------------------------------------------------------------------------------
"""
import os
import torch
from transformers import TrainerCallback


class SaveCompleteModelCallback(TrainerCallback):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = self.model

        device = next(model.parameters()).device
        model.to("cpu") # move to cpu for saving model

        if hasattr(model, "backbone"):
            backbone = model.backbone.to("cpu")
            backbone.save_pretrained(checkpoint_dir)

        if hasattr(model, "task_emb"):
            torch.save(
                model.task_emb.state_dict(),
                os.path.join(checkpoint_dir, "task_embedding.pt")
            )

        if hasattr(model, "heads"):
            head_dir = os.path.join(checkpoint_dir, "task_heads")
            os.makedirs(head_dir, exist_ok=True)
            for task_name, head in model.heads.items():
                torch.save(
                    head.state_dict(),
                    os.path.join(head_dir, f"{task_name}_head.pt")
                )

        self.tokenizer.save_pretrained(checkpoint_dir)

        print(f"[SaveCompleteModelCallback] Saved complete model to {checkpoint_dir}")
        model.to(device)
