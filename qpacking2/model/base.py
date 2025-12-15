"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/20

# Description: Base model with LoRA adaptation
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import EsmModel

def load_lora_model(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout):
    model = EsmModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        add_pooling_layer=False
    )
    model.enable_input_require_grads()  # Enable gradients for input embeddings

    if add_lora_layers <= 0:
        for p in model.parameters():
            p.requires_grad = False
        print("LoRA not added. Model returned with all parameters frozen.")
        return model

    num_layers = len(model.encoder.layer)
    target_modules = []
    for i in range(max(0, num_layers - add_lora_layers), num_layers):
        layer_prefix = f"encoder.layer.{i}."
        target_modules.extend([
            f"{layer_prefix}attention.self.query",
            f"{layer_prefix}attention.self.key",
            f"{layer_prefix}attention.self.value",
            f"{layer_prefix}attention.output.dense",
            f"{layer_prefix}output.dense"
        ])

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        inference_mode=False,
        bias="none"
    )

    model = get_peft_model(model, config)
    return model


class BaseESMLoraModel(nn.Module):
    def __init__(self, model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout, dropout_prob=0.1):
        super().__init__()
        self.backbone = load_lora_model(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout)
        self.hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)

        if add_lora_layers > 0:
            print("LoRA Layers:")
            lora_count = 0
            for name, module in self.backbone.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    print(f"   {name}")
                    lora_count += 1
            print(f"\nTotal LoRA layers found: {lora_count}")

            print("\nTrainable Parameters (LoRA only):")
            self.backbone.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in self.backbone.parameters())
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print("\nTrainable Parameters (No LoRA):")
            print(f"   Total parameters: {total}")
            print(f"   Trainable parameters: {trainable}")

    def encode(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)
