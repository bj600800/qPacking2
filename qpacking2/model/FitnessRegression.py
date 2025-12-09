"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: Model for fitness regression tasks
# ------------------------------------------------------------------------------
"""
from typing import Optional
from dataclasses import dataclass
from transformers import EsmModel
from peft import PeftConfig, PeftModel
import torch.nn as nn
import torch
from transformers.modeling_outputs import ModelOutput

@dataclass
class RegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction: torch.FloatTensor = None

class FitnessRegressionModel(nn.Module):
    def __init__(self, model_dir, model_src, unfreeze_last_n, emb_src, params):
        super().__init__()
        if model_src == 'official':
            encoder = EsmModel.from_pretrained(model_dir, add_pooling_layer=False)
            model_prefix = "encoder.layer"
        elif model_src == 'finetuned':
            peft_config = PeftConfig.from_pretrained(model_dir)
            base = EsmModel.from_pretrained(peft_config.base_model_name_or_path, add_pooling_layer=False)
            encoder = PeftModel.from_pretrained(base, model_dir)
            model_prefix = "base_model.model.encoder.layer"
        else:
            raise ValueError("model_src must be 'official' or 'finetuned'")
        self.model = encoder
        self.emb_src = emb_src
        hidden = self.model.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )

        self.loss_fn = nn.MSELoss()

        params.unfreeze_backbone(self.model, unfreeze_last_n, model_prefix)

        # Train header
        for name, param in self.regressor.named_parameters():
            param.requires_grad = True


    def forward(self, wt_input_ids, wt_attention_mask, mut_input_ids, mut_attention_mask, mutation_pos, labels=None):
        def extract(x_ids, x_mask):
            out = self.model(x_ids, x_mask)[0]
            if self.emb_src == 'cls':
                return out[:,0]
            elif self.emb_src == 'pos':
                B, L, H = out.size()
                pos = mutation_pos.view(-1,1,1).expand(-1,1,H)
                return torch.gather(out, 1, pos).squeeze(1)
            else:
                raise ValueError("emb_src must be 'cls' or 'pos'")
        diff = extract(mut_input_ids, mut_attention_mask) - extract(wt_input_ids, wt_attention_mask)
        pred = self.regressor(diff).squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels)
        return RegressionOutput(loss=loss, prediction=pred)
