"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/12/8

# Description: Model headers for classification, regression
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_class):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_class)
        self.loss_fn = CrossEntropyLoss(ignore_index=-100)

    def forward(self, hidden_states, labels=None):
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits)

class RegressionHead(nn.Module):
    def __init__(self, hidden_size, weighted=False):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, 1)
        self.weighted = weighted
        self.loss_fn = MSELoss(reduction="mean")

    def forward(self, hidden_states, labels=None):
        logits = self.regressor(hidden_states).squeeze(-1)
        loss = None
        if labels is not None:
            mask = labels != -100
            active_logits = logits[mask]
            active_labels = labels[mask].float()
            if self.weighted:
                mean_label = active_labels.mean() + 1e-6
                weights = active_labels / mean_label
                loss = torch.mean(weights * (active_logits - active_labels) ** 2)  # 对大值更敏感
            else:
                loss = self.loss_fn(active_logits, active_labels)
        return TokenClassifierOutput(loss=loss, logits=logits)