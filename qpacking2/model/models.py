"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/20

# Description: 
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from qpacking2.model.base import BaseESMLoraModel
from qpacking2.model.heads import ClassificationHead, RegressionHead

class MultiTaskOutput(TokenClassifierOutput):
    def __init__(self, loss=None, logits=None, task_loss=None):
        super().__init__(loss=loss, logits=logits)
        self.task_loss = task_loss

class MultiTaskModel(BaseESMLoraModel):
    TASKS = ["position", "rsa", "bsa", "degree", "order"]
    TASK_TYPES = {
        "position": "classification",
        "rsa": "classification",
        "bsa": "regression",
        "degree": "regression",
        "order": "regression"
    }

    def __init__(self, model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout,
                 regression_weighted=False):
        super().__init__(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout)
        self.num_classes_dict = {"position": 2, "rsa": 2}

        self.heads = nn.ModuleDict()
        for task in self.TASKS:
            if self.TASK_TYPES[task] == "classification":
                self.heads[task] = ClassificationHead(self.hidden_size, self.num_classes_dict[task])
            else:
                self.heads[task] = RegressionHead(self.hidden_size, weighted=regression_weighted)

        self.task_emb = nn.Embedding(len(self.TASKS), self.hidden_size)

    def forward(self, input_ids, attention_mask=None, labels=None, task_id=None, task_weights=None):
        hidden = self.encode(input_ids, attention_mask)  # [B, L, H]

        if task_id is not None:
            task_index = task_id.argmax(dim=-1)                 # [B]
            task_embedding = self.task_emb(task_index)          # [B, H]
            hidden[:, 0] = hidden[:, 0] + 0.1 * task_embedding

        if labels is None or task_id is None:
            return hidden

        if task_weights is None:
            task_weights = {task: 1.0 for task in self.TASKS}

        total_loss = 0.0
        logits_dict = {}
        labels_dict = {}
        task_loss_dict = {}
        total_weight = 0.0
        B, L, H = hidden.size()  # batch size, seq len, hidden dim

        for t_idx, task_name in enumerate(self.TASKS):
            mask = task_id[:, t_idx].bool()

            if mask.sum() > 0:
                task_hidden = hidden[mask]
                task_labels = labels[mask]

                if self.TASK_TYPES[task_name] == "classification":
                    task_labels = task_labels.long()

                head = self.heads[task_name]
                out = head(task_hidden, task_labels)

                weight = task_weights.get(task_name, 1.0)
                total_loss += weight * out.loss
                total_weight += weight

                logits_dict[task_name] = out.logits
                labels_dict[task_name] = task_labels
                task_loss_dict[task_name] = out.loss

            else:
                # 当前 batch 没有该任务样本，返回空 tensor
                if self.TASK_TYPES[task_name] == "classification":
                    C = self.num_classes_dict[task_name]
                    logits_dict[task_name] = torch.empty((0, L, C), device=hidden.device)
                    labels_dict[task_name] = torch.empty((0, L), device=hidden.device)
                else:  # regression
                    logits_dict[task_name] = torch.empty((0, L), device=hidden.device)
                    labels_dict[task_name] = torch.empty((0, L), device=hidden.device)

                task_loss_dict[task_name] = torch.tensor(0.0, device=hidden.device)

        avg_loss = total_loss / max(total_weight, 1e-6)
        return avg_loss, (logits_dict, labels_dict, task_loss_dict)

class DynamicMultiTaskModel(BaseESMLoraModel):
    TASKS = ["position", "rsa", "bsa", "degree", "order"]
    TASK_TYPES = {
        "position": "classification",
        "rsa": "classification",
        "bsa": "regression",
        "degree": "regression",
        "order": "regression"
    }

    def __init__(self, model_dir, num_features, add_lora_layers, lora_rank=4,
                 lora_alpha=4, lora_dropout=0.1, num_classes_dict=None, regression_weighted=False):
        super().__init__(model_dir, add_lora_layers, lora_rank, lora_alpha, lora_dropout)

        self.num_features = num_features
        self.feature_attention = nn.Parameter(torch.ones(num_features))  # learnable feature weights

        # 多任务 head
        self.heads = nn.ModuleDict()
        for task in self.TASKS:
            if self.TASK_TYPES[task] == "classification":
                self.heads[task] = ClassificationHead(self.hidden_size, num_classes_dict[task])
            else:
                self.heads[task] = RegressionHead(self.hidden_size, weighted=regression_weighted)

    def forward(self, input_ids, attention_mask=None, feature_tensor=None, labels=None, task_id=None):
        """
        feature_tensor: [B, L, num_features], 动态特征加权
        task_id: [B, num_tasks] one-hot
        """
        # -------------------------------
        # 特征加权
        # -------------------------------
        if feature_tensor is not None:
            attn_weights = torch.softmax(self.feature_attention, dim=0)  # [num_features]
            weighted_features = feature_tensor * attn_weights  # 广播
        else:
            weighted_features = None

        # -------------------------------
        # encoder 输出
        # -------------------------------
        hidden = self.encode(input_ids, attention_mask, weighted_features)

        if labels is None or task_id is None:
            return hidden

        total_loss = 0.0
        logits_list = []
        task_loss_dict = {}
        task_grads = []

        for t_idx, task_name in enumerate(self.TASKS):
            mask = task_id[:, t_idx].bool()
            if mask.sum() == 0:
                continue

            task_hidden = hidden[mask]
            task_labels = labels[mask]

            if self.TASK_TYPES[task_name] == "classification":
                task_labels = task_labels.long()

            head = self.heads[task_name]
            out = head(task_hidden, task_labels)
            task_loss_dict[task_name] = out.loss

            logits_list.append(out.logits)

            # 记录梯度用于动态 loss
            if out.loss is not None:
                grads = torch.autograd.grad(out.loss, self.encoder.parameters(), retain_graph=True, create_graph=True)
                grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
                task_grads.append(grad_norm)
            else:
                task_grads.append(torch.tensor(0.0, device=hidden.device))

        # -------------------------------
        # 动态 loss 权重
        # -------------------------------
        task_grads = torch.stack(task_grads)
        task_weights = task_grads / (task_grads.sum() + 1e-6)  # 梯度归一化
        avg_loss = sum(w * task_loss_dict[t] for w, t in zip(task_weights, task_loss_dict.keys()))

        return MultiTaskOutput(loss=avg_loss, logits=logits_list, task_loss=task_loss_dict)


