"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/20

# Description:
#   Multi-task metric computation for: position, rsa, bsa, degree, order
# ------------------------------------------------------------------------------
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import spearmanr


TASKS = ["position", "rsa", "bsa", "degree", "order"]
TASK_TYPES = {
    "position": "classification",
    "rsa": "classification",
    "bsa": "regression",
    "degree": "regression",
    "order": "regression",
}


def safe_to_numpy(x):
    """Convert tensor → numpy safely"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_metrics_hf(eval_pred):
    """
    eval_pred = (logits_dict, labels_dict)
    logits_dict: dict(task → tensor)
    labels_dict: dict(task → tensor)
    """

    ret, _  = eval_pred
    logits_dict, labels_dict, task_loss_dict = ret
    metrics = {}

    cls_f1_all = []
    reg_spearman = []

    for task in logits_dict.keys():
        preds_task = safe_to_numpy(logits_dict[task])
        labels_task = safe_to_numpy(labels_dict[task])

        # Build mask for valid labels
        mask = labels_task != -100
        if np.sum(mask) == 0:
            continue

        preds_masked = preds_task[mask]
        labels_masked = labels_task[mask]

        if TASK_TYPES[task] == "classification":
            # [N, L, C] or [N, C] → argmax
            if preds_masked.ndim == 3:
                preds_masked = np.argmax(preds_masked.reshape(-1, preds_masked.shape[-1]), axis=-1)
                labels_masked = labels_masked.reshape(-1)
            elif preds_masked.ndim == 2:
                preds_masked = np.argmax(preds_masked, axis=-1)

            f1 = f1_score(labels_masked, preds_masked, average="macro")
            acc = accuracy_score(labels_masked, preds_masked)
            metrics[f"eval_{task}_f1"] = float(f1)
            metrics[f"eval_{task}_acc"] = float(acc)
            cls_f1_all.append(f1)

        else:  # regression
            preds_masked = preds_masked.astype(float)
            labels_masked = labels_masked.astype(float)
            s = spearmanr(labels_masked.flatten(), preds_masked.flatten()).correlation
            if np.isnan(s):
                s = 0.0
            metrics[f"eval_{task}_spearman"] = float(s)
            reg_spearman.append(s)

    # Macro metrics
    metrics["eval_cls_f1_macro"] = float(np.mean(cls_f1_all)) if cls_f1_all else 0.0
    metrics["eval_reg_spearman"] = float(np.mean(reg_spearman)) if reg_spearman else 0.0
    metrics["eval_combined_score"] = metrics["eval_cls_f1_macro"] + metrics["eval_reg_spearman"]

    return metrics

