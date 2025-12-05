"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/20

# Description: Create multitask finetuning dataset for qpacking2 with mlm, degree, order, position, bsa, rsa
# ------------------------------------------------------------------------------
"""
import os
import random
import pickle
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk

from qpacking2.common.analyze_feature import load_existing_results
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BaseEncoder:
    """Base class shared by structure encoders and fitness encoder."""

    def __init__(self, tokenizer, cache_dir):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.mu = 0.0
        self.sigma = 1.0

    # Z-score
    @staticmethod
    def compute_zscore(values):
        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std() or 1.0
        return float(mu), float(sigma)

    # JSON save/load
    def dump_mu_sigma(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mu": self.mu, "sigma": self.sigma}, f, indent=4)
        logger.info(f"Saved mu={self.mu} sigma={self.sigma} → {path}")

    def load_mu_sigma(self, path):
        with open(path, "r") as f:
            d = json.load(f)
        self.mu = d["mu"]
        self.sigma = d["sigma"]
        logger.info(f"Loaded mu={self.mu} sigma={self.sigma} ← {path}")

class MultiTaskDataEncoder:
    """
    支持：
    - MLM 主任务
    - 多个结构任务（degree, bsa, rsa, order, position）为辅助
    - 动态 padding
    """
    TASK_TYPES = {
        "degree": "regression",
        "bsa": "regression",
        "rsa": "regression",
        "order": "regression",
        "position": "binary"
    }

    def __init__(self, feature_pkl, tokenizer, cache_dir):
        """
        feature_pkl: pickle 文件，格式：
            { protein_name: {
                  "seq": "MKT...",
                  "seq_dict": {resid: idx, ...},
                  "degree": {resid: val, ...},
                  "bsa": {resid: val, ...},
                  "rsa": {resid: val, ...},
                  "order": {resid: val, ...},
                  "position": {resid: val, ...}
              }
            }
        """
        self.feature_pkl = feature_pkl
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.mu_sigma = {}  # z-score info

    # --------------------------------
    # 加载 pickle
    # --------------------------------
    def _load_pkl(self):
        with open(self.feature_pkl, "rb") as f:
            return pickle.load(f)

    # --------------------------------
    # 格式化单条数据
    # --------------------------------
    def _format_item(self, pid, data, task_list):
        resid_name = data["resid_name"]
        seq = "".join(resid_name[k] for k in sorted(resid_name.keys()))
        first_id = min(resid_name.keys())
        L = len(seq)

        formatted = {"id": pid, "sequence": seq}

        for task in task_list:
            task_type = self.TASK_TYPES[task]
            labels = [0]*L if task_type=="binary" else [-100]*L
            for res_id, v in data.get(task, {}).items():
                idx = res_id - first_id
                if idx < 0 or idx >= L:
                    continue
                if task_type=="binary":
                    labels[idx] = 1 if v>0 else 0
                else:
                    labels[idx] = v
            formatted[task] = labels
        return formatted

    # --------------------------------
    # tokenization
    # --------------------------------
    def _encode_item(self, x, task_list):
        # 假设 taskid 是字母开头的特殊 token，比如 "<degree>"
        task_token = f"<{task_list[0]}>"  # 举例: batch 每次只有一个主任务
        enc = self.tokenizer(task_token + x["sequence"], truncation=False, padding=False, return_attention_mask=True)
        for task in task_list:
            enc[task] = x[task]
        return enc

    # --------------------------------
    # z-score
    # --------------------------------
    def _compute_zscore(self, values):
        arr = torch.tensor(values, dtype=torch.float32)
        mu, sigma = float(arr.mean()), float(arr.std())
        return mu, sigma

    # --------------------------------
    # 主函数
    # --------------------------------
    def get_dataloaders(self, test_ratio=0.1, batch_size=16, seed=42):
        os.makedirs(self.cache_dir, exist_ok=True)
        task_list = list(self.TASK_TYPES.keys())
        cache_train = os.path.join(self.cache_dir, "train")
        cache_val = os.path.join(self.cache_dir, "val")
        cache_mu_sigma = os.path.join(self.cache_dir, "mu_sigma.json")

        # 1️⃣ 如果缓存存在，直接加载
        if os.path.exists(cache_train) and os.path.exists(cache_val):
            logger.info("Loading cached tokenized dataset...")
            tokenized_train = load_from_disk(cache_train)
            tokenized_val = load_from_disk(cache_val)
            collator = LabelPaddingCollator(self.tokenizer)
            return (DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator),
                    DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collator),
                    self.tokenizer)

        # 2️⃣ 加载原始 pickle 并格式化
        raw = self._load_pkl()
        dataset_list = [self._format_item(pid, data, task_list) for pid, data in raw.items()]
        dataset = Dataset.from_list(dataset_list)

        # 3️⃣ train/test split
        split = dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_dataset = split["train"]
        val_dataset = split["test"]

        # 4️⃣ z-score 仅对回归任务
        for task in task_list:
            if self.TASK_TYPES[task]=="regression":
                values = [v for sample in train_dataset for v in sample[task] if v!=-100]
                mu, sigma = self._compute_zscore(values)
                self.mu_sigma[task] = {"mu": mu, "sigma": sigma}

                def normalize(sample, t=task, mu=mu, sigma=sigma):
                    sample[t] = [(v-mu)/sigma if v!=-100 else -100 for v in sample[t]]
                    return sample
                train_dataset = train_dataset.map(normalize)
                val_dataset = val_dataset.map(normalize)

        # 保存 mu/sigma
        with open(cache_mu_sigma, "w") as f:
            json.dump(self.mu_sigma, f)

        # 5️⃣ tokenization
        train_tokenized = train_dataset.map(lambda x: self._encode_item(x, task_list), remove_columns=["id","sequence"])
        val_tokenized = val_dataset.map(lambda x: self._encode_item(x, task_list), remove_columns=["id","sequence"])

        train_tokenized.save_to_disk(cache_train)
        val_tokenized.save_to_disk(cache_val)

        # 6️⃣ DataLoader + dynamic padding
        collator = LabelPaddingCollator(self.tokenizer)
        train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

        return train_loader, val_loader, self.tokenizer


class LabelPaddingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 找到 batch 内 labels 最大长度
        task_list = [k for k in features[0].keys() if k not in ["input_ids", "attention_mask", "token_type_ids"]]
        max_len = max(len(f["input_ids"]) for f in features)

        for f in features:
            for task in task_list:
                pad_value = -100 if task != "position" else 0
                f[task] = f[task] + [pad_value] * (max_len - len(f[task]))
        return super().__call__(features)

if __name__ == '__main__':
    feature_pkl = "/Users/douzhixin/Developer/qPacking2/data/test/feature/test_feature.pkl"
    model_dir = "/Users/douzhixin/Developer/qPacking2/data/checkpoints/esm2_t30_150M_UR50D"
    cache_dir = "/Users/douzhixin/Developer/qPacking2/data/feature/cache"

    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    special_tokens = ["<degree>", "<bsa>", "<rsa>", "<order>", "<position>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    encoder = MultiTaskDataEncoder(feature_pkl, tokenizer, cache_dir)
    train_loader, val_loader, tokenizer = encoder.get_dataloaders(test_ratio=0.1, batch_size=16, seed=3407)
    # 打印 train_loader 前 2 个 batch 的内容
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        # 打印 input_ids 的形状和前几个 token
        print("input_ids:", batch["input_ids"].shape)
        print("input_ids sample:", batch["input_ids"][0][:20])

        # 打印 attention_mask
        print("attention_mask:", batch["attention_mask"][0][:20])

        # 打印每个任务的 label 前 20 个
        for task in encoder.TASK_TYPES.keys():
            print(f"{task} labels:", batch[task][0][:20])

        if i >= 1:  # 只打印前 2 个 batch
            break
