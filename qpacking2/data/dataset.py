"""
------------------------------------------------------------------------------
Author:    Dou Zhixin
Email:     bj600800@gmail.com
Date:      2025/06/05

Description:
    Dataset construction utilities for qPacking / ESM fine-tuning
------------------------------------------------------------------------------
"""
from tqdm import tqdm
import os
import json
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk

from qpacking2.common.process_pkl import load_pkl
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

# =============================================================================
# Utility
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Base Encoder
# =============================================================================
class BaseEncoder:
    def __init__(self, tokenizer, cache_dir):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.mu = 0.0
        self.sigma = 1.0

    @staticmethod
    def compute_zscore(values):
        arr = np.array(values)
        mu = arr.mean()
        sigma = arr.std() or 1.0
        return float(mu), float(sigma)

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

# =============================================================================
# Multi-task Dataset
# =============================================================================
class MultiTaskDataset(BaseEncoder):
    TASKS = ["position", "rsa", "bsa", "degree", "order"]
    TASK_TYPES = {
        "position": "classification",
        "rsa": "classification",
        "bsa": "regression",
        "degree": "regression",
        "order": "regression"
    }

    def __init__(self, feature_pkl, tokenizer, cache_dir):
        super().__init__(tokenizer, cache_dir)
        self.feature_pkl = feature_pkl

    def format_raw(self, pkl_data):
        """把 list-of-dict 数据整理成统一格式"""
        formatted = []
        for feature_dict in tqdm(pkl_data):
            seq = feature_dict["sequence"]
            task_name = feature_dict["feature_name"]
            labels = feature_dict["label"]

            task_id = [0]*len(self.TASKS)
            task_idx = self.TASKS.index(task_name)
            task_id[task_idx] = 1

            formatted.append({
                "id": feature_dict["protein_name"],
                "sequence": seq,
                "labels": labels,
                "task_name": task_name,
                "task_id": task_id
            })

        return formatted

    def encode_item(self, x):
        tok = self.tokenizer(x["sequence"], padding=False, return_attention_mask=True)
        tok["labels"] = [-100] + x["labels"] + [-100]  # 左右加 pad token
        tok["task_id"] = x["task_id"]
        return tok

    def tokenize_dataset(self, dataset, cache_dir):
        """tokenization 只做一次，存在缓存就直接加载"""
        if os.path.exists(cache_dir):
            logger.info(f"Loading cached dataset from: {cache_dir}")
            return load_from_disk(cache_dir)

        logger.info(f"Tokenizing dataset → {cache_dir}")
        dataset = dataset.map(lambda x: self.encode_item(x),
                              remove_columns=["id","sequence","labels","task_name","task_id"])
        dataset.save_to_disk(cache_dir)
        return dataset

    def get(self, test_ratio, seed):
        tokenized_train_cache = os.path.join(self.cache_dir, "train_tokenized")
        tokenized_test_cache = os.path.join(self.cache_dir, "test_tokenized")

        # 如果已经有缓存，直接加载
        if os.path.exists(tokenized_train_cache) and os.path.exists(tokenized_test_cache):
            logger.info("Loading fully cached tokenized dataset...")
            tokenized_train = load_from_disk(tokenized_train_cache)
            tokenized_test = load_from_disk(tokenized_test_cache)
            total = len(tokenized_train) + len(tokenized_test)
            return tokenized_train, tokenized_test, total

        # 1. 加载原始数据
        raw = load_pkl(self.feature_pkl)
        formatted = self.format_raw(raw)
        dataset = Dataset.from_list(formatted)
        total = len(dataset)

        # 2. 切分训练/测试
        split = dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_dataset = split["train"]
        test_dataset = split["test"]

        # 3. 对回归任务做 μ/σ 标准化
        for task_name in tqdm(self.TASKS, desc='Normalizing regression tasks'):
            if self.TASK_TYPES[task_name] == "regression":
                mu_sigma_path = os.path.join(self.cache_dir, f"{task_name}_mu_sigma.json")
                # 先尝试加载缓存 μ/σ
                if os.path.exists(mu_sigma_path):
                    self.load_mu_sigma(mu_sigma_path)
                else:
                    vals = [v for sample in train_dataset for i, v in enumerate(sample["labels"])
                            if v != -100 and sample["task_name"] == task_name]
                    if len(vals) > 0:
                        self.mu, self.sigma = self.compute_zscore(vals)
                        self.dump_mu_sigma(mu_sigma_path)

                # 应用标准化
                def normalize(sample):
                    if sample["task_name"] == task_name:
                        sample["labels"] = [(v - self.mu) / self.sigma if v != -100 else -100
                                            for v in sample["labels"]]
                    return sample

                train_dataset = train_dataset.map(normalize)
                test_dataset = test_dataset.map(normalize)

        # 4. tokenized dataset
        tokenized_train = self.tokenize_dataset(train_dataset, tokenized_train_cache)
        tokenized_test = self.tokenize_dataset(test_dataset, tokenized_test_cache)

        return tokenized_train, tokenized_test, total


# =============================================================================
# Collator
# =============================================================================
class MultiTaskCollator(DataCollatorWithPadding):
    def __call__(self, features):
        max_len = max(len(f["labels"]) for f in features)
        new_features = []
        for f in features:
            padded_labels = f["labels"] + [-100]*(max_len - len(f["labels"]))
            task_id = f.get("task_id",[0]*5)
            new_features.append({
                **f,
                "labels": padded_labels,
                "task_id": task_id
            })
        return super().__call__(new_features)


# =============================================================================
# Loader
# =============================================================================
def run_multi_task_encoder(feature_pkl, model_dir, tokenized_cache_path,
                           test_ratio, batch_size, seed):
    set_seed(seed)
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    collator = MultiTaskCollator(tokenizer=tokenizer)

    dataset = MultiTaskDataset(feature_pkl=feature_pkl, tokenizer=tokenizer, cache_dir=tokenized_cache_path)
    tokenized_train, tokenized_test, total = dataset.get(test_ratio=test_ratio, seed=seed)

    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=collator)

    logger.info(f"[MultiTask] total={total}, train={len(tokenized_train)}, val={len(tokenized_test)}")
    return train_loader, val_loader, tokenizer

# =============================================================================
# Fitness Dataset (wt–mt pair)
# =============================================================================

class FitnessData(BaseEncoder):
    def __init__(self, pkl_file, tokenizer, cache_dir):
        super().__init__(tokenizer, cache_dir)
        self.pkl_file = pkl_file
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def read_pkl(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # 训练集计算 μ/σ → 应用于训练/测试集
    def zscore_split(self, train_items, test_items):
        vals = [x["fitness"] for x in train_items]
        self.mu, self.sigma = self.compute_zscore(vals)
        stats_path = os.path.join(self.cache_dir, "mu_sigma.json")
        self.dump_mu_sigma(stats_path)

        def normalize(items):
            for x in items:
                x["mutation"] = x["id"].split("_")[1]
                x["raw_fitness"] = x["fitness"]
                x["fitness"] = (x["fitness"] - self.mu) / self.sigma
            return items

        return normalize(train_items), normalize(test_items)

    @staticmethod
    def encode_pair(x, tokenizer):
        wt = tokenizer(x["wt_seq"], padding=False)
        mt = tokenizer(x["mt_seq"], padding=False)
        return {
            "wt_input_ids": wt["input_ids"],
            "wt_attention_mask": wt["attention_mask"],
            "mut_input_ids": mt["input_ids"],
            "mut_attention_mask": mt["attention_mask"],
            "mutation_pos": int(x["id"].split("_")[1][1:-1]),
            "labels": torch.tensor(x["fitness"], dtype=torch.float),
        }

    def tokenize(self, dataset, cache_name="dataset"):
        cache_path = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_path):
            try:
                return load_from_disk(cache_path)
            except:
                logger.info(f"Generating dataset {cache_name}.")

        tokenized = dataset.map(
            lambda x: self.encode_pair(x, self.tokenizer),
            remove_columns=["fitness", "wt_seq", "mt_seq"],
        )
        tokenized.save_to_disk(cache_path)
        return tokenized

    def get(self, test_ratio, seed):
        # 1️⃣ 划分训练/测试
        raw = self.read_pkl(self.pkl_file)
        split_idx = int(len(raw) * (1 - test_ratio))
        random.Random(seed).shuffle(raw)
        train_items, test_items = raw[:split_idx], raw[split_idx:]
        # 2️⃣ 训练集计算 μ/σ → 归一化
        train_items, test_items = self.zscore_split(train_items, test_items)

        dataset_train = Dataset.from_list(train_items)
        dataset_test = Dataset.from_list(test_items)

        # 3️⃣ Tokenize
        tokenized_train = self.tokenize(dataset_train, cache_name="train")
        tokenized_test = self.tokenize(dataset_test, cache_name="test")

        logger.info(f"Fitness samples = train:{len(tokenized_train)}, test:{len(tokenized_test)}")

        return tokenized_train, tokenized_test


class FitnessCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        wt = self.tokenizer.pad(
            [{"input_ids": b["wt_input_ids"], "attention_mask": b["wt_attention_mask"]} for b in batch],
            return_tensors="pt",
        )
        mt = self.tokenizer.pad(
            [{"input_ids": b["mut_input_ids"], "attention_mask": b["mut_attention_mask"]} for b in batch],
            return_tensors="pt",
        )
        return {
            "wt_input_ids": wt["input_ids"],
            "wt_attention_mask": wt["attention_mask"],
            "mut_input_ids": mt["input_ids"],
            "mut_attention_mask": mt["attention_mask"],
            "mutation_pos": torch.tensor([b["mutation_pos"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch]),
        }


def run_fitness_data(model_dir, feature_pkl, tokenized_cache_path, test_ratio, seed, batch_size):
    set_seed(seed)
    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)

    fd = FitnessData(pkl_file=feature_pkl, tokenizer=tokenizer, cache_dir=tokenized_cache_path)
    tokenized_train, tokenized_test = fd.get(test_ratio=test_ratio, seed=seed)
    collator = FitnessCollator(tokenizer)

    train = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train, val, tokenizer

