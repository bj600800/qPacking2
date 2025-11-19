"""
------------------------------------------------------------------------------
Author:    Dou Zhixin
Email:     bj600800@gmail.com
Date:      2025/06/05

Description:
    Dataset construction utilities for qPacking / ESM fine-tuning
------------------------------------------------------------------------------
"""

import os
import json
import pickle
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_from_disk

from qpacking2.common.analyze_feature import load_existing_results
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


# =============================================================================
# Structure Feature Dataset (sequence + residue-level label)
# =============================================================================

class DataEncoder(BaseEncoder):

    ENCODERS = {
        "position": "binary",
        "degree": "regression",
        "area": "regression",
        "rsa": "regression",
        "order": "regression"
    }

    def __init__(self, seq_pkl, pkl_file, tokenizer, cache_dir, task):
        super().__init__(tokenizer, cache_dir)
        self.seq_pkl = seq_pkl
        self.pkl_file = pkl_file
        self.task = task

    # Convert raw pkl → {"id", "sequence", "labels"}
    def format_raw(self, pkl_data):
        seqs = load_existing_results(self.seq_pkl)
        formatted = []

        for pid, feature in pkl_data.items():
            if pid not in seqs:
                continue
            seq_info = seqs[pid]
            seq = seq_info['seq']
            seq_dict = seq_info['seq_dict']
            first_id = min(seq_dict.keys())
            L = len(seq)

            # 初始化 labels
            if self.task == "position":
                labels = [0] * L
            else:
                labels = [-100] * L

            # 填充 feature 对应位置
            for res_id, v in feature.items():
                idx = res_id - first_id

                if self.task == "position":
                    labels[idx] = 1 if v > 0 else 0
                else:
                    labels[idx] = v

            formatted.append({
                "id": pid,
                "sequence": seq,
                "labels": labels
            })
        return formatted

    # Tokenize with task-specific encoders
    def encode_item(self, x):
        tok = self.tokenizer(x["sequence"], padding=False, return_attention_mask=True)
        tok["labels"] = [-100] + x["labels"] + [-100]
        return tok

    def tokenize_dataset(self, dataset, cache_dir):
        if os.path.exists(cache_dir):
            logger.info(f"[{self.task}] Loading cached dataset from: {cache_dir}")
            return load_from_disk(cache_dir)

        logger.info(f"[{self.task}] Tokenizing and caching to: {cache_dir}")
        tokenized = dataset.map(
            self.encode_item,
            remove_columns=["id", "sequence", "labels"]
        )
        tokenized.save_to_disk(cache_dir)
        return tokenized

    # Main function
    def get(self, test_ratio, seed):
        """
        Returns:
            tokenized_train, tokenized_test, total_samples
        """

        cache_root = os.path.join(self.cache_dir, self.task)
        cache_train = os.path.join(cache_root, "train")
        cache_test = os.path.join(cache_root, "test")
        mu_sigma_file = os.path.join(cache_root, "mu_sigma.json")

        # --------------------------------------------------------
        # 0️⃣ 若 train/test 缓存都存在 → 直接加载（不重复归一化、不重复 tokenize）
        # --------------------------------------------------------
        if os.path.exists(cache_train) and os.path.exists(cache_test):
            logger.info(f"[{self.task}] Loading cached tokenized dataset...")
            tokenized_train = load_from_disk(cache_train)
            tokenized_test = load_from_disk(cache_test)
            total = len(tokenized_train) + len(tokenized_test)
            return tokenized_train, tokenized_test, total

        # --------------------------------------------------------
        # 1️⃣ 读取原始 pkl
        # --------------------------------------------------------
        raw = load_existing_results(self.pkl_file)
        formatted = self.format_raw(raw)
        dataset = Dataset.from_list(formatted)
        total = len(dataset)

        # --------------------------------------------------------
        # 2️⃣ train/test split —— 这一步必须在计算 z-score 前执行
        # --------------------------------------------------------
        split = dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_dataset = split["train"]
        test_dataset = split["test"]

        # --------------------------------------------------------
        # 3️⃣ 计算 train 的 μ/σ（仅非分类任务）
        # --------------------------------------------------------
        if self.task not in ["position", "rsa"]:
            logger.info(f"[{self.task}] Computing Z-score on TRAIN only...")

            train_valid = [
                v for sample in train_dataset for v in sample["labels"] if v != -100
            ]
            self.mu, self.sigma = self.compute_zscore(train_valid)

            self.dump_mu_sigma(mu_sigma_file)

            def normalize(sample):
                sample["labels"] = [
                    (v - self.mu) / self.sigma if v != -100 else -100
                    for v in sample["labels"]
                ]
                return sample
            logger.info(f"Normalizing train set")
            train_dataset = train_dataset.map(normalize)

            logger.info(f"Normalizing test set")
            test_dataset = test_dataset.map(normalize)

        # --------------------------------------------------------
        # 4️⃣ Tokenization：对 train/test 分别 tokenize 并缓存
        # --------------------------------------------------------
        logger.info(f"[{self.task}] Tokenizing train...")
        tokenized_train = train_dataset.map(
            lambda x: self.encode_item(x),
            remove_columns=["id", "sequence", "labels"]
        )
        tokenized_train.save_to_disk(cache_train)

        logger.info(f"[{self.task}] Tokenizing test...")
        tokenized_test = test_dataset.map(
            lambda x: self.encode_item(x),
            remove_columns=["id", "sequence", "labels"]
        )
        tokenized_test.save_to_disk(cache_test)

        return tokenized_train, tokenized_test, total


# =============================================================================
# Collator with label padding
# =============================================================================

class LabelPaddingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        max_len = max(len(f["labels"]) for f in features)
        for f in features:
            f["labels"] += [-100] * (max_len - len(f["labels"]))
        return super().__call__(features)


# =============================================================================
# Structure Loader
# =============================================================================

def run_structure_encoder(seq_pkl, feature_pkl, model_dir, tokenized_cache_path, task, test_ratio, batch_size, seed):
    set_seed(seed)

    tokenizer = EsmTokenizer.from_pretrained(model_dir, do_lower_case=False)
    collator = LabelPaddingCollator(tokenizer=tokenizer)

    encoder = DataEncoder(seq_pkl, feature_pkl, tokenizer, tokenized_cache_path, task)
    tokenized_train, tokenized_test, total = encoder.get(test_ratio=test_ratio, seed=seed)

    train = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val = DataLoader(tokenized_test, batch_size=batch_size, shuffle=False, collate_fn=collator)

    logger.info(f"[{task}] total={total}, train={len(tokenized_train)}, val={len(tokenized_test)}")
    return train, val, tokenizer


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

    def tokenize(self, dataset):
        if os.path.exists(self.cache_dir):
            try:
                return load_from_disk(self.cache_dir)
            except:
                logger.info("Generating fitness dataset.")
        tokenized = dataset.map(
            lambda x: self.encode_pair(x, self.tokenizer),
            remove_columns=["fitness", "wt_seq", "mt_seq"],
        )
        tokenized.save_to_disk(self.cache_dir)
        return tokenized

    def get(self, test_ratio, seed):
        raw = self.read_pkl(self.pkl_file)
        # 1️⃣ 划分训练/测试
        split_idx = int(len(raw) * (1 - test_ratio))
        random.Random(seed).shuffle(raw)
        train_items, test_items = raw[:split_idx], raw[split_idx:]

        # 2️⃣ 训练集计算 μ/σ → 归一化
        train_items, test_items = self.zscore_split(train_items, test_items)

        dataset_train = Dataset.from_list(train_items)
        dataset_test = Dataset.from_list(test_items)

        # 3️⃣ Tokenize
        tokenized_train = self.tokenize(dataset_train)
        tokenized_test = self.tokenize(dataset_test)
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

if __name__ == '__main__':
    seq_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/test/feature/feature_resid_name.pkl"
    feature_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/test/feature/feature_rsa.pkl"
    model_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/checkpoints/esm2_t30_150M_UR50D"
    cache_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/test/tokenized_cache"
    test_ratio = 0.1
    batch_size = 16
    seed = 3407
    run_structure_encoder(seq_pkl=seq_pkl, feature_pkl=feature_pkl, model_dir=model_dir,
                          tokenized_cache_path=cache_dir, task='rsa', test_ratio=test_ratio, batch_size=batch_size, seed=seed)
    # model_dir = r"/Users/douzhixin/Developer/qPacking-esm/data/test/checkpoints/bsa/best"
    # feature_pkl = r"/Users/douzhixin/Developer/qPacking-esm/data/benchmark/tim-db/ss.pkl"
    # tokenized_cache_path = r"/Users/douzhixin/Developer/qPacking-esm/data/benchmark/tim-db/ss_tokenized_cache"
    # test_ratio = 0.2
    # seed = 3470
    # batch_size = 16
    # run_fitness_data(model_dir=model_dir, feature_pkl=feature_pkl, tokenized_cache_path=tokenized_cache_path,
    #                  test_ratio=test_ratio, seed=seed, batch_size=batch_size)