"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/2/14

# Description: Manage configuration datatype and value
# ------------------------------------------------------------------------------
"""
import yaml
from dataclasses import dataclass
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)


@dataclass
class PathConfig:
    model_dir: str
    checkpoints_dir: str
    logging_dir: str
    tokenized_cache_path: str
    feature_pkl: str


@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    add_lora_layers: int

@dataclass
class TrainingArgsConfig:
    task: str
    seed: int
    lr: float
    num_epochs: int
    test_ratio: float
    batch_size: int
    task_weights: dict
    eval_strategy: str
    save_strategy: str
    save_steps: str
    save_total_limit: int
    logging_strategy: str
    eval_steps: int
    logging_steps: int
    reporter: str
    metric_for_best_model: str
    greater_is_better: bool

@dataclass
class PathConfigFitness:
    model_dir: str
    checkpoints_dir: str
    logging_dir: str
    tokenized_cache_path: str
    seq_fasta: str
    feature_pkl: str
    model_src: str

@dataclass
class TrainingArgsFitnessConfig(TrainingArgsConfig):
    unfreeze_last_n: int
    emb_src: str


@dataclass
class ConfigMultitask:
    path: PathConfig
    lora: LoRAConfig
    training_args: TrainingArgsConfig


@dataclass
class ConfigFitness:
    path: PathConfigFitness
    training_args: TrainingArgsFitnessConfig

def from_yaml(path: str):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)

    task = raw.get('training_args', {}).get('task', None)

    if task == 'multitask':
        return ConfigMultitask(
            path=PathConfig(**raw['path']),
            lora=LoRAConfig(**raw['lora']),
            training_args=TrainingArgsConfig(**raw['training_args'])
        ), task

    elif task == "fitness":
        return ConfigFitness(
            path=PathConfigFitness(**raw['path']),
            training_args=TrainingArgsFitnessConfig(**raw['training_args'])
        ), task

    else:
        raise ValueError(f"Unsupported task type: {task}")

class ConfigLogger:
    """
    A class to log configuration details for different model tasks.
    """
    def __init__(self, config, task: str, logger=logger):
        self.config = config
        self.task = task.lower()
        self.logger = logger

    def log(self):
        self.logger.info(f"\n{'='*10} [Task: {self.task}] Config Summary {'='*10}")
        self._log_common()

        if self.task == "multitask":
            self._log_multitask()
        elif self.task == "fitness":
            self._log_fitness()
        else:
            self.logger.warning(f"Unknown task: {self.task}. Logging only common parameters.")
        self.logger.info(f"\n{'=' * 10} End of Config Summary {'=' * 10}")

    def _log_common(self):
        cfg = self.config
        self.logger.info(f"{'='*10}[Path]{'='*10}")
        for k, v in vars(cfg.path).items():
            self.logger.info(f"{k}: {v}")

        self.logger.info(f"{'='*10}[Training Args]{'='*10}")
        for k, v in vars(cfg.training_args).items():
            self.logger.info(f"{k}: {v}")

        if self.task != "fitness":
            self.logger.info(f"{'='*10}[LoRA]{'='*10}")
            for k, v in vars(cfg.lora).items():
                self.logger.info(f"{k}: {v}")

    def _log_multitask(self):
        pass

    def _log_fitness(self):
        self.logger.info("[Fitness Task Specific Config]")
        self.logger.info(f"model_src: {self.config.path.model_src}")
        self.logger.info(f"unfreeze_last_n: {self.config.training_args.unfreeze_last_n}")
        self.logger.info(f"emb_src: {self.config.training_args.emb_src}")