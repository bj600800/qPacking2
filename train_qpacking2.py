"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/11/19

# Description: Run qpacking2
# ------------------------------------------------------------------------------
"""
import os
import argparse
from datetime import datetime
import mlflow
from qpacking2.data import dataset
from qpacking2.model.setup_train import train_multitask_model
from train_configs import Config
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def multitask(config):
    task = config.training_args.task

    model_args = {
        "feature_pkl": config.path.feature_pkl,
        "tokenized_cache_path": config.path.tokenized_cache_path,
        "test_ratio": config.training_args.test_ratio,
        "model_dir": config.path.model_dir,
        "checkpoints_dir": os.path.join(config.path.checkpoints_dir, task),
        "logging_dir": config.path.logging_dir,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "seed": config.training_args.seed,
        "lr": config.training_args.lr,
        "add_lora_layers": config.lora.add_lora_layers,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout,
        "task_weights": config.training_args.task_weights,
        "eval_steps": config.training_args.eval_steps,
        "eval_strategy": config.training_args.eval_strategy,
        "save_total_limit": config.training_args.save_total_limit,
        "save_steps": config.training_args.save_steps,
        "save_strategy": config.training_args.save_strategy,
        "logging_strategy": config.training_args.logging_strategy,
        "logging_steps": config.training_args.logging_steps,
        "reporter": config.training_args.reporter,
        "metric_for_best_model": config.training_args.metric_for_best_model,
        "greater_is_better": config.training_args.greater_is_better
    }

    try:
        train_multitask_model(**model_args)
    except TypeError as e:
        logger.error("Failed to start model — argument mismatch!")
        logger.error(str(e))
        raise

def create_mlflow_experiment(config, task):
    """
    Create an MLflow experiment for position / degree / bsa / rsa / order.
    """
    pkl_name = os.path.basename(config.path.feature_pkl).split('.')[0]
    base_model_name = os.path.basename(config.path.model_dir)

    experiment_name = f"{task}_finetune"
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(experiment_name)
        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
        raise

    timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
    run_name = (
        f"{timestamp}_{task}_{base_model_name}_"
        f"{pkl_name}_"
        f"lora_layers:{config.lora.add_lora_layers}_"
        f"bs:{config.training_args.batch_size}_"
        f"task_weights:{config.training_args.task_weights}"
    )

    return {
        "run_name": run_name,
        "task": task,
        "model": base_model_name,
        "pkl_name": pkl_name,
        "lr": config.training_args.lr,
        "batch_size": config.training_args.batch_size,
        "num_epochs": config.training_args.num_epochs,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "lora_dropout": config.lora.dropout
    }

def run_multitask_with_mlflow(config):
    info = create_mlflow_experiment(config, config.training_args.task)

    with mlflow.start_run(run_name=info["run_name"]):
        mlflow.set_tags(info)

        multitask(config)

def main():
    parser = argparse.ArgumentParser(description="Protein model script")
    parser.add_argument(
        '--yaml',
        type=str,
        required=True,
        help="Hyper-params file needed for specifying the model task"
    )

    args = parser.parse_args()
    yaml_path = args.yaml

    config, task = Config.from_yaml(yaml_path)
    log = Config.ConfigLogger(config, task)
    log.log()
    if task == 'multitask':
        run_multitask_with_mlflow(config)

    # elif task == 'fitness':
    #     run_fitness_with_mlflow(config)

    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()