"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# DATE:      2025/11/20

# Description:
#   Multi-task training setup for hydrophobic packing tasks
# ------------------------------------------------------------------------------
"""
import os
import torch
from qpacking2.model.models import MultiTaskModel
from qpacking2.model.metrics import compute_metrics_hf
from qpacking2.model.save import SaveCompleteModelCallback
from qpacking2.data import dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from qpacking2.common import logger

logger = logger.setup_log(name=__name__)

def train_multitask_model(feature_pkl, model_dir, tokenized_cache_path, test_ratio,
                          add_lora_layers, lora_rank, lora_alpha, lora_dropout, logging_strategy,
                          batch_size, seed, checkpoints_dir, lr, num_epochs,logging_steps,
                          eval_strategy, logging_dir, eval_steps, save_strategy, save_steps,
                          save_total_limit, metric_for_best_model, greater_is_better, reporter
                          ):

    model = MultiTaskModel(model_dir=model_dir,
                           add_lora_layers=add_lora_layers,
                           lora_rank=lora_rank,
                           lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout)

    dataset_args = {
        "feature_pkl": feature_pkl,
        "model_dir": model_dir,
        "tokenized_cache_path": tokenized_cache_path,
        "test_ratio": test_ratio,
        "batch_size": batch_size,
        "seed": seed
    }

    train_dataloader, valid_dataloader, tokenizer = dataset.run_multi_task_encoder(**dataset_args)
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,
        ddp_find_unused_parameters=False,
        num_train_epochs=num_epochs,
        logging_strategy=logging_strategy,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
        report_to=reporter
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=valid_dataloader.dataset,
        data_collator=train_dataloader.collate_fn,
        compute_metrics=compute_metrics_hf,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                   SaveCompleteModelCallback(model=model, tokenizer=tokenizer)],
    )

    trainer.train()

    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    # 1️⃣ 保存 backbone
    model.backbone.save_pretrained(best_model_path)

    # 2️⃣ 保存 tokenizer
    tokenizer.save_pretrained(best_model_path)

    # 3️⃣ 保存每个 task head
    head_dir = os.path.join(best_model_path, "task_heads")
    os.makedirs(head_dir, exist_ok=True)
    for task_name, head in model.heads.items():
        head_path = os.path.join(head_dir, f"{task_name}_head.pt")
        torch.save(head.state_dict(), head_path)

    # 4️⃣ 保存 task embedding
    task_emb_path = os.path.join(best_model_path, "task_emb.pt")
    torch.save(model.task_emb.state_dict(), task_emb_path)

    logger.info(f"The best trained MultiTaskModel (backbone + heads + task_emb) saved to: {best_model_path}")


def train_fitness_regression_head(model_dir, model_src, unfreeze_last_n, emb_src,
                                  checkpoints_dir, lr, eval_strategy, save_strategy, logging_strategy,
                                  save_total_limit, eval_steps, save_steps, batch_size, num_epochs,
                                  logging_dir, logging_steps, seed, reporter, metric_for_best_model, greater_is_better,
                                  train_dataloader, valid_dataloader, tokenizer):

    model = FitnessRegressionModel(model_dir, model_src, unfreeze_last_n, emb_src, params)

    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        learning_rate=lr,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_strategy=logging_strategy,
        eval_steps=eval_steps,
        save_steps=save_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
        seed=seed,
        report_to=reporter,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        fp16=torch.cuda.is_available(),
    )

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader.dataset,
            eval_dataset=valid_dataloader.dataset,
            compute_metrics=compute_regression_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                       SaveCompleteModelCallback(model=model, tokenizer=tokenizer)]
        )
        trainer.train()
    except TypeError as e:
        pass

    best_model_path = os.path.join(checkpoints_dir, 'best')
    os.makedirs(best_model_path, exist_ok=True)

    model.model.save_pretrained(best_model_path)

    torch.save(model.regressor.state_dict(), f"{best_model_path}/regression_head.pt")

    tokenizer.save_pretrained(best_model_path)

    logger.info(f"The best trained regressor head [Fitness] saved to: {best_model_path}")

