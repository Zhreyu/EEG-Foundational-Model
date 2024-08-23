#!/usr/bin/env python3
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve,precision_score
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import TrainingArguments, TrainerCallback
from trainer.base import Trainer
from torch.profiler import profile, record_function, ProfilerActivity
import csv
from scipy.special import softmax
import sys
from datetime import datetime
from rich import print
import torch.distributed as dist

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        print('Not using distributed mode')


rank = get_rank()

def create_readme(output_dir, script_name, args):
    os.makedirs(output_dir, exist_ok=True)
    readme_content = f"Script ran: {script_name}\nArguments:\n"
    # Get the absolute path of the current script
    script_path = os.path.abspath(__file__)
    
    # Replace the script name in the path
    base_path = script_path.rsplit('/', 2)[0]
    new_script_path = os.path.join(base_path, script_name)
    
    readme_content += f'Script path: {new_script_path}\n'
    for arg in args:
        readme_content += f"{arg}\n"
    readme_content += f"Date and Time: {datetime.now()}\n"
    readme_path = os.path.join(output_dir, 'readme.txt')
    with open(readme_path, 'w') as readme_file:
        readme_file.write(readme_content)


class ProfCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.profiler = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
            profile_memory=True,
            with_stack=True,
            record_shapes=True
        )
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.stop()
            self.profiler = None  # profiler is released properly

class CSVLogCallback(TrainerCallback):

    def __init__(self):
        super().__init__()
        self.train_log_filepath = None
        self.eval_log_filepath = None

    def on_log(self, args, state, control, model, **kwargs) -> None:
        if args.local_rank not in {-1, 0}:
            return

        if self.train_log_filepath is None:
            self.train_log_filepath = os.path.join(args.output_dir, 'train_history.csv')
            with open(self.train_log_filepath, 'a') as f:
                f.write('step,loss,lr\n')

        if self.eval_log_filepath is None:
            self.eval_log_filepath = os.path.join(args.output_dir, 'eval_history.csv')
            with open(self.eval_log_filepath, 'a') as f:
                f.write('step,loss,accuracy,f1_score,roc_auc\n')

        is_eval = any('eval' in k for k in state.log_history[-1].keys())

        if is_eval:
            with open(self.eval_log_filepath, 'a') as f:
                f.write('{},{},{},{},{}\n'.format(
                    state.global_step,
                    state.log_history[-1]['eval_loss'],
                    state.log_history[-1]['eval_accuracy'] if 'eval_accuracy' in state.log_history[-1] else np.nan,
                    state.log_history[-1]['eval_f1_score'] if 'eval_f1_score' in state.log_history[-1] else np.nan,
                    state.log_history[-1]['eval_roc_auc'] if 'eval_roc_auc' in state.log_history[-1] else np.nan,
                    state.log_history[-1]['eval_precision'] if 'eval_precision' in state.log_history[-1] else np.nan
                ))

        else:
            with open(self.train_log_filepath, 'a') as f:
                f.write('{},{},{}\n'.format(
                    state.global_step,
                    state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else state.log_history[-1]['train_loss'],
                    state.log_history[-1]['learning_rate'] if 'learning_rate' in state.log_history[-1] else None
                ))

def _cat_data_collator(features: List) -> Dict[str, torch.tensor]:

    if not isinstance(features[0], dict):
        features = [vars(f) for f in features] 

    return {
        k: torch.cat(
            [
                f[k]
                for f in features
            ]
        )
        for k in features[0].keys()
        if not k.startswith('__')
    }

def check_label_consistency(labels, num_classes):
    unique_labels = np.unique(labels)
    expected_classes = np.arange(num_classes)
    if not np.array_equal(unique_labels, expected_classes):
        print(f"Adjusting expected classes from {expected_classes} to {unique_labels}")
        return len(unique_labels)  # Adjust the number of classes based on actual labels
    return num_classes  # Return the original number of classes if consistent

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

def decoding_accuracy_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.array(preds)
    labels = np.array(labels).ravel()  # Ensure labels are 1D

    # Apply softmax to predictions
    prob_preds = softmax(preds, axis=1)

    # Determine the actual number of classes from labels
    num_classes = len(np.unique(labels))
    
    # Handle binary case specifically: use column 1 as the score if only two classes
    if num_classes == 2:
        prob_preds = prob_preds[:, 1]  

    # Compute metrics
    pred_labels = preds.argmax(axis=1)
    try:
        auc = roc_auc_score(labels, prob_preds, multi_class='ovr', average='weighted') if num_classes > 2 else roc_auc_score(labels, prob_preds)
    except Exception as e:
        print(f"\nError computing AUC: {e}")
        auc = np.nan

    accuracy = accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels, average='weighted')
    precision = precision_score(labels, pred_labels, average='weighted', zero_division=0)
    if rank == 0:
        print(f"\nAccuracy: {accuracy}")
        print(f"AUC: {auc}")
        print(f"Weighted F1 score: {f1}")
        print(f"Precision: {precision}")
        print('---------------')

    return {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else np.nan,
        "precision": round(precision, 4)
    }


def make_trainer(
    model_init,
    training_style,
    train_dataset,
    validation_dataset,
    do_train: bool = True,
    do_eval: bool = True,
    run_name: str = None,
    output_dir: str = None,
    overwrite_output_dir: bool = True,
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    optim: str='adamw_hf',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.1,
    adam_beta1: float=0.9,
    adam_beta2: float=0.999,
    adam_epsilon: float=1e-8,
    max_grad_norm: float=1.0,
    per_device_train_batch_size: int = 8,  # Reduced batch size
    per_device_eval_batch_size: int = 8,   # Reduced batch size
    dataloader_num_workers: int = 4,
    max_steps: int = 400000,
    num_train_epochs: int = 1,
    lr_scheduler_type: str = 'linear',
    warmup_ratio: float = 0.01,
    evaluation_strategy: str = 'steps',
    logging_strategy: str = 'steps',
    save_strategy: str = 'steps',
    save_total_limit: int = 5,
    save_steps: int = 10000,
    logging_steps: int = 10000,
    eval_steps: int = None,
    logging_first_step: bool = True,
    greater_is_better: bool = True,
    seed: int = 1,
    fp16: bool = True,
    deepspeed: str = None,
    compute_metrics = None,
    **kwargs
    ) -> Trainer:
    """
    Make a Trainer object for training a model.
    Returns an instance of transformers.Trainer.
    
    See the HuggingFace transformers documentation for more details
    on input arguments:
    https://huggingface.co/transformers/main_classes/trainer.html

    Custom arguments:
    ---
    model_init: callable
        A callable that does not require any arguments and 
        returns model that is to be trained (see scripts.train.model_init)
    training_style: str
        The training style (ie., framework) to use.
        One of: 'BERT', 'CSM', 'NetBERT', 'autoencoder',
        'decoding'.
    train_dataset: src.batcher.dataset
        The training dataset, as generated by src.batcher.dataset
    validation_dataset: src.batcher.dataset
        The validation dataset, as generated by src.batcher.dataset

    Returns
    ----
    trainer: transformers.Trainer
    """
    trainer_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        do_train=do_train,
        do_eval=do_eval,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_num_workers=dataloader_num_workers,
        optim=optim,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        greater_is_better=greater_is_better,
        save_steps=save_steps,
        logging_strategy=logging_strategy,
        logging_first_step=logging_first_step,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if eval_steps is not None else logging_steps,
        seed=seed,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        eval_accumulation_steps=1,
        deepspeed=deepspeed,
        report_to="none",  # Disable WANDB
        **kwargs
    )

    data_collator = _cat_data_collator
    is_deepspeed = deepspeed is not None
    create_readme(output_dir, sys.argv[0], sys.argv[1:])
    if training_style == 'decoding':
        compute_metrics = decoding_accuracy_metrics
    else:
        compute_metrics = None
    
    trainer = Trainer(
        args=trainer_args,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
        is_deepspeed=is_deepspeed
    )

    trainer.add_callback(CSVLogCallback)

    return trainer
