#!/usr/bin/env python3

"""
train.py

Training of models on given data. See get_args() for 
details on command line arguments.

To train a model, multiple core components from ..src/ 
are invoked:

src/batcher: Building PyTorch dataloaders for given data.
src/embedder: Embedding of inputs into embedding space, 
    training-style specific addition of training tokens
    and masking, and computation of training-style specific 
    losses.
    Valid training styles:
        - CSM (Causal Sequence Modeling)
        - decoding
src/decoder: Model architecture used for decoding / sequence modeling. 
    One of the following:
        - GPT
        - PretrainedBERT (as provided by HuggingFace)
src/unembedder: Projecting sequence output of decoder back 
    to input space.
src/trainer: Trainer for model; invokes instance of 
    Hugging Face's Trainer object.
src/model: Build full model from components (ie., embedder, 
    decoder, unembedder). See make_model() below for details.
"""

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score
import os
from rich import print
from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import TrainingArguments, TrainerCallback 
from trainer.base import Trainer
import pandas as pd
import csv
from batcher.downstream_dataset import EEGDatasetCls,EEG2dstDataset
import torch
import os
import argparse
from transformers.integrations import TensorBoardCallback
from typing import Dict
import json
from datetime import datetime
from numpy import random
import pandas as pd
import numpy as np
from encoder.conformer_braindecode import EEGConformer
from torch import manual_seed
import sys
import wandb
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

from utils import cv_split_bci, read_threshold_sub, load_data_EEG2, EarlyStoppingCallback
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(script_path, '../'))
# from batcher.make import make_batcher
from batcher.base import EEGDataset, EEG2Dataset
from decoder.make_decoder import make_decoder
from embedder.make import make_embedder
from trainer.make import make_trainer
from trainer.base import Trainer
from decoder.unembedder import make_unembedder
os.environ["WANDB_DISABLED"] = "true"

current_time = datetime.now()



def train(config: Dict=None) -> Trainer:
    """Model training according to config.
        -> see get_args() below for all command line arguments.
    """
    
    if config is None:
        config = get_config()

    if config['do_train']:
        os.makedirs(config["log_dir"], exist_ok=True)
        
        # Create readme file in the log directory
        #create_readme(config["log_dir"], sys.argv[0], sys.argv[1:])

        #if config["resume_from"]:
         #   create_readme(config["resume_from"], sys.argv[0], sys.argv[1:])

        resume_path = str(config["resume_from"]) if config["resume_from"] is not None else None
        
        if resume_path is not None:
            config_filepath = os.path.join(config["resume_from"], 'train_config.json')

            if os.path.isfile(config_filepath):
                print(f'Loading training config from {config_filepath}')
                with open(config_filepath, 'r') as f:
                    config = json.load(f)
            else:
                with open(config_filepath, 'w') as f:
                    json.dump(config, f, indent=2)
            
            checkpoints = [
                int(p.split('checkpoint-')[1])
                for p in os.listdir(resume_path)
                if 'checkpoint-' in p and os.path.isdir(os.path.join(resume_path, p))
            ]
            for checkpoint in checkpoints:
               create_readme(checkpoint, sys.argv[0], sys.argv[1:])

            last_checkpoint = max(checkpoints)
            print(f'\nResuming training from checkpoint-{last_checkpoint} in {resume_path}')
            config["resume_from"] = os.path.join(resume_path, f'checkpoint-{last_checkpoint}')
        else:
            config_filepath = os.path.join(config["log_dir"], 'train_config.json')
            with open(config_filepath, 'w') as f:
                json.dump(config, f, indent=2)
            config["resume_from"] = None

    assert config["training_style"] in {'CSM', 'CSM_causal', 'decoding'}, f'{config["training_style"]} is not supported.'
    assert config["architecture"] in {'GPT', 'PretrainedGPT2','DistilledGPT2'}, f'{config["architecture"]} is not supported.'
    
    if config['set_seed']:
        random.seed(config["seed"])
        manual_seed(config["seed"])

    # Initialize wandb
    wandb.init(project="neuroGPT_FR1_catFR1_PAL1", config=config)

    if config["training_style"] == 'decoding':
        downstream_path = config["dst_data_path"]
        if config['data']=='EEG':
            all_data = sorted(os.listdir(downstream_path))
        else:
            all_data, val_files, subjects = load_data_EEG2(downstream_path)
        if config['kfold'] is False:
            k_folds = 1  # No k-fold cross-validation, so we just do one iteration
        elif config['kfold'] is None:
            k_folds = len(all_data)
        else:
            k_folds = config['kfold']

        val_results = []
        trn_results = []
        metrics_df = pd.DataFrame(columns=['subject_path', 'val_loss', 'val_acc', 'f1_score', 'roc_auc', 'precision'])

        for i in range(k_folds):
            if config['kfold'] is None:
                 if config['data']=='EEG':
                     train_files = all_data[:i] + all_data[i+1:]
                     val_files = [all_data[i]]
                 else:
                     train_files = all_data[:i] + all_data[i+1:]
                     val_files = [all_data[i]]
                     train_events = val_files[:i] + val_files[i+1:]
                     val_events = [val_files[i]]
            elif config['kfold'] is False:
                if config['data']=='EEG':
                    train_files = all_data[config['fold_i']:]  #all_data[:int(0.75 * len(all_data))]
                    val_files = all_data[:config['fold_i']]  #all_data[int(0.75 * len(all_data)):]
                    train_events = val_files[config['fold_i']:]
                    val_events = val_files[:config['fold_i']]

            else:
                fold_size = len(all_data) // k_folds
                if config['data']=='EEG':
                    val_files = all_data[i*fold_size:(i+1)*fold_size]
                    train_files = all_data[:i*fold_size] + all_data[(i+1)*fold_size:]
                else:
                    val_files = all_data[i*fold_size:(i+1)*fold_size]
                    train_files = all_data[:i*fold_size] + all_data[(i+1)*fold_size:]
                    train_events = val_files[i*fold_size:(i+1)*fold_size]
                    val_events = val_files[:i*fold_size] + all_data[(i+1)*fold_size:]
            if rank == 0:
                
                print('\ndownstream_path : ',downstream_path)
                #print('Train files : ',train_files)
                print('Val_files: ',val_files)

            if config['data']=='EEG':
                train_dataset = EEGDatasetCls(folder_path=downstream_path, files=train_files)
                validation_dataset = EEGDatasetCls(folder_path=downstream_path, files=val_files)
            else:
                train_dataset=EEG2dstDataset(train_files, train_events)
                validation_dataset = EEG2dstDataset(val_files, val_events)

            trainer = make_trainer(
                model_init=lambda: make_model(config),
                training_style=config["training_style"],
                run_name=config["run_name"],
                output_dir=config["log_dir"],
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                per_device_train_batch_size=config["per_device_training_batch_size"],
                per_device_eval_batch_size=config["per_device_validation_batch_size"],
                dataloader_num_workers=config["num_workers"],
                optim=config["optim"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                adam_beta1=config["adam_beta_1"],
                adam_beta2=config["adam_beta_2"],
                adam_epsilon=config["adam_epsilon"],
                max_grad_norm=config["max_grad_norm"],
                lr_scheduler_type=config["lr_scheduler"],
                warmup_ratio=config["warmup_ratio"],
                max_steps=config["training_steps"],
                save_steps=config["log_every_n_steps"],
                logging_steps=config["log_every_n_steps"],
                eval_steps=config["eval_every_n_steps"],
                seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
                fp16=config["fp16"],
                deepspeed=config["deepspeed"],
            )
            trainer.add_callback(TensorBoardCallback())
                    # Add early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 5),
                early_stopping_threshold=config.get("early_stopping_threshold", 0.0001)
            )
            trainer.add_callback(early_stopping_callback)

            if config['do_train']:
                trainer.train(resume_from_checkpoint=config["resume_from"])
                trainer.save_model(os.path.join(config["log_dir"], 'model_final'))
                
            

            val_prediction = trainer.predict(validation_dataset)
            d = config['dst_data_path']
            k = val_files[0]
            #print(val_prediction.metrics)
            #print(val_prediction.metrics.keys())
            pd.DataFrame(val_prediction.metrics, index=[0]).to_csv(
                os.path.join(config["log_dir"], f'val_metrics_fold_{i}.csv'), index=False)
            np.save(os.path.join(config["log_dir"], f'val_predictions_fold_{i}.npy'), val_prediction.predictions)
            np.save(os.path.join(config["log_dir"], f'val_label_ids_fold_{i}.npy'), val_prediction.label_ids)
        
            fold_metrics = {
                'subject_path': k.replace(d,''), 
                'val_loss': val_prediction.metrics['test_loss'],
                'val_acc': val_prediction.metrics['test_accuracy'],
                'f1_score': val_prediction.metrics['test_f1_score'],
                'roc_auc': val_prediction.metrics['test_auc'],
                'precision': val_prediction.metrics['test_precision']
            }

            # Append to DataFrame
            metrics_df = pd.concat([metrics_df, pd.DataFrame([fold_metrics])], ignore_index=True)

            trn_results.append(trainer.state.log_history)
            val_results.append(val_prediction.metrics)

        # Save the aggregated metrics DataFrame to a CSV file
        metrics_df.to_csv(os.path.join(config["log_dir"], 'metrics_log.csv'), index=False)

        # Calculate and print cross-validation scores
        if k_folds > 1:
            avg_metrics = {}
            for metric in val_results[0]:
                metric_values = [fold[metric] for fold in val_results]
                avg_metrics[metric] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values)
                }
            if rank == 0:    
                print("\nCross-Validation Scores:")
                for metric, values in avg_metrics.items():
                    print(f"{metric}: Mean = {values['mean']:.4f}, Std = {values['std']:.4f}")

      #  print("Starting analysis of feature importance...")
      # analyze_feature_importance(trainer.model, validation_dataset)

        return trn_results, val_results

    else:
        # Handling for CSM or CSM_causal
        root_path = config["train_data_path"]
        files = read_threshold_sub(config["sub_list"], lower_bound=1000, upper_bound=1000000)
        random.shuffle(files)
        if config['data'] == 'EEG':
            DatasetClass = EEGDataset
            val_len=1000
        elif config['data'] == 'EEG2':
            DatasetClass = EEG2Dataset
            val_len=1000
        else:
            raise ValueError("Unsupported data type: {}".format(config['data']))
        #train_len = int(len(files) * 0.75)
        train_dataset = DatasetClass(files[val_len:], sample_keys=[
            'inputs',
            'attention_mask'
        ], chunk_len=config["chunk_len"], num_chunks=config["num_chunks"], ovlp=config["chunk_ovlp"], root_path=root_path, gpt_only= not config["use_encoder"], normalization=config["do_normalization"])
        validation_dataset = DatasetClass(files[:val_len], sample_keys=[
            'inputs',
            'attention_mask'
        ], chunk_len=config["chunk_len"], num_chunks=config["num_chunks"], ovlp=config["chunk_ovlp"], root_path=root_path, gpt_only= not config["use_encoder"], normalization=config["do_normalization"])
        test_dataset = None

        trainer = make_trainer(
            model_init=lambda: make_model(config),
            training_style=config["training_style"],
            run_name=config["run_name"],
            output_dir=config["log_dir"],
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            per_device_train_batch_size=config["per_device_training_batch_size"],
            per_device_eval_batch_size=config["per_device_validation_batch_size"],
            dataloader_num_workers=config["num_workers"],
            optim=config["optim"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            adam_beta1=config["adam_beta_1"],
            adam_beta2=config["adam_beta_2"],
            adam_epsilon=config["adam_epsilon"],
            max_grad_norm=config["max_grad_norm"],
            lr_scheduler_type=config["lr_scheduler"],
            warmup_ratio=config["warmup_ratio"],
            max_steps=config["training_steps"] if config["training_type"] == "steps" else -1,
            num_train_epochs=config["num_epochs"] if config["training_type"] == "epoch" else 1,
            save_steps=config["log_every_n_steps"],
            logging_steps=config["log_every_n_steps"],
            eval_steps=config["eval_every_n_steps"],
            evaluation_strategy=config["evaluation_strategy"],
            seed=config["seed"] if config['set_seed'] else np.random.choice(range(1, 100000)),
            fp16=config["fp16"],
            deepspeed=config["deepspeed"],
        )
        trainer.add_callback(TensorBoardCallback())

        if config['do_train']:
            trainer.train(resume_from_checkpoint=config["resume_from"])
            trainer.save_model(os.path.join(config["log_dir"], 'model_final'))
        
        if test_dataset is not None:
            test_prediction = trainer.predict(test_dataset)
            pd.DataFrame(
                test_prediction.metrics,
                index=[0]
            ).to_csv(
                os.path.join(
                    config["log_dir"],
                    'test_metrics.csv'
                ),
                index=False
            )
            np.save(
                os.path.join(
                    config["log_dir"],
                    'test_predictions.npy'
                ),
                test_prediction.predictions
            )
            np.save(
                os.path.join(
                    config["log_dir"],
                    'test_label_ids.npy'
                ),
                test_prediction.label_ids
            )
        wandb.finish()
        return trainer


def make_model(model_config: Dict=None):
    """Make model from model_config 
    (as generated by get_config()).
    """
    if model_config["use_encoder"] == True:
        chann_coords = None
        #print('C B')
        #print(model_config['num_channels'])
        
        encoder = EEGConformer(n_outputs=model_config["num_decoding_classes"], n_chans=model_config['num_channels'], n_times=model_config['chunk_len'], ch_pos=chann_coords, is_decoding_mode=model_config["ft_only_encoder"],data=model_config['data'])
       
        #calculates the output dimension of the encoder, which is the output of transformer layer.
        #model_config["parcellation_dim"] = 1080
        model_config["parcellation_dim"] = ((model_config['chunk_len'] - model_config['filter_time_length'] + 1 - model_config['pool_time_length']) // model_config['stride_avg_pool'] + 1) * model_config['n_filters_time']
        #print('parcellation_dim',model_config["parcellation_dim"]) 

    else:
        encoder = None
        model_config["parcellation_dim"] = model_config["chunk_len"] * model_config['num_channels']

    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"], # flattened, channel x chunk length
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        n_positions=model_config["n_positions"]
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"]
    )
    #decoder.config.gradient_checkpointing = True


    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )
    else:
        print("No Embedder and Unembedder!")
        unembedder = None

   
    from model import Model
    model = Model(
        encoder=encoder,
        embedder=embedder,
        decoder=decoder,
        unembedder=unembedder
    )    

    if model_config["ft_only_encoder"]:
        model.switch_ft_mode(ft_encoder_only=True)

    if model_config["training_style"] == 'decoding':
        model.switch_decoding_mode(
            is_decoding_mode=True,
            num_decoding_classes=model_config["num_decoding_classes"]
        )

    if model_config["pretrained_model"] is not None:
        if model_config["pretrained_model"].lower() == "random":
            weights_seed = model_config.get("weights_seed", None) # Use weights_seed from config if available
            if rank == 0:
                print(f"\nInitializing model with random weights and seed {weights_seed}")
            model.init_weights(seed=weights_seed)
        else:
            model.from_pretrained(model_config["pretrained_model"])

    if model_config["freeze_embedder"]:
        for param in model.embedder.parameters():
            param.requires_grad = False

    if model_config["freeze_decoder"]:
        for param in model.decoder.parameters():
            param.requires_grad = False

    if model_config["freeze_encoder"]:
        for name, param in model.encoder.named_parameters():
            if 'fc.' in name \
            or 'final_layer' in name:
                continue
            else:
                param.requires_grad = False

    if 'freeze_decoder_without_pooler_heads' in model_config \
        and model_config["freeze_decoder_without_pooler_heads"]:
        for name, param in model.decoder.named_parameters():
            if 'pooler_layer' in name \
            or 'decoding_head' in name \
            or 'is_next_head' in name:
                continue
            else:
                param.requires_grad = False

    if model_config["freeze_unembedder"] and unembedder is not None:
        for param in model.unembedder.parameters():
            param.requires_grad = False

    return model



def get_config(args: argparse.Namespace=None) -> Dict:
    """
    Make config from command line arguments (as created by get_args()).
    Performs additional formating of args required for calling train().
    """

    if args is None:
        args = get_args().parse_args()

    if args.smoke_test == "True":
        args.per_device_training_batch_size =  2
        args.per_device_validation_batch_size = 2
        args.training_steps = 2
        args.validation_steps = 2
        args.test_steps = 2
        args.log_every_n_steps = 1

    if args.num_attention_heads == -1:
        assert (
            args.embedding_dim%64
         ) == 0, f'embedding-dim needs be be multiple of 64 (currently: {args.embedding_dim})' 
        args.num_attention_heads = args.embedding_dim//64

    if args.run_name == 'none':
        args.run_name = f'{args.architecture}'

        if args.architecture != 'LinearBaseline':
            
            if 'Pretrained' not in args.architecture:
                args.run_name += f'_lrs-{args.num_hidden_layers}'

                args.run_name += f'_hds-{args.num_attention_heads}'

            # args.run_name += f'_embd-{args.embedding_dim}'
            # args.run_name += f'_train-{args.training_style}'
            # args.run_name += f'_lr-{str(args.learning_rate).replace(".", "")[1:]}'
            # args.run_name += f'_bs-{args.per_device_training_batch_size}'
            # args.run_name += f'_drp-{str(args.dropout).replace(".", "")}'
            args.run_name += f'_ChunkLen-{args.chunk_len}'
            args.run_name += f'_NumChunks-{args.num_chunks}'
            args.run_name += f'_ovlp-{args.chunk_ovlp}'

        else:
            args.run_name += f'_train-{args.training_style}'

        args.run_name += f"_{datetime.now().strftime('%Y-%m-%d_%H')}"

    if args.training_style == "decoding":
        args.run_name += '-' + str(args.fold_i)

    if args.no_evaluation == 'True':
        args.evaluation_strategy = 'no'
    else:
        args.evaluation_strategy = 'steps' if args.training_type == 'steps' else 'epoch'

    if args.smoke_test == "True":
        args.run_name = f'smoke-test_{args.run_name}'

    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.wandb_mode = args.wandb_mode if args.wandb_mode in {'online', 'offline'} and args.local_rank in {-1, 0} else "disabled"
    
    config = vars(args)

    for arg in config:
        
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        
        elif config[arg] == 'none':
            config[arg] = None

        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]
    
    if 'kfold' in config:
        if config['kfold'] == False:
            config['kfold'] = False
        else:
            try:
                config['kfold'] = int(config['kfold'])
                if config['kfold'] == 0:
                    config['kfold'] = None  # This will be interpreted as LOOCV
            except ValueError:
                raise ValueError("Invalid value for --kfold. It must be an integer or 'False'.")


    return config


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )

    # Data pipeline settings:
    parser.add_argument(
        '--train-data-path',
        metavar='DIR',
        default='../../tuh_tensors/',
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )

    parser.add_argument(
        '--dst-data-path',
        metavar='DIR',
        default="../../bci2a_egg_npz/",
        type=str,
        help='path to training data directory '
             '(default: data/upstream)'
    )

    parser.add_argument(
        '--parcellation-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input data parcellation (default: 1024). '
             '! This is fixed for the current up-/downstream data.'
    )
    parser.add_argument(
        '--pretrained-model',
        metavar='DIR',
        type=str,
        default='none',
        help='checkpoint used to initialize model weights '
             '(default: none)'
    )

    # Embedder settings:    
    parser.add_argument(
        '--embedding-dim',
        metavar='INT',
        default=1024,
        type=int,
        help='dimension of input embedding '
             '(default: 1024)'
    )
    parser.add_argument(
        '--num-hidden-layers-embedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of layers of linear embedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-embedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze embedder weights during training '
             '(default: False) '
    )

    # UnEmbedder settings:
    parser.add_argument(
        '--num-hidden-layers-unembedding-model',
        metavar='INT',
        default=1,
        type=int,
        help='numer of hidden layers for linear unembedding model '
             '(default: 1)'
    )
    parser.add_argument(
        '--freeze-unembedder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze unembedder weights during training '
             '(default: False) '
    )


    # Decoder settings:
    parser.add_argument(
        '--architecture',
        metavar='STR',
        default='GPT',
        choices=(
            'GPT',
            'PretrainedGPT2',
            'DistilledGPT2'
        ),
        type=str,
        help='Model architecture used for sequence modeling / decoding. '
             '(default: GPT) '
    )
    parser.add_argument(
        '--num-hidden-layers',
        metavar='INT',
        default=4,
        type=int,
        help='number of hidden model layers in --architecture '
             '(default: 4). '
             '! Does not apply to LinearBaseline; '
             '! Same number of hidden layers is used for decoder / encoder '
             'parts of autoencoder (ie., default creates encoder and decoder '
             'with 4 hidden layers each)'
    )
    parser.add_argument(
        '--num-attention-heads',
        metavar='INT',
        default=-1,
        type=int,
        help='number of attention heads per transformer layer '
             '(default: embedding-dim // 64). '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--intermediate-dim-factor',
        metavar='INT',
        default=4,
        type=int,
        help='scales feed-forward transformer layer dimension relative to '
             'embedding-dim: intermediate-dim-factor * embedding-dim '
             '(default: 4)'
    )
    parser.add_argument(
        '--hidden-activation',
        metavar='STR',
        default='gelu_new',
        choices=(
            'gelu',
            'gelu_new',
            'relu',
            'silu'
        ),
        type=str,
        help='type of hidden activation of transformer layers '
             '(default: gelu_new); '
             'one of {"gelu", "gelu_new", "relu", "silu"}. '
             '! Does not apply to non-transformer models'
    )
    parser.add_argument(
        '--freeze-decoder',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture '
             '(default: False) '
    )
    parser.add_argument(
        '--freeze-decoder-without-pooler-heads',
        metavar='BOOL',
        default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze decoder model weights during training '
             'as specified by --architecture, without pooler layer and '
             ' is-next-pred / decoding heads '
             '(default: False) '
    )


    # Trainer settings:
    parser.add_argument(
        '--resume-from',
        metavar='DIR',
        type=str,
        default='none',
        help='continue training from specified checkpoint '
             '(default: none)'
    )
    parser.add_argument(
        '--training-style',
        metavar='STR',
        default='CSM_causal',
        choices=(
            'CSM',
            'CSM_causal',
            'decoding'
        ),
        type=str,
        help='training framework / style (default: CSM); '
             'one of CSM, decoding'
    )
    parser.add_argument(
        '--decoding-target',
        metavar='STR',
        default='none',
        type=str,
        help='key for decoding target variable in .tar-files in --data'
             '(default: none). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--num-decoding-classes',
        metavar='INT',
        default=4,
        type=int,
        help='number of decoding classes (ie., mental states) in --data '
             '(default: 0). '
             '! Must be specified when setting --training-style to "decoding"'
    )
    parser.add_argument(
        '--training-steps',
        metavar='INT',
        default=60000,
        type=int,
        help='number of training steps to perform '
             '(default: 400000)'
    )
    parser.add_argument(
        '--validation-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of validation steps to perform at evaluation time '
             '(default: 1000)'
    )
    parser.add_argument(
        '--test-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='number of test steps to perform at test time'
             '(default: 2000). '
             '! Test evaluation only performed if test set created by '
             'setting --n-test-subjects-per-dataset != -1'
    )
    parser.add_argument(
        '--per-device-training-batch-size',
        metavar='INT',
        default=16,
        type=int,
        help='batch size during training per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--per-device-validation-batch-size',
        metavar='INT',
        default=16,
        type=int,
        help='batch size during evaluation per training device '
             '(default: 64)'
    )
    parser.add_argument(
        '--optim',
        metavar='STR',
        default='adamw_hf',
        type=str,
        help='optimizer to use for training '
             '(default: adamw_hf) -> adamw from HuggingFrace transformer library. '
             'For other options see Huggingface TrainerArgs.'
    )
    parser.add_argument(
        '--learning-rate',
        metavar='FLOAT',
        default=1e-4,
        type=float,
        help='maximum learning rate during training '
             '(default: 1e-4)'
    )
    parser.add_argument(
        '--warmup-ratio',
        metavar='FLOAT',
        default=0.01,
        type=float,
        help='warm-up steps for linear learning rate scheduler '
             'specified as fraction of --training-steps '
             '(default: 0.01)'
    )
    parser.add_argument(
        '--weight-decay',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='weight decay strength (indicating l2-regularisation strength) '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--adam-beta-1',
        metavar='FLOAT',
        default=0.9,
        type=float,
        help='adam beta 1 (default: 0.9)'
    )
    parser.add_argument(
        '--adam-beta-2',
        metavar='FLOAT',
        default=0.999,
        type=float,
        help='adam beta 2 (default: 0.999)'
    )
    parser.add_argument(
        '--adam-epsilon',
        metavar='FLOAT',
        default=1e-8,
        type=float,
        help='adam beta 2 (default: 1e-8)'
    )
    parser.add_argument(
        '--max-grad-norm',
        metavar='FLOAT',
        default=1.0,
        type=float,
        help='maximum gradient clipping norm (default: 1.0)'
    )
    parser.add_argument(
        '--lr-scheduler',
        metavar='STR',
        default='linear',
        choices=(
            'linear',
            'constant_with_warmup',
            'none'
        ),
        type=str,
        help='learning rate scheduler; '
             'one of {linear, constant_with_warmup, none} '
             '(default: linear)'
    )
    parser.add_argument(
        '--dropout',
        metavar='FLOAT',
        default=0.1,
        type=float,
        help='dropout ratio for hidden layers of embedder and decoder model parts '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--kfold',
        metavar='INT',
        default='False',
        type=str,  # Use string type to allow both integer values and the 'False' string
        help='number of folds for K-fold cross-validation; set to 0 for LOOCV, set to "False" for no cross-validation (default: False)'
    )
  
    # Logging settings:
    parser.add_argument(
        '--log-dir',
        metavar='DIR',
        type=str,
        default='results/models/upstream',
        help='path where training is logged '
             '(default: results/models/upstream)'
    )
    parser.add_argument(
        '--log-every-n-steps',
        metavar='INT',
        default=1000,
        type=int,
        help='frequence of logging in training steps '
             '(default: 10000)'
    )
    parser.add_argument(
        '--run-name',
        metavar='STR',
        type=str,
        default='none',
        help='descriptor of the training run used for logging and wandb; '
             '! if set to "none", a unique identifier is automatically created'
    )
    parser.add_argument(
        '--wandb-mode',
        metavar='STR',
        choices=(
            'online',
            'offline',
            'disabled'
        ),
        default='disabled',
        help='track training w/ wandb online or offline or not at all '
             '(default: disabled) '
             '! requires setting up weights-and-bias for this machine; '
             'see: https://docs.wandb.ai/'
    )
    parser.add_argument(
        '--wandb-project-name',
        metavar='STR',
        type=str,
        default='learning-from-brains',
        help='name of wandb project where data is logged '
             '(default: learning-from-brains)'
    )

    # Other settings:
    parser.add_argument(
        '--seed',
        metavar='INT',
        default=1234,
        type=int,
        help='random seed (default: 1234)'
    )
    parser.add_argument(
        '--set-seed',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        type=str,
        help='whether or not to set random seed (default: True)'
    )
    parser.add_argument(
        '--fp16',
        metavar='BOOL',
        choices=('True', 'False'),
        default='True',
        help='whether or not to use 16-bit precision GPU training '
             '(default: True)'
    )
    parser.add_argument(
        '--deepspeed',
        metavar='DIR',
        default="none",
        type=str,
        help='location of deepspeed configuration file; '
             'automatically adds deepspeed functionality to training if specified '
             '(default: none)'
    )
    parser.add_argument(
        '--local_rank',
        metavar='INT',
        default=-1,
        type=int,
        help='Rank of the process during distributed training '
             '(default: -1)'
    )
    parser.add_argument(
        '--num-workers',
        metavar='INT',
        default=8,
        type=int,
        help='number of data loading workers '
             '(default: 0 -> load in main process)'
    )
    parser.add_argument(
        '--plot-model-graph',
        metavar='BOOL',
        default="False",
        type=str,
        choices=('True', 'False'),
        help='whether or not to save an image of the model graph to log-dir '
             '(default: False)'
    )
    parser.add_argument(
        '--smoke-test',
        metavar='BOOL',
        default="False",
        type=str,
        choices=("True", "False"),
        help='whetehr or not to run training in smoke test-mode '
             '(default: False)'
             'If set to "True", training is restricted by setting: '
             '--per-device-training_batch_size 2 '
             '--per-device-validation_batch_size 2 '
             '--training-steps 2 '
             '--validation-steps 2 '
             '--test-steps 2 '
             '--log-every-n-steps 1'
    )
    parser.add_argument(
        '--bold-dummy-mode',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to replace BOLD with dummy during training; '
             'for internal testing purposes only! '
             '(default: False)'
    )
    parser.add_argument(
        '--do-train',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether or not to run training '
             '(default: True). '
             'If "False", train() still returns trainer'
    )
    
    parser.add_argument(
        '--n-positions',
        metavar='INT',
        default=512,
        type=int,
        help='maximum sequence length that transformer model might ever be used with '
             '(default: 512)'
    )
    ## EEG settings
    parser.add_argument(
        '--chunk_len',
        default=500,
        type=int)
    parser.add_argument(
    '--num_chunks',
    default=2,
    type=int)
    parser.add_argument(
    '--chunk_ovlp',
    default=100,
    type=int)
    parser.add_argument(
    '--sampling_rate',
    default=250,
    type=int)
    parser.add_argument(
    '--fold_i',
    default=0,
    type=int)

    parser.add_argument(
        '--use-encoder',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether to use encoder or not'
    )
    parser.add_argument(
        '--do-normalization',
        metavar='BOOL',
        default='True',
        type=str,
        choices=('True', 'False'),
        help='whether to use encoder or not'
    )

    parser.add_argument('--filter-time-length', metavar='INT', default=25, type=int, help='length of the temporal filter (default: 25)')
    parser.add_argument('--pool-time-length', metavar='INT', default=75, type=int, help='length of temporal pooling filter (default: 75)')
    parser.add_argument('--stride-avg-pool', metavar='INT', default=15, type=int, help='length of stride between temporal pooling filters (default: 15)')
    parser.add_argument('--n-filters-time', metavar='INT', default=40, type=int, help='number of temporal filters (default: 40)')
    parser.add_argument('--num-encoder-layers', metavar='INT', default=6, type=int, help='number of transformer layers in encoder')

    parser.add_argument('--eval_every_n_steps', default=2000, type=int)
    parser.add_argument('--freeze-encoder', metavar='BOOL', default='False',
        choices=('True', 'False'),
        type=str,
        help='whether or not to freeze encoder weights during training '
             '(default: False) '
    )
    parser.add_argument('--ft-only-encoder', metavar='BOOL', default='False',
        choices=('True', 'False'),
        type=str,
        help='finetune with only encoder or not '
             '(default: False) '
    )
    parser.add_argument(
        '--sub_list',
        metavar='PATH',
        default='../inputs/sub_list2.csv',
        type=str,
        help='path to the CSV file containing subject list'
    )
    parser.add_argument('--data',metavar='STR',default='EEG',choices=('EEG','EEG2'),type=str,help='Type of data to use for training (default: EEG)')

    parser.add_argument(
        '--num_channels',
        metavar='INT',
        default=20,
        type=int,
        help='Number of channels in the input data, Specify with EEG2 data. (default: 20)'
    )
    parser.add_argument(
        '--training-type',
        metavar='STR',
        default='steps',
        choices=('steps', 'epoch'),
        type=str,
        help='Specify training type: steps or epochs (default: steps)'
    )
    parser.add_argument(
        '--num-epochs',
        metavar='INT',
        default=80,
        type=int,
        help='Number of epochs to train for if training type is epochs (default: 80)'
    )
    parser.add_argument(
        '--weights-seed',
        metavar='INT',
        type=int,
        default=90,
        help='random seed for initializing model weights (default: 90)'
    )
    parser.add_argument(
        '--no_evaluation',
        metavar='BOOL',
        default='False',
        type=str,
        choices=('True', 'False'),
        help='whether or not to disable evaluation during training (default: False)'
    )
    parser.add_argument(
    '--early-stopping-patience',
    metavar='INT',
    default=5,
    type=int,
    help='Number of evaluations with no improvement after which training will be stopped (default: 5)'
    )
    parser.add_argument(
        '--early-stopping-threshold',
        metavar='FLOAT',
        default=0.0001,
        type=float,
        help='Minimum change in the monitored quantity to qualify as an improvement (default: 0.0001)'
    )


    return parser


if __name__ == '__main__':
    init_distributed_mode()
    rank = 0 #get_rank()

    if rank == 0:
        args = get_args().parse_args()
    
        # Print input arguments
        print("\nDate and time of running:")
        print(f"Script name and path: {os.path.abspath(__file__)}")
        print("\nInput arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
    
        current_time = datetime.now()
        print("\nCurrent Time and Date:")
        print(f"  {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"\nStarting at  {current_time.strftime('%H:%M:%S')}")

        trainer = train()
        current_time = datetime.now()
        print("\nScript ends at: ", current_time.strftime("%H:%M:%S"))
