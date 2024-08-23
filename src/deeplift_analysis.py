import argparse
import torch
from torch.utils.data import DataLoader
from captum.attr import DeepLift
from batcher.downstream_dataset import EEGDatasetCls, EEG2DatasetCls
from encoder.conformer_braindecode import EEGConformer
from embedder.make import make_embedder
from decoder.make_decoder import make_decoder
from decoder.unembedder import make_unembedder
from trainer.make import make_trainer
from trainer.base import Trainer
from transformers import TrainingArguments, Trainer as HFTrainer
from transformers.integrations import TensorBoardCallback
import os
import numpy as np
from datetime import datetime
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_model
from typing import Dict
import warnings

from utils import load_data_eeg2

class Model(torch.nn.Module):
    def __init__(self, encoder, embedder, decoder, unembedder=None):
        super().__init__()
        self.name = f'Embedder-{embedder.name}_Decoder-{decoder.name}'
        self.encoder = encoder
        self.embedder = embedder
        self.decoder = decoder
        self.unembedder = unembedder
        self.is_decoding_mode = False
        self.ft_only_encoder = False

    def from_pretrained(self, pretrained_path: str) -> None:
        print(f'Loading pretrained model from {pretrained_path}')
        file_ext = os.path.splitext(pretrained_path)[1]
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        if file_ext == '.pt' or file_ext == '.bin':
            pretrained = torch.load(pretrained_path, map_location=torch.device(device))
        elif file_ext == '.safetensors':
            load_model(self, pretrained_path, strict=False)
        else:
            raise ValueError("Unsupported file format. Expected '.pt' or '.safetensors'.")
        print('Pretrained model loaded successfully.')

    def switch_ft_mode(self, ft_encoder_only=False):
        self.ft_only_encoder = ft_encoder_only

    def switch_decoding_mode(self, is_decoding_mode=False, num_decoding_classes=None) -> None:
        self.is_decoding_mode = is_decoding_mode
        self.embedder.switch_decoding_mode(is_decoding_mode=is_decoding_mode)
        self.decoder.switch_decoding_mode(is_decoding_mode=is_decoding_mode, num_decoding_classes=num_decoding_classes)

    def compute_loss(self, batch: Dict[str, torch.tensor], return_outputs: bool = False) -> Dict[str, torch.tensor]:
        (outputs, batch) = self.forward(batch=batch, return_batch=True)
        losses = self.embedder.loss(batch=batch, outputs=outputs)
        return (losses, outputs) if return_outputs else losses

    def prep_batch(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        return self.embedder.prep_batch(batch=dict(batch))

    def forward(self, batch: Dict[str, torch.tensor], prep_batch: bool = True, return_batch: bool = False) -> torch.tensor:
        if self.encoder is not None:
            inputs = batch['inputs']
            features = self.encoder(inputs)
            if self.is_decoding_mode and self.ft_only_encoder:
                outputs = {'outputs': features, 'decoding_logits': features}
                return (outputs, batch) if return_batch else outputs
            b, f1, f2 = features.size()
            nchunks = inputs.size()[1]
            batch['inputs'] = features.view(b // nchunks, nchunks, f1 * f2)
        if prep_batch:
            if len(batch['inputs'].size()) > 3:
                bsize, chunk, chann, time = batch['inputs'].size()
                batch['inputs'] = batch['inputs'].view(bsize, chunk, chann * time)
            batch = self.prep_batch(batch=batch)
        else:
            assert 'inputs_embeds' in batch, 'inputs_embeds not in batch'
        batch['inputs_embeds'] = self.embedder(batch=batch)
        outputs = self.decoder(batch=batch)
        if self.unembedder is not None and not self.is_decoding_mode:
            outputs['outputs'] = self.unembedder(inputs=outputs['outputs'])['outputs']
        return (outputs, batch) if return_batch else outputs

class DeepLiftModel(torch.nn.Module):
    def __init__(self, model):
        super(DeepLiftModel, self).__init__()
        self.model = model

    def forward(self, inputs):
        batch = {'inputs': inputs}
        outputs = self.model.forward(batch)
        if isinstance(outputs, tuple):
            outputs = outputs[0]['outputs']
        else:
            outputs = outputs['outputs']
        return outputs

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

def filter_channels(inputs, threshold=1e-6):
    channel_variances = np.var(inputs, axis=(0, 1, 3))
    significant_mask = channel_variances > threshold
    return inputs[:, :, significant_mask, :], significant_mask

def plot_eeg_signals(inputs, channel_names, title,num_samples):
    #num_samples = 1537
    time_axis = np.linspace(0, 60, num_samples)  # Ensure the x-axis is scaled to 60 seconds
    fig, axs = plt.subplots(inputs.shape[2], 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title, fontsize=16)
    for ch in range(inputs.shape[2]):
        signal = inputs[0, 0, ch, :]
        ax = axs[ch]
        ax.plot(time_axis, signal, color='black', linewidth=0.5)
        ax.set_ylabel(channel_names[ch], rotation=0, labelpad=40, va='center')
        ax.set_yticks([])
        if ch == inputs.shape[2] - 1:
            ax.set_xlabel('Time (s)')
            ax.set_xticks(np.arange(0, 61, 10))
            ax.set_xticklabels([str(x) for x in range(0, 61, 10)])
        else:
            ax.set_xticks([])
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()

def visualize_importance_heatmap(attributions, inputs, channel_names, sampling_rate=256):
    inputs = inputs.numpy()
    a,b,c,num_samples = inputs.shape
    if c==10:
        channel_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ']
    filtered_inputs, significant_mask = filter_channels(inputs)
    filtered_channel_names = np.array(channel_names)[significant_mask]
    if filtered_inputs.shape[2] == 0:
        print("No significant channels found. Please check the variance threshold or the input data.")
        return
    mean_attributions = np.mean(attributions, axis=(0, 1))
    filtered_attributions = mean_attributions[significant_mask, :]
    plot_eeg_signals(filtered_inputs, filtered_channel_names, 'Raw EEG Signals',num_samples)
    time_axis = np.linspace(0, 60, num_samples)  # Ensure the x-axis is scaled to 60 seconds    
    fig, axs = plt.subplots(filtered_inputs.shape[2], 1, figsize=(15, 10), sharex=True)
    fig.suptitle('EEG Signals with Highlighted Important Regions', fontsize=16)
    for ch in range(filtered_inputs.shape[2]):
        signal = filtered_inputs[0, 0, ch, :]
        contrib = filtered_attributions[ch, :]
        ax = axs[ch]
        ax.plot(time_axis, signal, color='black', linewidth=0.5)
        contrib_img = ax.imshow(contrib[np.newaxis, :], aspect='auto', extent=[0, 61, np.min(signal), np.max(signal)], cmap='binary', alpha=0.6, interpolation='nearest')
        ax.set_yticks([])
        ax.set_ylabel(filtered_channel_names[ch], rotation=0, labelpad=40, va='center')
        if ch == filtered_inputs.shape[2] - 1:
            ax.set_xlabel('Time (s)')
            ax.set_xticks(np.arange(0, 61, 10))
            ax.set_xticklabels([str(x) for x in range(0, 61, 10)])
        else:
            ax.set_xticks([])
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(contrib_img, cax=cbar_ax, label='Contribution')
    plt.show()

def analyze_feature_importance(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    data_loader = DataLoader(dataset, batch_size=11, shuffle=False, num_workers=0, pin_memory=True)
    deep_lift = DeepLift(DeepLiftModel(model))
    all_attributions = []
    for batch in data_loader:
        inputs = batch['inputs']
        labels = batch['labels']
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(inputs.shape[0]):
                attributions = deep_lift.attribute(inputs[i:i+1], target=labels[i].item())
                all_attributions.append(attributions.cpu().detach().numpy())
    all_attributions = np.concatenate(all_attributions, axis=0)
    print("All attributions shape:", all_attributions.shape)
    channel_names = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'Comp']
    visualize_importance_heatmap(all_attributions, inputs.cpu().detach(), channel_names)

def get_args():
    parser = argparse.ArgumentParser(description='Train and analyze model')
    parser.add_argument('--training-style', type=str, default='decoding')
    parser.add_argument('--num-decoding-classes', type=int, default=4)
    parser.add_argument('--training-steps', type=int, default=20)
    parser.add_argument('--eval-every-n-steps', type=int, default=10)
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    parser.add_argument('--num-chunks', type=int, default=2)
    parser.add_argument('--per-device-training-batch-size', type=int, default=2)
    parser.add_argument('--per-device-validation-batch-size', type=int, default=2)
    parser.add_argument('--chunk-len', type=int, default=500)
    parser.add_argument('--chunk-ovlp', type=int, default=0)
    parser.add_argument('--run-name', type=str, default='dst_our')
    parser.add_argument('--ft-only-encoder', type=str, default='True')
    parser.add_argument('--fold_i', type=int, default=0)
    parser.add_argument('--num-encoder-layers', type=int, default=6)
    parser.add_argument('--num-hidden-layers', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--use-encoder', type=str, default='True')
    parser.add_argument('--embedding-dim', type=int, default=1024)
    parser.add_argument('--pretrained-model', type=str, default='pytorch_model.bin')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--dst-data-path', type=str, default="C:/Users/shreyas/Documents/GitHub/Archives/NeuroGPT/train/")
    parser.add_argument('--kfold', type=str, default='False')
    parser.add_argument('--fp16', type=str, default='False')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--do_train', type=str, default='True')
    parser.add_argument('--data', type=str, default='EEG', choices=['EEG', 'EEG2'])
    return parser.parse_args()

def make_model(config):
    num_channels = 20 if config.data == 'EEG' else 10
    filter_time_length = 25
    pool_time_length = 75
    stride_avg_pool = 15
    n_filters_time = 40
    n_positions = 512
    num_hidden_layers_embedding_model = 1
    dropout = 0.1
    architecture = 'GPT'
    intermediate_dim_factor = 4
    hidden_activation = 'gelu_new'
    num_hidden_layers_unembedding_model = 1
    num_attention_heads = 16
    if config.use_encoder == 'True':
        encoder = EEGConformer(n_outputs=config.num_decoding_classes, n_chans=num_channels, n_times=config.chunk_len, is_decoding_mode=config.ft_only_encoder == 'True',data=config.data)
        config.parcellation_dim = ((config.chunk_len - filter_time_length + 1 - pool_time_length) // stride_avg_pool + 1) * n_filters_time
    else:
        encoder = None
        config.parcellation_dim = config.chunk_len * num_channels
    embedder = make_embedder(training_style=config.training_style, architecture=architecture, in_dim=config.parcellation_dim, embed_dim=config.embedding_dim, num_hidden_layers=num_hidden_layers_embedding_model, dropout=dropout, n_positions=n_positions)
    decoder = make_decoder(architecture=architecture, num_hidden_layers=config.num_hidden_layers, embed_dim=config.embedding_dim, num_attention_heads=num_attention_heads, n_positions=n_positions, intermediate_dim_factor=intermediate_dim_factor, hidden_activation=hidden_activation, dropout=dropout)
    if config.embedding_dim != config.parcellation_dim:
        unembedder = make_unembedder(embed_dim=config.embedding_dim, num_hidden_layers=num_hidden_layers_unembedding_model, out_dim=config.parcellation_dim, dropout=dropout)
    else:
        unembedder = None
    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder)
    if config.ft_only_encoder == 'True':
        model.switch_ft_mode(ft_encoder_only=True)
    if config.training_style == 'decoding':
        model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=config.num_decoding_classes)
    if config.pretrained_model is not None:
        model.from_pretrained(config.pretrained_model)
    return model

def main():
    args = get_args()
    if args.data == 'EEG':
        train_dataset = EEGDatasetCls(args.dst_data_path)
    elif args.data == 'EEG2':
       train_files, val_files, subjects = load_data_eeg2(args.dst_data_path)
        train_dataset = EEG2DatasetCls(train_files, val_files)
    model = make_model(args)
    model.to(torch.device("cuda"))
    output_dir = os.path.join(os.getcwd(), args.run_name)
    os.makedirs(output_dir, exist_ok=True)
    model.from_pretrained(pretrained_path='eeg_finetuned.safetensors')
    print("Model loaded successfully:")
    model.eval()
    print("Analyzing feature importance...")
    analyze_feature_importance(model, train_dataset)

if __name__ == '__main__':
    main()
