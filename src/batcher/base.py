#!/usr/bin/env python3
from rich import print
from typing import Dict
import numpy as np
# import webdataset as wds
import torch
# import gzip
# import pickle
import mne
import h5py
import os
# import webdataset as wds

from torch.utils.data import Dataset
ds_max, ds_min = 100, -100
def _pad_seq_right_to_n(
    seq: np.ndarray,
    n: int,
    pad_value: float = 0.
    ) -> np.ndarray:
    if n == seq.shape[0]:
        return seq
    return np.concatenate(
        [
            seq,
            np.ones(
                (
                    n-seq.shape[0],
                    *seq.shape[1:]
                )
            ) * pad_value,  
        ],
        axis=0,
    )

ch_map = {
    "EEG-Fz": "FZ",
    "EEG-0": "FC3",
    "EEG-1": "FC1",
    "EEG-2": "FCZ",
    "EEG-3": "FC2",
    "EEG-4": "FC4",
    "EEG-5": "C5",
    "EEG-C3": "C3",
    "EEG-6": "C1",
    "EEG-Cz": "CZ",
    "EEG-7": "C2",
    "EEG-C4": "C4",
    "EEG-8": "C6",
    "EEG-9": "CP3",
    "EEG-10": "CP1",
    "EEG-11": "CPZ",
    "EEG-12": "CP2",
    "EEG-13": "CP4",
    "EEG-14": "P1",
    "EEG-Pz": "PZ",
    "EEG-15": "P2",
    "EEG-16": "POZ",
}
ch_list = [
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "T3",
    "C3",
    "CZ",
    "C4",
    "T4",
    "T5",
    "P3",
    "PZ",
    "P4",
    "T6",
    "O1",
    "O2",
]

keys_with_values_in_list = [key for key, value in ch_map.items() if value in ch_list]

def scaler(x):
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Check if the array is empty
    if x.size == 0:
        raise ValueError("Input array must not be empty.")

    # Calculate min and max
    x_min = np.min(x)
    x_max = np.max(x)

    # Check for division by zero
    if x_max == x_min:
        x_scaled = x / x_max if x_max != 0 else np.zeros_like(x)
        return x_scaled

    # Perform scaling
    x_std = (x - x_min) / (x_max - x_min)
    x_scaled = (x_std * 2) - 1

    return x_scaled

def process_file(raw, ch_map, ch_list, ds_max=100, ds_min=-100):
    # selects 19 standard channels and adds a 20th
    raw = raw.copy()
    try:
        raw = raw.pick(ch_list)
    except ValueError as v:
        pl = v.args[0].split("[")[1].split("]")[0].split(",")
        pl = [p.strip(" ' ") for p in pl]
        new_pick = list(set(ch_list) - set(pl))
        raw = raw.pick(new_pick)

    if len(raw.ch_names) != len(ch_list):
        missing_channels = [ch for ch in ch_list if ch not in raw.ch_names]

        new_channel_data = np.vstack(
            [np.full((1, raw.n_times), 0)] * len(missing_channels)
        )
        new_channel_info = mne.create_info(
            ch_names=missing_channels,
            sfreq=raw.info["sfreq"],
            ch_types=["eeg"] * len(missing_channels),
        )
        new_channel_raw = mne.io.RawArray(
            data=new_channel_data, info=new_channel_info, first_samp=raw.first_samp
        )
        raw.load_data().add_channels([new_channel_raw], force_update_info=True)

    try:
        # raw = raw.rename_channels(ch_map)
        raw = raw.reorder_channels(ch_list)
    except Exception as e:
        print(f"\nError in renaming or reordering channels: {e}")
        return None

    # scale
    trial_min = np.min(raw.get_data())
    trial_max = np.max(raw.get_data())
    raw = raw.load_data().apply_function(scaler, channel_wise=False)

    # add compensation channel
    compensation = (trial_max - trial_min) / (ds_max - ds_min)
    comp_ch_data = np.full((1, raw.n_times), compensation)
    comp_ch_info = mne.create_info(
        ch_names=["compensation"], sfreq=raw.info["sfreq"], ch_types="misc"
    )
    comp_ch_raw = mne.io.RawArray(
        data=comp_ch_data, info=comp_ch_info, first_samp=raw.first_samp
    )
    raw.add_channels([comp_ch_raw], force_update_info=True)

    return raw


class EEGDataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=34, ovlp=100, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(root_path+fn)]
            self.root_path = root_path
            
        print("\nNumber of subjects loaded: ", len(self.filenames))
        #print("Filenames loaded:", self.filenames)
        
        # self.data = data_all
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only=gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        #print(f"Dataset Length: {len(self.filenames)}")
        return len(self.filenames)

    def __getitem__(self, idx):
        #print(f"Getting item: {idx}")
        data = self.load_tensor(self.filenames[idx])
        #===reorder channels====
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.num_chunks)

    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        return _pad_seq_right_to_n(
            seq=seq,
            n=n,
            pad_value=pad_value
        )

    def load_single_file(self, filename):
        with h5py.File(filename, 'r') as file:
            data_dict = file['Result']
            data = []
            for i in range(data_dict['data'].shape[0]):  
                ref = data_dict['data'][i][0]
                time_series = data_dict[ref]
                if len(data) > 0 and time_series.shape[0] < data[0].shape[0]:
                    time_series = np.zeros_like(data[0])
                data.append(np.array(time_series).squeeze())
        return data

    def load_tensor(self, filename):
        # tensor_fn = filename[:-3] + 'pt'
        #print(f"Attempting to load file: {filename}")
        tensor_data = torch.load(filename)
       # print(f"Loaded file successfully: {filename}")
        return tensor_data.numpy()

    def reorder_channels(self, data):
    # Updated channel labels with 'T1' and 'T2' removed
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'FZ': 16, 'CZ': 17, 'PZ': 18, 'OZ': 19}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T3': 7, 'C3': 8, 'CZ': 9, 'C4': 10, 'T4': 11, 'T5': 12, 'P3': 13, 'PZ': 14, 'P4': 15, 'T6': 16, 'O1': 17, 'OZ': 18, 'O2': 19}

        # Adjust the array size to 20 channels
        reordered = np.zeros((20, data.shape[1]))
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered


    def split_chunks(self, data, length=500, ovlp=100, num_chunks=34, start_point=-1): 
        '''2 seconds, 0.2 seconds overlap'''
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        is_numpy = isinstance(data, np.ndarray)  # Check if data is a numpy array
    
        if is_numpy:
            data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor for processing
    
        # Calculate mean and standard deviation using PyTorch
        mean = torch.mean(data, dim=-1, keepdim=True)
        std = torch.std(data, dim=-1, keepdim=True)
        
        # Normalize the data
        normalized_data = (data - mean) / (std + 1e-25)
    
        # Convert back to numpy array if the original input was a numpy array
        if is_numpy:
            normalized_data = normalized_data.numpy()
    
        return normalized_data

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) -> Dict[str, torch.Tensor]:
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len, self.start_samp_pnt)

        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = self._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        if self.gpt_only == True:
            chunks = np.reshape(chunks, (seq_len, chunks.shape[1]*chunks.shape[2]))
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out

class EEG2Dataset(Dataset):
    def __init__(self, filenames, sample_keys, chunk_len=500, num_chunks=34, ovlp=100, root_path="", population_mean=0, population_std=1, gpt_only=False, normalization=True, start_samp_pnt=-1):
        if root_path == "":
            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(root_path+fn)]
            self.root_path = root_path
            
        print("\nNumber of subjects loaded: ", len(self.filenames))
        #print("Filenames loaded:", self.filenames)
        
        self.chunk_len = chunk_len
        self.num_chunks = num_chunks
        self.ovlp = ovlp
        self.sample_keys = sample_keys
        self.mean = population_mean
        self.std = population_std
        self.do_normalization = normalization
        self.gpt_only = gpt_only
        self.start_samp_pnt = start_samp_pnt

    def __len__(self):
        #print(f"Dataset Length: {len(self.filenames)}")
        return len(self.filenames)

    def __getitem__(self, idx):
        #print(f"Getting item: {idx}")
        data = self.load_tensor(self.filenames[idx])
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.num_chunks)

    @staticmethod
    def _pad_seq_right_to_n(seq: np.ndarray, n: int, pad_value: float = 0) -> np.ndarray:
        return _pad_seq_right_to_n(seq=seq, n=n, pad_value=pad_value)

    def load_single_file(self, filename):
        with h5py.File(filename, 'r') as file:
            data_dict = file['Result']
            data = []
            for i in range(data_dict['data'].shape[0]):
                ref = data_dict['data'][i][0]
                time_series = data_dict[ref]
                if len(data) > 0 and time_series.shape[0] < data[0].shape[0]:
                    time_series = np.zeros_like(data[0])
                data.append(np.array(time_series).squeeze())
        return data

    def load_tensor(self, filename):
        #print(f"Attempting to load file: {filename}")
        tensor_data = torch.load(filename)
        #print(f"Loaded file successfully: {filename}")
        return tensor_data.numpy()

    def reorder_channels(self, data):
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9}
        # No changes in the channel order for this test dataset
        reordered = np.zeros((10, data.shape[1]))
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered

    def split_chunks(self, data, length=500, ovlp=100, num_chunks=34, start_point=-1):
        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        is_numpy = isinstance(data, np.ndarray)  # Check if data is a numpy array
    
        if is_numpy:
            data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor for processing
    
        # Calculate mean and standard deviation using PyTorch
        mean = torch.mean(data, dim=-1, keepdim=True)
        std = torch.std(data, dim=-1, keepdim=True)
        
        # Normalize the data
        normalized_data = (data - mean) / (std + 1e-25)
    
        # Convert back to numpy array if the original input was a numpy array
        if is_numpy:
            normalized_data = normalized_data.numpy()
    
        return normalized_data

    def split_chunk_auto(self, data, length=500, ovlp=100, start_point=0):
        all_chunks = []
        start_points = []
        total_len = data.shape[1]
        
        # Calculate the effective stride (distance between start of each chunk)
        stride = length - ovlp
        
        # Calculate how many chunks we can create
        num_chunks = max(1, (total_len - length) // stride + 1)
        
        for i in range(num_chunks):
            chunk_start = start_point + i * stride
            chunk_end = chunk_start + length
            
            # Store the start point of this chunk
            start_points.append(chunk_start)
            
            # Handle case where last chunk might exceed data length
            if chunk_end > total_len:
                chunk_end = total_len
            
            chunk = data[:, chunk_start:chunk_end]
            
            # Pad the last chunk if it's smaller than the specified length
            if chunk.shape[1] < length:
                print('Padding chunk')
                pad_width = ((0, 0), (0, length - chunk.shape[1]))
                chunk = np.pad(chunk, pad_width, mode='constant', constant_values=0)
            
            all_chunks.append(np.array(chunk))
        
        return np.array(all_chunks), start_points, num_chunks
    
    
    def preprocess_sample(self, sample, seq_len, labels=None):
        out = {}
        # print('Sample shape: ', sample.shape)
        if self.do_normalization:
            sample = self.normalize(sample)
        chunks, seq_on, num_chunks = self.split_chunk_auto(sample, self.chunk_len, self.ovlp)
        seq_len = num_chunks
        # print('chunks start points :', seq_on)
        # print('Seq_len : ', seq_len)
        
        attention_mask = torch.ones(seq_len)
        chunks = self._pad_seq_right_to_n(seq=chunks, n=seq_len, pad_value=0)
        attention_mask = self._pad_seq_right_to_n(seq=attention_mask, n=seq_len, pad_value=0)
        
        if self.gpt_only:
            chunks = chunks.reshape(seq_len, -1)
        
        out["inputs"] = torch.from_numpy(chunks).float()
        out["attention_mask"] = attention_mask.long()
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {key: out[key] for key in self.sample_keys if key in out}
        
        if labels is not None:
            out['labels'] = labels
           # print('Labels shape after expand in preprocess sample:', labels.shape)
    
       # print('Encoder input shape prepped in dataloader : ', out["inputs"].shape)
        return out
