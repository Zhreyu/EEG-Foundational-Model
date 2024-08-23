import os
import pdb
import mne
import numpy as np
from batcher.base import EEGDataset,EEG2Dataset
import torch
from batcher.base import process_gdf_file
import pandas as pd

class EEGDatasetCls(EEGDataset):
    def __init__(self, folder_path, files=None):
        self.dataframes = []
        self.indices = []
        
        if files is None:
            files = os.listdir(folder_path)
        
        for file in files:
            if file.endswith('.gdf'):
                file_path = os.path.join(folder_path, file)
                df, idx = process_gdf_file(file_path)  
                if df is not None and idx is not None:
                    self.dataframes.append(df)
                    self.indices.extend(idx)
        
        # Combine all dataframes into a single DataFrame
        if self.dataframes:
            self.df = pd.concat(self.dataframes, ignore_index=True)
            self.df.set_index(["person", "epoch"], inplace=True)
        else:
            self.df = pd.DataFrame()

        self.idxs = self.indices

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
    
        # Extract the DataFrame row corresponding to the current index
        current_data = self.df.loc[self.idxs[idx]]
        #print(current_data.head())
        #print(current_data.columns)

        # Extract the 'condition' column and use it as the label
        label = current_data['condition'].unique().astype(int)

        # Exclude the 'condition' and 'time' columns from the input data
        input_columns = current_data.drop(columns=['condition', 'time']).columns
        data = current_data[input_columns].values.astype(np.float64)

        # Convert data to a PyTorch tensor and reshape
        data = torch.tensor(data, dtype=torch.float).transpose(0, 1).unsqueeze(0)  # Reshape to [1, channels, timepoints]

        batch = {
            'inputs': data,
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
        return batch


class EEG2dstDataset(EEG2Dataset):
    def __init__(self, file_paths, labels, sample_keys=None, chunk_len=500, num_chunks=2, ovlp=100, normalization=True):
        super().__init__(
            filenames=file_paths,
            sample_keys=sample_keys,
            chunk_len=chunk_len,
            num_chunks=num_chunks,
            ovlp=ovlp,
            normalization=normalization
        )
        self.data = []
        self.labels = []
        for data_path, label_path in zip(file_paths, labels):
            train_data = np.load(data_path)
            val_data = np.load(label_path)
            selected_rows = train_data[:, :9, :]     
            self.data.append(selected_rows)
            self.labels.append(val_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  
        out = self.preprocess_sample(sample, self.num_chunks, labels=label) # Preprocess it like pretraining data
        # out['labels'] = label.repeat(out['inputs'].shape[0])          
        # print('inputs shape : ', out['inputs'].shape)
        # print('Labels shape : ', out['labels'].shape)
        return out


