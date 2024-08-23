import os
import pdb
import shutil

import h5py
import numpy as np
import gzip
import pickle
import time
import pandas as pd
from transformers import TrainerCallback
import numpy as np

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.0001):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_score = None
        self.early_stopping_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_acc = metrics.get("eval_accuracy")
        if eval_acc is not None:
            if self.best_score is None:
                self.best_score = eval_acc
            elif eval_acc > self.best_score + self.early_stopping_threshold:
                self.best_score = eval_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping_patience:
                    control.should_training_stop = True
                    print(f"\nEarly stopping triggered after {self.early_stopping_counter} evaluations without improvement.")

def get_subject_path_eeg2(base_dir):
    subject_paths = []
    walk_list = list(os.walk(base_dir))
    sorted_walk_list = sorted(walk_list, key=lambda x: x[0])

    for root, dirs, files in sorted_walk_list:
        if 'TrainData.npy' in files and 'val.npy' in files:
            subject_paths.append(root)
    return subject_paths

def load_data_eeg2(base_dir):
    subject_paths = get_subject_path_eeg2(base_dir)
    Train_files = [os.path.join(path, 'TrainData.npy') for path in subject_paths]
    val_files = [os.path.join(path, 'val.npy') for path in subject_paths]
    subjects = [os.path.basename(path).split('_')[0] for path in subject_paths]
    print(subjects)
    return Train_files, val_files, subjects

def load_tuh_all(path):
    # files = os.listdir(path)
    filepath = []
    file=""
    # for file in files:
    groups = os.listdir(path)
    for group in groups:
        if os.path.isdir(os.path.join(path, group)):
            subs = os.listdir(os.path.join(path, file, group))
        else:
            continue
        for sub in subs:
            sessions = os.listdir(os.path.join(path, file, group, sub))
            for sess in sessions:
                montages = os.listdir(os.path.join(path, file, group, sub, sess))
                for mont in montages:
                    edf_files = os.listdir(os.path.join(path, file, group, sub, sess, mont))
                    for edf in edf_files:
                        full_path = os.path.join(path, file, group, sub, sess, mont, edf)
                        filepath.append(full_path)
                        # pdb.set_trace()
                        shutil.move(full_path, os.path.join(path, group, sess + "_" + mont + "_" + edf))
                        # pdb.set_trace()
                # load_eeg(filepath[-1])
    return filepath


def load_pickle(filename):
    start_time = time.time()
    with gzip.open(filename, "rb") as file:
        data = pickle.load(file)
    print(data)
    end_time = time.time()
    print("Compressed Elapsed time:", end_time - start_time, "seconds")
    
    return data['data'], np.array(data['channel'])
  

def read_threshold_sub(csv_file, lower_bound=2599, upper_bound=1000000):
    df_read = pd.read_csv(csv_file)
    # Access the list of filenames and time_len
    filenames = df_read['filename'].tolist()
    time_lens = df_read['time_len'].tolist()
    filtered_files = []
    for fn, tlen in zip(filenames, time_lens):
        if (tlen > lower_bound) and (tlen < upper_bound):
            filtered_files.append(fn)
    return filtered_files

def get_epi_files(path, epi_csv, nonepi_csv, lower_bound=2599, upper_bound=1000000):
    epi_full_path = []
    nonepi_full_path = []
    if epi_csv is not None:
        epi_filtered_files = read_threshold_sub(epi_csv, lower_bound, upper_bound)
        epi_full_path = [path + "/epilepsy_edf/" + fn for fn in epi_filtered_files]
    if nonepi_csv is not None:
        nonepi_filtered_files = read_threshold_sub(nonepi_csv, lower_bound, upper_bound)
        nonepi_full_path = [path + "/no_epilepsy_edf/" + fn for fn in nonepi_filtered_files]

    return epi_full_path + nonepi_full_path

def read_sub_list(epi_list):
    with open(epi_list, 'r') as file:
        items = file.readlines()
    # Remove newline characters
    epi_subs = [item.strip() for item in items]
    return epi_subs

def exclude_epi_subs(csv_file, epi_list, lower_bound=2599, upper_bound=1000000, files_all=None):
    epi_subs = read_sub_list(epi_list)
    group_epi_subs = epi_subs
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in group_epi_subs)]
    # pdb.set_trace()
    return filtered_files

def exclude_sz_subs(csv_file, lower_bound=2599, upper_bound=1000000, files_all=None):
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    with open('sz_subs.txt', 'r') as f:
        sz_subs = f.readlines()
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in sz_subs)]
    # pdb.set_trace()
    return filtered_files        

def cv_split_bci(filenames):
    train_folds = []
    val_folds = []
    for i in range(9):
        train_files = filenames[0:i*2] + filenames[i*2+2:]
        validation_files = filenames[i*2 : i*2+2]
        train_folds.append(train_files)
        val_folds.append(validation_files)
    return train_folds, val_folds
