# Standard library imports
import os
import random
import csv
import numpy as np
import torch
import mne
from tqdm import tqdm

# Function to set the seed for reproducibility
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

seed = 0
seed_everything(seed)

# Directories
root_folder = ''
processed_folder = ''
extension = '.fif'

# Create processed folder if it doesn't exist
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Function to list EDF files
def list_files_with_extension(root_folder, extension):
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(extension):
                matching_files.append(os.path.join(dirpath, filename))
    return matching_files



# Save processed data and log info
def save_and_log_data(raw, filename, save_path, csv_logger):
    data = raw.get_data()
    torch_data = torch.from_numpy(data)
    total_samples = data.shape[1]  # Total number of samples post-resampling
    torch.save(torch_data, os.path.join(save_path, filename + '.pt'))
    csv_logger.writerow([filename + '.pt', total_samples])

# Processing EDF files
Raw_fif_Files = list_files_with_extension(root_folder, extension)

# Open CSV file for logging
with open(os.path.join(processed_folder, 'sequence_info.csv'), mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'time_len'])

    # Process each file with progress monitoring
    for fif_file in tqdm(Raw_fif_Files, desc="Processing files",mininterval=60):
        try:
            preprocessed_data = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
            filename_without_extension = os.path.splitext(os.path.basename(fif_file))[0]
            save_and_log_data(preprocessed_data, filename_without_extension, processed_folder, csv_writer)
        except Exception as e:
            print(f"Error processing {fif_file}: {e}")
