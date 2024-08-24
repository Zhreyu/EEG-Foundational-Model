# EEG Foundation Model

## Google Summer of Code 2024 Project Under Emory University

This repository contains the codebase for the foundational EEG model developed during the Google Summer of Code (GSoC) 2024, with the Department of Biomedical Informatics at Emory University.

## Overview

The primary objective of this project was to enhance the NeuroGPT codebase, a framework for processing and analyzing EEG data. Key enhancements include:

- **Data Loaders and Model Architecture**: Adaptations for handling EEG data, especially in `.edf` and `.fif` formats.
- **SafeTensors Support**: Added support for loading models in SafeTensors format, particularly useful in distributed environments.
- **DistilledGPT Architecture**: Integration of a lightweight variant of GPT-2 for reduced computational load and faster training.
- **DeepLIFT Implementation**: Added to improve model interpretability by identifying which EEG channels contributed most to the model's decisions.
- **Early Stopping Callback**: Implemented to prevent overfitting by halting training when performance improvements stagnate.
- **LOOCV and Kfold Cross Validaiton**: Implemented for proper and flexible cross validation of the model.  

## How to Use

To set up the project locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/zhreyu/eeg-foundation-model.git
    cd eeg-foundation-model
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Model**:
    
```
torchrun --nproc_per_node=<num_gpu> src/train_gpt.py --training-type=epoch --num-epochs=25 --log-every-n-steps=10000 --per-device-training-batch-size=8 --per-device-validation-batch-size=8 --num-workers=0 --num_chunks=2 --chunk_len=500 --chunk_ovlp=100 --num-hidden-layers=6 --num-encoder-layers=6 --training-style='CSM_causal' --embedding-dim=1024 --data=EEG --num_channels=20 --train-data-path='' --fp16=True --sub_list='sub_list.csv' --architecture=DistilledGPT2 --no_evaluation=True
```

### Command-Line Arguments

- `--data`: Type of data to use. Options are `EEG` (default) or `IEEG`.
- `--architecture`: Select the architecture to use. Options include `GPT`, `PretrainedGPT2`, and `DistilledGPT2`.
- `--kfold`: Set to `false` for regular training, `0` for LOOCV, or any integer for k-fold cross-validation.
- `--training-type`: Specify the training type. Options are `steps` (default) or `epoch`.
    - If `--training-type=epoch`, use the `--num-epochs` argument to specify the number of epochs.
- `--num-epochs`: Number of epochs to train when `--training-type=epoch` is selected (default is `3`).
- `--sublist`: Path to the CSV file containing the subject list for training.
- `--no_evaluation`: Set to `True` to disable evaluation during training (default is `False`).
- `--early-stopping-patience`: Number of evaluations with no improvement before stopping training (default is `5`).
- `--early-stopping-threshold`: Minimum change in monitored quantity to qualify as improvement (default is `0.00`).

## Acknowledgments

This project builds upon the [NeuroGPT codebase](https://github.com/wenhui0206/NeuroGPT) developed by Wenhui Cui, Woojae Jeong, Philipp Thölke, Takfarinas Medani, Karim Jerbi, Anand A. Joshi, and Richard M. Leahy. Their contributions have been instrumental to the success of this project.

For more details, refer to their [arXiv paper](https://arxiv.org/abs/2311.03764).

## Conclusion

I’ve had a great time during these 2-3 months working on this project. The experience has been incredibly rewarding, and I’ve gained a lot of valuable knowledge and skills. Special thanks to my mentor, Dr. Mahmoud Zeydabadinezhad, for his constant support and guidance throughout GSoC 2024. 

## License

This project is licensed under the BSD 3-Clause "New" or "Revised" License. See the `LICENSE` file for more details.

