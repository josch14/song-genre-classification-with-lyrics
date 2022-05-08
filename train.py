from models.lstm import LSTM
from lib.dataset import Dataset
import numpy as np
import argparse
import os
from constants import MODELS, RESULTS_FOLDER

def lstm(n_target_genres: int):
    dataset = Dataset(n_target_genres)
    model = LSTM(
        ds=dataset, 
        batch_size=32, 
        vocab_size=10000,
        embedding_dim=64,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=100,
        early_stop_epochs=10)
    history = model.train()
    return history


"""
Example call: 
python train.py -m lstm
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help='Model to train.')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    if not model_name in MODELS:
        print(f"Model {model_name} is not implemented ..")

    # create results folder if it does not exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # train for different numbers of target genres
    for n_target_genres in range(3, 12+1):
        save_path = os.path.join(RESULTS_FOLDER, f"{model_name}_{n_target_genres}.npy")

        if model_name == "lstm":
            history = lstm(n_target_genres)
            np.save(save_path, history.history)