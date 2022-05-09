from models.lstm import LSTM
from models.lstm_glove import LSTM_Glove
from models.naive_bayes import Naive_Bayes
from models.svm import SVM
from lib.dataset import Dataset
import numpy as np
import argparse
import os
from constants import MODELS, RESULTS_FOLDER, GLOVE_FILENAME_42B_300D, \
    GLOVE_FILENAME_6B_50D, GLOVE_FILENAME_6B_100D, GLOVE_FILENAME_6B_200D, GLOVE_FILENAME_6B_300D, NAIVE_BAYES_BERNOULLI_NB, NAIVE_BAYES_MULTINOMIAL_NB

def train_lstm(dataset: Dataset):
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

def train_lstm_glove(dataset: Dataset):
    model = LSTM_Glove(
        ds=dataset, 
        batch_size=32, 
        vocab_size=50000,
        max_length = 256,
        lstm_dim=128,
        dropout=0.3,
        learning_rate=2e-3,
        epochs=100,
        early_stop_epochs=10,
        glove_filename=GLOVE_FILENAME_42B_300D)
    history = model.train()
    return history

def naive_bayes(dataset: Dataset, classifier_str: str):
    model = Naive_Bayes(dataset, classifier_str)
    results = model.train()

def svm(dataset: Dataset):
    model = SVM(dataset)
    results = model.train()



"""
Example call: 
python train.py -m lstm
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help='Model to train.')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model

    # create results folder if it does not exist
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # train for different numbers of target genres
    for n_target_genres in range(2, 12+1):
        if not model_name in MODELS:
            print(f"\nModel {model_name} is not implemented ..")
            exit()
        print(f"\nMethod: {model_name}, # genres: {n_target_genres}")

        dataset = Dataset(n_target_genres)
        save_path = os.path.join(RESULTS_FOLDER, f"{model_name}_{n_target_genres}.npy")

        # call desired method
        if model_name == "naive_bayes_bernoulli":
            naive_bayes(dataset, NAIVE_BAYES_BERNOULLI_NB)

        if model_name == "naive_bayes_multinomial":
            naive_bayes(dataset, NAIVE_BAYES_MULTINOMIAL_NB)

        if model_name == "svm":
            svm(dataset)

        if model_name == "lstm":
            history = train_lstm(dataset)
            np.save(save_path, history.history)

        if model_name == "lstm_glove":
            try:
                history = train_lstm_glove(dataset)
                np.save(save_path, history.history)
            except:
                continue
